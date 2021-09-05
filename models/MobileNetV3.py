import tensorflow as tf
import tensorflow.keras as nn
from .SeNet import SEBlock
from .layers import get_activation_layer
from utils.registry import register_model

"""
    Implementation of MobileNetV3 for CIFAR/SVHN/32x32

    From: Searching for MobileNetV3, https://arxiv.org/abs/1905.02244
    By: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
"""


def MobileNetV3_Block(input,
                      filters,
                      kernel_size,
                      expansion_filters,
                      stride,
                      use_SE,
                      activation,
                      width_multiplier,
                      block_id,
                      data_format):
    """
    Inverted residual block
    Arguments:
    ----------
    input: Tensor
        Input tensor
    filters: int
        Number of output filters
    expansion_filters: int
        Number of internal filters
    stride: int
        -
    use_SE: boolean
        -
    activation: str
        -
    width_multiplier: float
        Controls the width of the network.
    block_id: int
        Id of current block in the network
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs.
    """
    block_name = f"block_{block_id}"
    channel_axis = -1 if data_format == 'channels_last' else 1
    in_channels = input.shape[channel_axis]
    out_channels = filters * width_multiplier
    
    x = input
    # Expand
    x = nn.layers.Conv2D(
        filters=expansion_filters,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=block_name + "_expand_conv",
        kernel_regularizer=nn.regularizers.l2(0.00002)
    )(x)
    x = nn.layers.BatchNormalization(
        axis=channel_axis,
        name=block_name + "_expand_BN")(x)
    x = get_activation_layer(activation, name=block_name + f"_expand_{activation}")(x)
    
    # Depthwise
    x = nn.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        data_format=data_format,
        use_bias=False,
        name=block_name + "_depthwise",
        kernel_regularizer=nn.regularizers.l2(0.00002)
    )(x)
    x = nn.layers.BatchNormalization(
        axis=channel_axis,
        name=block_name + "_depthwise_BN")(x)
    x = get_activation_layer(activation, name=block_name + f"_depthwise_{activation}")(x)
    
    if use_SE:
        x = SEBlock(
            in_channels = expansion_filters,
            reduction = 4,
            interanl_activation='relu',
            final_activation='swish',
            data_format = data_format
        )(x)
    
    # Compress
    x = nn.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=block_name + "_compress_conv",
        kernel_regularizer=nn.regularizers.l2(0.00002)
    )(x)
    x = nn.layers.BatchNormalization(
        axis=channel_axis,
        name=block_name + "_compress_BN")(x)
    
    if out_channels == in_channels and stride == 1:
        return nn.layers.Add(name=block_name + "_add")([x, input])

    return x


def MobileNetV3_builder(config,
                        input_shape=(32, 32, 3),
                        upsample_resolution=224,
                        width_multiplier=1.0,
                        classes=10,
                        data_format='channels_last',
                        name="MobileNetV3_custom"):

    """
    Template for
    Parameters:
    -----------
    config: list, shape=[layers,6]
        Network configuration, layer format = [kernel size, expansion, out, use_SE, activation, stride]
    input_shape: list, tuple
        Shape of an input image
    upsample_resolution: int
        Resolution to which input image will be upsampled. (MobileNetV2 was designed for 224px image input)
    width_multiplier: float
        Controls the width of the network.
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs.
    """
    assert width_multiplier > 0
    channel_axis = -1 if data_format == 'channels_last' else 1
    # mean_axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    block_cnt = 0

    input = tf.keras.layers.Input(shape=input_shape)
    x = input
    
    if input_shape[1] != upsample_resolution:
        upsample = upsample_resolution // input_shape[1]
        x = nn.layers.UpSampling2D([upsample, upsample], data_format=data_format)(x)

    x = nn.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False, data_format=data_format)(x)
    x = nn.layers.BatchNormalization(axis=channel_axis)(x)
    x = get_activation_layer('hswish')(x)
    
    for kernel_size, expansion, filters, use_SE, activation, stride in config:
        x = MobileNetV3_Block(
            input=x,
            filters=filters,
            kernel_size=kernel_size,
            expansion_filters=expansion,
            stride=stride,
            use_SE=use_SE,
            activation=activation,
            width_multiplier=width_multiplier,
            block_id=block_cnt,
            data_format=data_format
        )
        block_cnt += 1

    last_stage_filters = config[block_cnt - 1][1]
    x = nn.layers.Conv2D(
        filters = last_stage_filters,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"Conv_{last_stage_filters}"
    )(x)
    x = nn.layers.BatchNormalization(axis=channel_axis,  name="Conv_960_BN")(x)
    x = get_activation_layer('hswish', name="Conv_960_hswish")(x)
    x = nn.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    x = nn.layers.Dense(1280 if last_stage_filters == 960 else 1024, use_bias=False)(x)
    # x = nn.layers.Lambda(lambda x: tf.reduce_mean(x, axis=mean_axis, keepdims=True))(x)

    # x = nn.layers.Conv2D(
    #    filters = 1280 if last_stage_filters == 960 else 1024,
    #    kernel_size=1,
    #    data_format=data_format,
    #    use_bias=False,
    #    name=f"Conv_{1280 if last_stage_filters == 960 else 1024}"
    # )(x)
    x = get_activation_layer(activation, name="Conv_1280_hswish")(x)
    # x = nn.layers.Conv2D(
    #    filters = classes,
    #    kernel_size=1,
    #    data_format=data_format,
    #    use_bias=False
    # )(x)

    # x = nn.layers.Flatten(data_format=data_format)(x)
    output = nn.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=name)


def MobileNetV3Large(input_shape=(32, 32, 3),
                     upsample_resolution=224,
                     width_multiplier=1.0,
                     classes=10,
                     data_format='channels_last'):
    """
    """
    # kernel, exp, out, use_SE, activ, stride
    large_cfg = [
        [3,  16,   16,  False,  'relu',  1],
        [3,  64,   24,  False,  'relu',  2],
        [3,  72,   24,  False,  'relu',  1],
        [5,  72,   40,   True,  'relu',  2],
        [5, 120,   40,   True,  'relu',  1],
        [5, 120,   40,   True,  'relu',  1],
        [3, 240,   80,  False, 'hswish', 2],
        [3, 240,   80,  False, 'hswish', 1],
        [3, 184,   80,  False, 'hswish', 1],
        [3, 184,   80,  False, 'hswish', 1],
        [3, 480,  112,   True, 'hswish', 1],
        [3, 672,  112,   True, 'hswish', 1],
        [5, 672,  160,   True, 'hswish', 2],
        [5, 960,  160,   True, 'hswish', 1],
        [5, 960,  160,   True, 'hswish', 1],
    ]

    return MobileNetV3_builder(large_cfg,
                               input_shape=input_shape,
                               upsample_resolution=upsample_resolution,
                               width_multiplier=width_multiplier,
                               classes=classes,
                               data_format=data_format,
                               name=f'MobileNetV3L_{upsample_resolution}px_{width_multiplier}k')


def MobileNetV3Small(input_shape=(32, 32, 3),
                     upsample_resolution=224,
                     width_multiplier=1.0,
                     classes=10,
                     data_format='channels_last'):
    """
    """
    # kernel, exp,  out, use_SE, activ, stride
    small_cfg = [
        [3,   16,   16,   True,   'relu',  2],
        [3,   72,   24,  False,   'relu',  2],
        [3,   88,   24,  False,   'relu',  1],
        [5,   96,   40,   True, 'hswish',  2],
        [5,  240,   40,   True, 'hswish',  1],
        [5,  240,   40,   True, 'hswish',  1],
        [5,  120,   48,   True, 'hswish',  1],
        [5,  144,   48,   True, 'hswish',  1],
        [5,  288,   96,   True, 'hswish',  2],
        [5,  576,   96,   True, 'hswish',  1],
        [5,  576,   96,   True, 'hswish',  1]
    ]

    return MobileNetV3_builder(small_cfg,
                               input_shape=input_shape,
                               upsample_resolution=upsample_resolution,
                               width_multiplier=width_multiplier,
                               classes=classes,
                               data_format=data_format,
                               name=f'MobileNetV3S_{upsample_resolution}px_{width_multiplier}k')


############## Predefined Nets ##############
@register_model
def MobileNetV3L_320(width_multiplier=1):
    """
    MobileNetV3 Large with 320px upupsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Large(upsample_resolution=320,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3L_224(width_multiplier=1):
    """
    MobileNetV3 Large with 224px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Large(upsample_resolution=224,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3L_192(width_multiplier=1):
    """
    MobileNetV3 Large with 192px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Large(upsample_resolution=192,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3L_160(width_multiplier=1):
    """
    MobileNetV3 Large with 160px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Large(upsample_resolution=160,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3L_128(width_multiplier=1):
    """
    MobileNetV3 Large with 128px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Large(upsample_resolution=128,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3S_320(width_multiplier=1):
    """
    MobileNetV3 Small with 320px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Large(upsample_resolution=320,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3S_224(width_multiplier=1):
    """
    MobileNetV3 Small with 224px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Small(upsample_resolution=224,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3S_192(width_multiplier=1):
    """
    MobileNetV3 Small with 192px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Small(upsample_resolution=192,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3S_160(width_multiplier=1):
    """
    MobileNetV3 Small with 160px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Small(upsample_resolution=160,
                            width_multiplier=width_multiplier)


@register_model
def MobileNetV3S_128(width_multiplier=1):
    """
    MobileNetV3 Small with 128px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV3Small(upsample_resolution=128,
                            width_multiplier=width_multiplier)



