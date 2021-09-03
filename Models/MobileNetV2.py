import tensorflow as tf
import tensorflow.keras as nn
from utils.registry import register_model
""" 
    Implementation of MobileNetV2 for CIFAR/SVHN/32x32
    
    From: MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/abs/1801.04381 
    By: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
"""

def inverted_res_block(input, 
                       filters, 
                       expansion, 
                       stride, 
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
    expansion: int
        Value of internal channel expansion
    stride: int
        -
    width_multiplier: float
        Controls the width of the network.
    block_id: int
        Id of current block in the network
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    block_name = f"block_{block_id}"
    channel_axis = -1 if data_format=='channels_last' else 1
    in_channels = input.shape[channel_axis]
    out_channels = filters * width_multiplier
    
    x = input
    # Expand
    if block_id:
        x = nn.layers.Conv2D(
            filters=in_channels * expansion,
            kernel_size=1,
            data_format=data_format,
            use_bias=False,
            name=block_name+"_expand_conv",
            kernel_regularizer=nn.regularizers.l2(0.00004)
        )(x)
    x = nn.layers.BatchNormalization(
        axis=channel_axis, 
        momentum=0.999, 
        name=block_name+"_expand_BN")(x)
    x = nn.layers.ReLU(max_value=6.0, name=block_name+"_expand_relu6")(x)
    
    # Depthwise
    x = nn.layers.DepthwiseConv2D(
        kernel_size=3, 
        strides=stride, 
        padding='same',
        data_format=data_format,
        use_bias=False,
        name=block_name+"_depthwise",
        kernel_regularizer=nn.regularizers.l2(0.00004)
    )(x)
    x = nn.layers.BatchNormalization(
        axis=channel_axis, 
        momentum=0.999, 
        name=block_name+"_depthwise_BN")(x)
    x = nn.layers.ReLU(max_value=6.0, name=block_name+"_depthwise_relu6")(x)

    # Compress
    x = nn.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=block_name+"_compress_conv",
        kernel_regularizer=nn.regularizers.l2(0.00004)
    )(x)
    x = nn.layers.BatchNormalization(
        axis=channel_axis, 
        momentum=0.999, 
        name=block_name+"_compress_BN")(x)
    
    if out_channels == in_channels and stride == 1:
        return nn.layers.Add(name=block_name+"_add")([x, input])

    return x


def MobileNetV2(input_shape=(32, 32, 3),
                upsample_resolution=224,
                width_multiplier=1.0,
                classes=10,
                data_format='channels_last',
                **kwargs):

    """
    Template for
    Parameters:
    -----------
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
    channel_axis = -1 if data_format=='channels_last' else 1
    block_cnt = 0
    config = [
    #    t,   c, n, s
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    input = tf.keras.layers.Input(shape=input_shape)
    x = input
    
    if input_shape[1] != upsample_resolution:
        upsample = upsample_resolution // input_shape[1]
        x = nn.layers.UpSampling2D([upsample, upsample], data_format=data_format)(x)

    x = nn.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False,data_format=data_format)(x)
    
    for expansion, filters, n, first_stride in config:
        for strides in [first_stride] + [1]*(n-1):
            x = inverted_res_block(
                input=x,
                filters=filters,
                expansion=expansion,
                stride=strides,
                width_multiplier=width_multiplier,
                block_id=block_cnt,
                data_format=data_format
            )
            block_cnt += 1

    x = nn.layers.Conv2D(
        filters = 1280,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        padding='same'
    )(x)
    x = nn.layers.BatchNormalization(axis=channel_axis)(x)
    x = nn.layers.ReLU(6.0)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x) # use Conv 1x1

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'MobileNetV2_{upsample_resolution}_{width_multiplier}')

############## Predefined Nets ##############
@register_model
def MobileNetV2_320(width_multiplier=1,
                    input_shape=(32,32,3),
                    classes=10,
                    data_format='channels_last'):
    """
    MobileNetV2 with 320px sampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    input_shape: list/tuple
        Shape of an input image 
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    return MobileNetV2(input_shape=input_shape,
                       upsample_resolution=320,
                       width_multiplier=width_multiplier,
                       classes=classes,
                       data_dormat=data_format)


@register_model
def MobileNetV2_224(width_multiplier=1,
                    input_shape=(32,32,3),
                    classes=10,
                    data_format='channels_last'):
    """
    MobileNetV2 with 224px sampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    input_shape: list/tuple
        Shape of an input image 
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    return MobileNetV2(input_shape=input_shape,
                       upsample_resolution=224,
                       width_multiplier=width_multiplier,
                       classes=classes,
                       data_dormat=data_format)


@register_model
def MobileNetV2_192(width_multiplier=1,
                    input_shape=(32,32,3),
                    classes=10,
                    data_format='channels_last'):
    """
    MobileNetV2 with 192px sampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    input_shape: list/tuple
        Shape of an input image 
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    return MobileNetV2(input_shape=input_shape,
                       upsample_resolution=192,
                       width_multiplier=width_multiplier,
                       classes=classes,
                       data_dormat=data_format)


@register_model
def MobileNetV2_160(width_multiplier=1,
                    input_shape=(32,32,3),
                    classes=10,
                    data_format='channels_last'):
    """
    MobileNetV2 with 160px sampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    input_shape: list/tuple
        Shape of an input image 
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    return MobileNetV2(input_shape=input_shape,
                       upsample_resolution=160,
                       width_multiplier=width_multiplier,
                       classes=classes,
                       data_dormat=data_format)


@register_model
def MobileNetV2_128(width_multiplier=1,
                    input_shape=(32,32,3),
                    classes=10,
                    data_format='channels_last'):
    """
    MobileNetV2 with 128px sampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    input_shape: list/tuple
        Shape of an input image 
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    return MobileNetV2(input_shape=input_shape,
                       upsample_resolution=128,
                       width_multiplier=width_multiplier,
                       classes=classes,
                       data_dormat=data_format)


@register_model
def MobileNetV2_96(width_multiplier=1,
                   input_shape=(32,32,3),
                   classes=10,
                   data_format='channels_last'):
    """
    MobileNetV2 with 96px sampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    input_shape: list/tuple
        Shape of an input image 
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    return MobileNetV2(input_shape=input_shape,
                       upsample_resolution=96,
                       width_multiplier=width_multiplier,
                       classes=classes,
                       data_dormat=data_format)

