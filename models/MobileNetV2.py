import tensorflow as tf
import tensorflow.keras as nn

from .SeNet import SEBlock
from .layers import get_activation_layer
from utils.registry import register_model


"""
    Implementation of MobileNetV2 for CIFAR/SVHN/32x32
    
    From: MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/abs/1801.04381
    By: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
"""


class InvertedResidualBlock(nn.layers.Layer):
    """
    Inverted residual block
    Arguments:
    ----------
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
    def __init__(self,
                 out_channels,
                 expansion,
                 stride,
                 kernel_size = (3, 3),
                 use_SE = False,
                 activation="relu6",
                 emit_first_stage = False,
                 data_format = 'channels_last',
                 **kwargs):
        super().__init__(**kwargs)

        channel_axis = -1 if data_format == 'channels_last' else 1
        self.out_channels = out_channels
        self.in_channels = None
        self.stride = stride
        self.emit_first_stage = emit_first_stage
        self.expansion = expansion
        self.data_format = data_format
        self.use_SE = use_SE

        # Expand
        self.expand_conv = None
        self.expand_act = get_activation_layer(activation)
        self.expand_bn = nn.layers.BatchNormalization(axis=channel_axis)
        
        # depthwise
        self.depthwise_act = get_activation_layer(activation)
        self.depthwise_conv = nn.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                        strides=stride,
                                                        padding='same',
                                                        data_format=data_format,
                                                        use_bias=False,
                                                        kernel_regularizer=nn.regularizers.l2(0.00012))
        self.depthwise_bn = nn.layers.BatchNormalization(axis=channel_axis)

        self.se_block = None

        # Compress
        self.compress_conv = nn.layers.Conv2D(filters=self.out_channels,
                                              kernel_size=1,
                                              data_format=data_format,
                                              use_bias=False,
                                              kernel_regularizer=nn.regularizers.l2(0.00012))
        self.compress_bn = nn.layers.BatchNormalization(axis=channel_axis)

    
    def build(self, input_shape):
        channel_axis = -1 if self.data_format == 'channels_last' else 1
        self.in_channels = input_shape[channel_axis]

        if not self.emit_first_stage:
            self.expand_conv = nn.layers.Conv2D(filters=self.expansion,
                                                kernel_size=1,
                                                data_format=self.data_format,
                                                use_bias=False,
                                                kernel_regularizer=nn.regularizers.l2(0.00012))

        if self.use_SE:
            self.se_block = SEBlock(
                in_channels = self.expansion,
                reduction = 4,
                internal_activation='relu',
                final_activation='swish',
                data_format = self.data_format
            )

        return super().build(input_shape)

    def call(self, inputs):
        x = inputs
        
        # Expand
        if not self.emit_first_stage:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)

        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)

        if self.use_SE:
            x = self.se_block(x)

        # Compress
        x = self.compress_conv(x)
        x = self.compress_bn(x)

        if self.out_channels == self.in_channels and self.stride == 1:
            x = x + inputs

        return x


def MobileNetV2(input_shape=(32, 32, 3),
                upsample_resolution=224,
                width_multiplier=1.0,
                classes=10,
                data_format='channels_last'):

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
    assert width_multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], "'width_multiplier' has to be one of [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]"
    channel_axis = -1 if data_format == 'channels_last' else 1
    config = [
        # t,  c, n, s
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

    x = nn.layers.Conv2D(filters=32 * width_multiplier, kernel_size=3, strides=2, padding='same', use_bias = False, data_format = data_format)(x)
    x = nn.layers.BatchNormalization(axis=channel_axis)(x)
    x = nn.layers.ReLU(6)(x)

    expansion, filters, n, first_stride = config[0]
    x = InvertedResidualBlock(
        out_channels=filters * width_multiplier,
        expansion=32 * expansion * width_multiplier,
        stride=first_stride,
        emit_first_stage=True,
        data_format=data_format
    )(x)
    input_filters = filters

    for expansion, filters, n, first_stride in config[1:]:
        for stride in [first_stride] + [1] * (n - 1):
            x = InvertedResidualBlock(
                out_channels=filters * width_multiplier,
                expansion=input_filters * expansion * width_multiplier,
                stride=stride,
                data_format=data_format
            )(x)
            input_filters = filters

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
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'MobileNetV2_{upsample_resolution}px_{width_multiplier}k')


############## Predefined Nets ##############

@register_model
def MobileNetV2_320(width_multiplier=1):
    """
    MobileNetV2 with 320px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV2(upsample_resolution=320,
                       width_multiplier=width_multiplier)


@register_model
def MobileNetV2_224(width_multiplier=1):
    """
    MobileNetV2 with 224px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV2(upsample_resolution=224,
                       width_multiplier=width_multiplier)


@register_model
def MobileNetV2_192(width_multiplier=1):
    """
    MobileNetV2 with 192px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV2(upsample_resolution=192,
                       width_multiplier=width_multiplier)


@register_model
def MobileNetV2_160(width_multiplier=1):
    """
    MobileNetV2 with 160px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV2(upsample_resolution=160,
                       width_multiplier=width_multiplier)


@register_model
def MobileNetV2_128(width_multiplier=1):
    """
    MobileNetV2 with 128px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV2(upsample_resolution=128,
                       width_multiplier=width_multiplier)


@register_model
def MobileNetV2_96(width_multiplier=1):
    """
    MobileNetV2 with 96px upsampled resolution
    
    Arguments:
    ----------
    width_multiplier: float
        Controls the width of the network.
    
    """
    return MobileNetV2(upsample_resolution=96,
                       width_multiplier=width_multiplier)

