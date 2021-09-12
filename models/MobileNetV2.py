import tensorflow as tf
import tensorflow.keras as nn
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
    def __init__(self,
                 filters,
                 expansion,
                 stride,
                 width_multiplier,
                 first_block = False,
                 data_format = 'channels_last',
                 **kwargs):
        super().__init__(**kwargs)

        channel_axis = -1 if data_format == 'channels_last' else 1
        self.out_channels = filters * width_multiplier
        self.in_channels = None
        self.stride = stride
        self.first_block = first_block
        self.expansion = expansion
        self.data_format = data_format

        # Expand
        self.expand_conv = None
        self.expand_relu = nn.layers.ReLU(max_value=6.0)
        self.expand_bn = nn.layers.BatchNormalization(axis=channel_axis,
                                                      momentum=0.999)
        
        # depthwise
        self.depthwise_relu = nn.layers.ReLU(max_value=6.0)
        self.depthwise_conv = nn.layers.DepthwiseConv2D(kernel_size=3,
                                                        strides=stride,
                                                        padding='same',
                                                        data_format=data_format,
                                                        use_bias=False,
                                                        kernel_regularizer=nn.regularizers.l2(0.00004))
        self.depthwise_bn = nn.layers.BatchNormalization(axis=channel_axis,
                                                         momentum=0.999)

        # Compress
        self.compress_conv = nn.layers.Conv2D(filters=self.out_channels,
                                              kernel_size=1,
                                              data_format=data_format,
                                              use_bias=False,
                                              kernel_regularizer=nn.regularizers.l2(0.00004))
        self.compress_bn = nn.layers.BatchNormalization(axis=channel_axis,
                                                        momentum=0.999)
    
    def build(self, input_shape):
        channel_axis = -1 if self.data_format == 'channels_last' else 1
        self.in_channels = input_shape[channel_axis]

        if not self.first_block:
            self.expand_conv = nn.layers.Conv2D(filters=self.in_channels * self.expansion,
                                                kernel_size=1,
                                                data_format=self.data_format,
                                                use_bias=False,
                                                kernel_regularizer=nn.regularizers.l2(0.00004))

        return super().build(input_shape)

    def call(self, inputs):
        x = inputs
        
        # Expand
        if not self.first_block:
            x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.expand_relu(x)

        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)

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
    assert width_multiplier > 0
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

    x = nn.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias = False, data_format = data_format)(x)
    
    expansion, filters, n, first_stride = config[0]
    x = InvertedResidualBlock(
        filters=filters,
        expansion=expansion,
        stride=first_stride,
        width_multiplier=width_multiplier,
        first_block=True,
        data_format=data_format
    )(x)

    for expansion, filters, n, first_stride in config[1:]:
        for stride in [first_stride] + [1] * (n - 1):
            x = InvertedResidualBlock(
                filters=filters,
                expansion=expansion,
                stride=stride,
                width_multiplier=width_multiplier,
                data_format=data_format
            )(x)

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
    output = tf.keras.layers.Dense(classes)(x)  # use Conv 1x1

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

