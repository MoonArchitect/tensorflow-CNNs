import tensorflow as tf
import tensorflow.keras as nn

from utils.registry import register_model
from .SeNet import SEBlock
from .layers import get_channels, AntiAliasDownsampling, _make_divisible

"""
    Implementation of TResNet for CIFAR/SVHN/32x32
    
    From: TResNet: High Performance GPU-Dedicated Architecture, https://arxiv.org/abs/2003.13630
    By: Tal Ridnik, Hussam Lawen, Asaf Noy, Emanuel Ben Baruch, Gilad Sharir, Itamar Friedman
"""

# TODO this implementation is not equivalent to one presented in the paper
# Differences:
#   - Image is upsampled, from 'input_shape' -> to specified 'upsample_resolution' (ie. 224, 192, etc.)
#   - Paper uses 1024 and 2048 channels for bottleneck layers


class SpaceToDepth(nn.layers.Layer):
    """
    SpaceToDepth Stem \\
    Rearranges spatial data into depth
    """
    def __init__(self):
        super().__init__()
        self.compress_shape = None
        self.expand_channel_shape = None

    def build(self, input_shape):  # TODO allow different data_format
        super().build(input_shape)
        
        N, C, H, W = input_shape
        N = -1
        k = 4
        self.compress_shape = [N, C, H // k, k, W // k, k]
        self.expand_channel_shape = [N, C * k * k, H // k, W // k]



    def call(self, inputs):
        x = tf.reshape(inputs, self.compress_shape)
        x = tf.transpose(x, [0, 3, 5, 1, 2, 4])
        x = tf.reshape(x, self.expand_channel_shape)
        return x


class BasicBlock(nn.layers.Layer):
    """
    """
    def __init__(self,
                 filters,
                 strides,
                 se_block,
                 data_format,
                 **kwargs):
        super().__init__(**kwargs)

        self.channel_axis = -1 if data_format == "channels_last" else 1
        self.data_format = data_format
        self.downsample = (strides == 2)
        self.filters = filters
        self.residual_conv = self.residual_bn = self.se = None

        self.conv1 = nn.layers.Conv2D(filters=filters,
                                      kernel_size=3,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      kernel_regularizer=nn.regularizers.l2(0.0001),
                                      data_format=data_format)
        self.bn1 = nn.layers.BatchNormalization(self.channel_axis)
        
        self.conv2 = nn.layers.Conv2D(filters=filters,
                                      kernel_size=3,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      kernel_regularizer=nn.regularizers.l2(0.0001),
                                      data_format=data_format)
        self.bn2 = nn.layers.BatchNormalization(self.channel_axis)

        if se_block:
            self.se = SEBlock(in_channels=filters,
                              reduction=4,
                              data_format=data_format)

        if self.downsample:
            self.aa_downsample = AntiAliasDownsampling(in_channels=filters,
                                                       data_format=data_format)
            self.input_downsample = nn.layers.AvgPool2D(pool_size=2,
                                                        strides=2,
                                                        padding='same',
                                                        data_format=data_format)


    def build(self, input_shape):
        super().build(input_shape)
        
        if self.filters != get_channels(input_shape, self.data_format):
            self.residual_conv = nn.layers.Conv2D(filters=self.filters,
                                                  kernel_size=1,
                                                  strides=1,
                                                  data_format=self.data_format,
                                                  use_bias=False)
            self.residual_bn = nn.layers.BatchNormalization(self.channel_axis)
        

    def call(self, inputs):
        residual = inputs
        if self.downsample:
            residual = self.input_downsample(residual)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
        

        x = self.conv1(inputs)
        if self.downsample:
            x = self.aa_downsample(x)
        x = self.bn1(x)
        x = tf.nn.leaky_relu(x, 0.001)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.se is not None:
            x = self.se(x)

        return tf.nn.relu(x + residual)


class BottleneckBlock(nn.layers.Layer):
    """
    """
    def __init__(self,
                 filters,
                 strides,
                 se_block,
                 data_format,
                 **kwargs):
        super().__init__(**kwargs)

        self.channel_axis = -1 if data_format == "channels_last" else 1
        self.data_format = data_format
        self.downsample = (strides == 2)
        self.filters = filters
        self.residual_conv = self.residual_bn = self.se = None
        
        self.conv1 = nn.layers.Conv2D(filters=filters // 4,
                                      kernel_size=1,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      kernel_regularizer=nn.regularizers.l2(0.0001),
                                      data_format=data_format)
        self.bn1 = nn.layers.BatchNormalization(self.channel_axis)
        
        self.conv2 = nn.layers.Conv2D(filters=filters // 4,
                                      kernel_size=3,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      kernel_regularizer=nn.regularizers.l2(0.0001),
                                      data_format=data_format)
        self.bn2 = nn.layers.BatchNormalization(self.channel_axis)

        self.conv3 = nn.layers.Conv2D(filters=filters,
                                      kernel_size=1,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      kernel_regularizer=nn.regularizers.l2(0.0001),
                                      data_format=data_format)
        self.bn3 = nn.layers.BatchNormalization(self.channel_axis)

        if se_block:
            self.se = SEBlock(in_channels=filters // 4,
                              reduction=8,
                              data_format=data_format)

        if self.downsample:
            self.aa_downsample = AntiAliasDownsampling(in_channels=filters // 4,
                                                       data_format=data_format)
            self.input_downsample = nn.layers.AvgPool2D(pool_size=2,
                                                        strides=2,
                                                        padding='same',
                                                        data_format=data_format)


    def build(self, input_shape):
        super().build(input_shape)
        
        if self.filters != get_channels(input_shape, self.data_format):
            self.residual_conv = nn.layers.Conv2D(filters=self.filters,
                                                  kernel_size=1,
                                                  strides=1,
                                                  data_format=self.data_format,
                                                  use_bias=False)
            self.residual_bn = nn.layers.BatchNormalization(self.channel_axis)
        

    def call(self, inputs):
        residual = inputs
        if self.downsample:
            residual = self.input_downsample(residual)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.leaky_relu(x, 0.001)

        x = self.conv2(x)
        if self.downsample:
            x = self.aa_downsample(x)
        x = self.bn2(x)
        x = tf.nn.leaky_relu(x, 0.001)

        if self.se is not None:
            x = self.se(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        return tf.nn.relu(x + residual)



def TResNet(layers,
            upsample_resolution,
            width_factor=1.0,
            data_format='channels_last',
            input_shape=(32, 32, 3),
            classes=10,
            name="TResNet_custom"):
    """
    Builds TResNet
    -----------
    Parameters:
    layers: list, shape=[4]
        Number of blocks in each stage [n_stage1, n_stage2, n_stage3, n_stage4]
    upsample_resolution: int/tuple
        Resolution to which input image will be upsampled
    width_factor: float
        Width coefficient of the network's layers
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs
    input_shape: list, tuple
        Shape of an input image
    classes: int
        Number of classification classes
    name: str
        Name of the model
    """
    img_size = input_shape[:2] if data_format == 'channels_last' else input_shape[1:]
    if isinstance(upsample_resolution, int):
        upsample_resolution = (upsample_resolution, upsample_resolution)
    
    assert isinstance(upsample_resolution, (tuple, list))
    assert upsample_resolution[0] % img_size[0] == 0 \
        and upsample_resolution[1] % img_size[1] == 0, \
        f"Upsample resolution ({upsample_resolution}px) should be divisible by input img size ({img_size}px)"



    upsample = [upsample_resolution[0] // img_size[0],
                upsample_resolution[1] // img_size[1]]

    channel_axis = -1 if data_format == 'channels_last' else 1
    stem_filters = _make_divisible(width_factor * 64, 8)
    cfg = [
        #              filters                         s       Block         SE
        [ _make_divisible(width_factor *  64,  8), 1,      BasicBlock,  True ],
        [ _make_divisible(width_factor * 128,  8), 2,      BasicBlock,  True ],
        [ _make_divisible(width_factor * 256,  8), 2, BottleneckBlock,  True ],
        [ _make_divisible(width_factor * 512,  8), 2, BottleneckBlock, False ]
    ]
    
    
    input = tf.keras.layers.Input(shape=input_shape)
    
    x = input
    if data_format == 'channels_last':                  # Mannually transpose input to NCHW format if necessary, low to 0 perfomance hit
        x = tf.transpose(input, [0, 3, 1, 2])           # Required due to bug with layout_optimizer which does unnecessary back to back transpose ops
        data_format = 'channels_first'                  # with custom layers, eg. NHWC -> transpose_to_NCHW -> Op(eg.Conv2d) -> transpose_to_NHWC -> NHWC

    x = nn.layers.UpSampling2D(upsample, data_format=data_format)(x)

    x = SpaceToDepth()(x)
    
    x = nn.layers.Conv2D(filters=stem_filters,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         data_format=data_format,
                         use_bias=False)(x)
    x = nn.layers.BatchNormalization(channel_axis)(x)
    x = nn.layers.LeakyReLU(0.01)(x)
    
    # 4 Stages
    for n, (filters, stride, Block, se_block) in zip(layers, cfg):
        for strides in [stride] + [1] * (n - 1):
            x = Block(filters, strides, se_block, data_format)(x)
            

    x = nn.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=name)


def TResNetM(upsample_resolution,
             width_factor=1.0,
             data_format='channels_last',
             input_shape=(32, 32, 3),
             classes=10,
             name="TResNetM_custom",
             **kwargs):
    """
    """
    return TResNet(layers=[3, 4, 11, 3],
                   upsample_resolution=upsample_resolution,
                   width_factor=width_factor,
                   data_format=data_format,
                   input_shape=input_shape,
                   classes=classes,
                   name=name,
                   **kwargs)


def TResNetL(upsample_resolution,
             width_factor=1.0,
             data_format='channels_last',
             input_shape=(32, 32, 3),
             classes=10,
             name="TResNetM_custom",
             **kwargs):
    """
    """
    return TResNet(layers=[4, 5, 18, 3],
                   upsample_resolution=upsample_resolution,
                   width_factor=width_factor,
                   data_format=data_format,
                   input_shape=input_shape,
                   classes=classes,
                   name=name,
                   **kwargs)


def TResNetXL(upsample_resolution,
              width_factor=1.0,
              data_format='channels_last',
              input_shape=(32, 32, 3),
              classes=10,
              name="TResNetM_custom",
              **kwargs):
    """
    """
    return TResNet(layers=[3, 5, 24, 3],
                   upsample_resolution=upsample_resolution,
                   width_factor=width_factor,
                   data_format=data_format,
                   input_shape=input_shape,
                   classes=classes,
                   name=name,
                   **kwargs)


############## Predefined Nets ##############
## TResNetM
@register_model
def TResNetM_64px(width_factor = 1.0, **kwargs):
    return TResNetM(upsample_resolution=64,
                    width_factor=width_factor,
                    name="TResNetM_64px",
                    **kwargs)


@register_model
def TResNetM_96px(width_factor = 1.0, **kwargs):
    return TResNetM(upsample_resolution=96,
                    width_factor=width_factor,
                    name="TResNetM_96px",
                    **kwargs)


@register_model
def TResNetM_128px(width_factor = 1.0, **kwargs):
    return TResNetM(upsample_resolution=128,
                    width_factor=width_factor,
                    name="TResNetM_128px",
                    **kwargs)


@register_model
def TResNetM_160px(width_factor = 1.0, **kwargs):
    return TResNetM(upsample_resolution=160,
                    width_factor=width_factor,
                    name="TResNetM_160px",
                    **kwargs)


@register_model
def TResNetM_192px(width_factor = 1.0, **kwargs):
    return TResNetM(upsample_resolution=192,
                    width_factor=width_factor,
                    name="TResNetM_192px",
                    **kwargs)


@register_model
def TResNetM_224px(width_factor = 1.0, **kwargs):
    return TResNetM(upsample_resolution=224,
                    width_factor=width_factor,
                    name="TResNetM_224px",
                    **kwargs)


## TResNetL
@register_model
def TResNetL_64px(width_factor = 1.0, **kwargs):
    return TResNetL(upsample_resolution=64,
                    width_factor=width_factor,
                    name="TResNetL_64px",
                    **kwargs)


@register_model
def TResNetL_96px(width_factor = 1.0, **kwargs):
    return TResNetL(upsample_resolution=96,
                    width_factor=width_factor,
                    name="TResNetL_96px",
                    **kwargs)


@register_model
def TResNetL_128px(width_factor = 1.0, **kwargs):
    return TResNetL(upsample_resolution=128,
                    width_factor=width_factor,
                    name="TResNetL_128px",
                    **kwargs)


@register_model
def TResNetL_160px(width_factor = 1.0, **kwargs):
    return TResNetL(upsample_resolution=160,
                    width_factor=width_factor,
                    name="TResNetL_160px",
                    **kwargs)


@register_model
def TResNetL_192px(width_factor = 1.0, **kwargs):
    return TResNetL(upsample_resolution=192,
                    width_factor=width_factor,
                    name="TResNetL_192px",
                    **kwargs)


@register_model
def TResNetL_224px(width_factor = 1.0, **kwargs):
    return TResNetL(upsample_resolution=224,
                    width_factor=width_factor,
                    name="TResNetL_224px",
                    **kwargs)


## TResNetXL
@register_model
def TResNetXL_64px(width_factor = 1.0, **kwargs):
    return TResNetXL(upsample_resolution=64,
                     width_factor=width_factor,
                     name="TResNetXL_64px",
                     **kwargs)


@register_model
def TResNetXL_96px(width_factor = 1.0, **kwargs):
    return TResNetXL(upsample_resolution=96,
                     width_factor=width_factor,
                     name="TResNetXL_96px",
                     **kwargs)


@register_model
def TResNetXL_128px(width_factor = 1.0, **kwargs):
    return TResNetXL(upsample_resolution=128,
                     width_factor=width_factor,
                     name="TResNetXL_128px",
                     **kwargs)


@register_model
def TResNetXL_160px(width_factor = 1.0, **kwargs):
    return TResNetXL(upsample_resolution=160,
                     width_factor=width_factor,
                     name="TResNetXL_160px",
                     **kwargs)


@register_model
def TResNetXL_192px(width_factor = 1.0, **kwargs):
    return TResNetXL(upsample_resolution=192,
                     width_factor=width_factor,
                     name="TResNetXL_192px",
                     **kwargs)


@register_model
def TResNetXL_224px(width_factor = 1.0, **kwargs):
    return TResNetXL(upsample_resolution=224,
                     width_factor=width_factor,
                     name="TResNetXL_224px",
                     **kwargs)
