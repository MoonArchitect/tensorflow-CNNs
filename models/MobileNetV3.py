import tensorflow as tf
import tensorflow.keras as nn

from utils.registry import register_model
from .MobileNetV2 import InvertedResidualBlock
from .layers import get_activation_layer
from .layers.utils import _make_divisible

"""
    Implementation of MobileNetV3 for CIFAR/SVHN/32x32

    From: Searching for MobileNetV3, https://arxiv.org/abs/1905.02244
    By: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
"""

# TODO this mobilenetv3 is not equivalent to one presented in paper for imagenet
# Differences:
#   - Image is upsampled, from 'input_shape' -> to specified 'upsample_resolution' (ie. 224, 192, etc.)
#   - Pre-last 1x1 conv with activation is omitted
#   - Last layer is a Dense not 1x1 Conv


def MobileNetV3_builder(config,
                        input_shape=(32, 32, 3),
                        classes=10,
                        upsample_resolution=224,
                        width_factor=1.0,
                        data_format='channels_last',
                        name="MobileNetV3_builder"):

    """
    Builds MobileNetV3 from config
    Parameters:
    -----------
    config: list, shape=[layers,6]
        Network configuration, layer format = [kernel size, expansion channels, output channels, SE used, activation, stride]
    input_shape: list, tuple
        Shape of an input image
    upsample_resolution: int
        Resolution to which input image will be upsampled. (MobileNetV3 was designed for 224px image input)
    width_factor: float
        Width coefficient of the network's layers
    classes: int
        Number of classification classes.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs
    """
    assert width_factor > 0.2 and width_factor % 0.25 == 0, "'width_factor' should be divisible by 0.25"
   
    channel_axis = -1 if data_format == 'channels_last' else 1
    

    input = tf.keras.layers.Input(shape=input_shape)
    x = input
    
    if input_shape[1] != upsample_resolution:
        upsample = upsample_resolution // input_shape[1]
        x = nn.layers.UpSampling2D([upsample, upsample], data_format=data_format)(x)


    x = nn.layers.Conv2D(filters=_make_divisible(16 * width_factor, 8),
                         kernel_size=3,
                         strides=2,
                         #  padding='same',
                         use_bias=False,
                         data_format=data_format)(x)
    x = nn.layers.BatchNormalization(axis=channel_axis)(x)
    x = get_activation_layer('hswish')(x)
    emit_first_stage = True  # TODO code a better solution

    for kernel_size, expansion, filters, use_SE, activation, stride in config:
        filters = _make_divisible(filters * width_factor, 8)
        expansion = _make_divisible(expansion * width_factor, 8)

        x = InvertedResidualBlock(
            out_channels=filters,
            kernel_size=kernel_size,
            expansion=expansion,
            stride=stride,
            use_SE=use_SE,
            activation=activation,
            emit_first_stage=emit_first_stage,
            data_format=data_format
        )(x)
        emit_first_stage = False


    x = nn.layers.Conv2D(filters = expansion,
                         kernel_size=1,
                         data_format=data_format,
                         use_bias=False)(x)
    x = nn.layers.BatchNormalization(axis=channel_axis)(x)
    x = get_activation_layer('hswish')(x)

    x = nn.layers.GlobalAveragePooling2D(data_format=data_format)(x)

    output = nn.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=name)


def MobileNetV3Large(input_shape=(32, 32, 3),
                     upsample_resolution=224,
                     width_factor=1.0,
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
                               width_factor=width_factor,
                               classes=classes,
                               data_format=data_format,
                               name=f'MobileNetV3L_{upsample_resolution}px_{width_factor}k')


def MobileNetV3Small(input_shape=(32, 32, 3),
                     upsample_resolution=224,
                     width_factor=1.0,
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
                               width_factor=width_factor,
                               classes=classes,
                               data_format=data_format,
                               name=f'MobileNetV3S_{upsample_resolution}px_{width_factor}k')


############## Predefined Nets ##############
@register_model
def MobileNetV3L_224(width_factor=1):
    """
    MobileNetV3 Large with 224px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Large(upsample_resolution=224,
                            width_factor=width_factor)


@register_model
def MobileNetV3L_192(width_factor=1):
    """
    MobileNetV3 Large with 192px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Large(upsample_resolution=192,
                            width_factor=width_factor)


@register_model
def MobileNetV3L_160(width_factor=1):
    """
    MobileNetV3 Large with 160px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Large(upsample_resolution=160,
                            width_factor=width_factor)


@register_model
def MobileNetV3L_128(width_factor=1):
    """
    MobileNetV3 Large with 128px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Large(upsample_resolution=128,
                            width_factor=width_factor)


@register_model
def MobileNetV3S_224(width_factor=1):
    """
    MobileNetV3 Small with 224px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Small(upsample_resolution=224,
                            width_factor=width_factor)


@register_model
def MobileNetV3S_192(width_factor=1):
    """
    MobileNetV3 Small with 192px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Small(upsample_resolution=192,
                            width_factor=width_factor)


@register_model
def MobileNetV3S_160(width_factor=1):
    """
    MobileNetV3 Small with 160px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Small(upsample_resolution=160,
                            width_factor=width_factor)


@register_model
def MobileNetV3S_128(width_factor=1):
    """
    MobileNetV3 Small with 128px upsampled resolution
    
    Arguments:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    
    """
    return MobileNetV3Small(upsample_resolution=128,
                            width_factor=width_factor)



