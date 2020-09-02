import tensorflow as tf
import tensorflow.keras as nn
from .ResNetV2 import BasicUnit, BottleneckUnit
from ..Layers import get_activation_layer, get_channels

""" 
    Implementation of ResNetXt for CIFAR/SVHN/32x32

    From: Aggregated Residual Transformations for Deep Neural Networks, https://arxiv.org/abs/1611.05431
    By: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
"""

############## Building Blocks ##############
def ResNeXtStage(layers,
                 filters,
                 groups=16,
                 kernel_size=(3,3),
                 strides=(1,1),
                 expansion = 2,
                 data_format='channels_last',
                 activation='relu',
                 **kwargs):
    """
    Arguments:
    ----------
    layers: int
        Number of residual Units in that stage
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the convolution).
    groups: int
        -
    kernel_size: int, tuple/list of 2 integers
        Central 2D convolution window's Height and Width
    strides: int, tuple/list of 2 integers
        Specifying the strides of the central convolution along the height and width
    expansion: int
        -
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    activation: String or keras.Layer
        Activation function to use after each convolution.
    bottleneck: Boolean
        Whether to use bottleneck type unit
    """

    Block = BottleneckUnit
    def fwd(input):
        sc = nn.layers.AvgPool2D(strides, data_format=data_format)(input) if strides != (1, 1) else input
        if get_channels(input, data_format) != filters:
            pad = [(filters - get_channels(input, data_format)) // 2] * 2
            sc = tf.pad(sc, [[0,0], [0,0], [0,0], pad] if data_format=='channels_last' else [[0,0], pad, [0,0], [0,0]])
        
        x = Block(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  groups=groups,
                  expansion=expansion,
                  activation=activation,
                  data_format=data_format,
                  **kwargs)(input)
        input = nn.layers.Add()([x, sc])
        
        for i in range(layers-1):
            x = Block(filters=filters,
                      kernel_size=kernel_size,
                      groups=groups,
                      expansion=expansion,
                      activation=activation,
                      data_format=data_format,
                      **kwargs)(input)
            input = nn.layers.Add()([x, input])
        
        return input
    
    return fwd


def ResNeXt(conv_per_stage,
            img_size=(32,32),
            img_channels=3,
            classes=10,
            cardinality=32,#16,
            filters=32,#64,
            activation='relu',
            data_format='channels_last',
            **kwargs):
    """
    Template for Bottleneck ResNet with 4 stages
    Parameters:
    -----------
    conv_per_stage: list, tuple
        Number of residual blocks in each stage
    img_size: list, tuple
        Size of a single input image
    img_channels: int
        Number of channels in a single input image
    classes: int
        Number of classification classes.
    filters: int
        Number of filters in stem layer
    cardinality: int
        -
    activation: string, keras.Layer
        Activation function to use after each convolution.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """
    assert filters % cardinality == 0, f"Number of filters ({filters}) has to be divisible by cardinality ({cardinality})"

    input_shape = (*img_size, img_channels) if data_format=='channels_last' else (img_channels, *img_size)
    strides = [(1,1)] + [(2,2)]*3
    expansion = 2

    input = tf.keras.layers.Input(shape=input_shape)
    # Initial Convolution
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               data_format=data_format,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    # Residual Stages
    for layers, strides in zip(conv_per_stage, strides):
        x = ResNeXtStage(layers=layers,
                         filters=filters * expansion,
                         groups=cardinality,
                         kernel_size=(3,3),
                         strides=strides,
                         expansion=expansion,
                         data_format=data_format,
                         activation=activation,
                         **kwargs)(x)
        filters *= 2

    x = tf.keras.layers.BatchNormalization(-1 if data_format=='channels_last' else 1)(x)
    x = get_activation_layer(activation)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'ResNeXt{sum(conv_per_stage) * 3 + 2}_{cardinality}x{filters // 16 // cardinality}d')


############## Predefined Nets ##############
def ResNeXt35(img_size=(32,32),
              img_channels=3,
              classes=10,
              activation='relu',
              data_format='channels_last',
              **kwargs):
    """
    ResNeXt35 model for CIFAR/SVHN
    Parameters:
    ----------
    img_size: list, tuple
        Size of a single input image
    img_channels: int
        Number of channels in a single input image
    classes: int
        Number of classification classes.
    activation: string, keras.Layer
        Main activation function of the network.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    Returns:
    ----------
    keras.Model
    """
    return ResNeXt(conv_per_stage=[2, 3, 4, 2],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNeXt50(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNeXt50 model for CIFAR/SVHN
    Parameters:
    ----------
    img_size: list, tuple
        Size of a single input image
    img_channels: int
        Number of channels in a single input image
    classes: int
        Number of classification classes.
    activation: string, keras.Layer
        Main activation function of the network.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    Returns:
    ----------
    keras.Model
    """
    return ResNeXt(conv_per_stage=[3, 4, 6, 3],
                   img_size=img_size,
                   img_channels=img_channels,
                   classes=classes,
                   activation=activation,
                   data_format=data_format,
                   **kwargs)

def ResNeXt101(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNeXt101 model for CIFAR/SVHN
    Parameters:
    ----------
    img_size: list, tuple
        Size of a single input image
    img_channels: int
        Number of channels in a single input image
    classes: int
        Number of classification classes.
    activation: string, keras.Layer
        Main activation function of the network.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    Returns:
    ----------
    keras.Model
    """
    return ResNeXt(conv_per_stage=[3, 4, 23, 3],
                   img_size=img_size,
                   img_channels=img_channels,
                   classes=classes,
                   activation=activation,
                   data_format=data_format,
                   **kwargs)

def ResNeXt152(img_size=(32,32),
                img_channels=3,
                classes=10,
                activation='relu',
                data_format='channels_last',
                **kwargs):
    """
    ResNeXt152 model for CIFAR/SVHN
    Parameters:
    ----------
    img_size: list, tuple
        Size of a single input image
    img_channels: int
        Number of channels in a single input image
    classes: int
        Number of classification classes.
    activation: string, keras.Layer
        Main activation function of the network.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    Returns:
    ----------
    keras.Model
    """
    return ResNeXt(conv_per_stage=[3, 8, 36, 3],
                   img_size=img_size,
                   img_channels=img_channels,
                   classes=classes,
                   activation=activation,
                   data_format=data_format,
                   **kwargs)

