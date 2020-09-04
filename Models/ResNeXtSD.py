import tensorflow as tf
import tensorflow.keras as nn
from .ResNetV2 import BasicUnit, BottleneckUnit
from .ResNetSD import StochResWrapper
from ..Layers import get_activation_layer, get_channels, linear_decay_fn

""" 
    Implementation of ResNetXt for CIFAR/SVHN/32x32

    From: Aggregated Residual Transformations for Deep Neural Networks, https://arxiv.org/abs/1611.05431
    By: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
"""

############## Building Blocks ##############
def ResNeXtSD_Stage(layers,
                    filters,
                    groups=16,
                    kernel_size=(3,3),
                    strides=(1,1),
                    survival_fn=None,
                    stage_start_pos=None,
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
    assert not (survival_fn is None and stage_start_pos is None), "\'survival_fn\' has to be a function and \'stage_start_pos\' has to be an int" 

    def pool_pad_input(input, 
                       filters, 
                       strides, 
                       data_format):
        """
        Pools and pads input if necessary
        Arguments:
        ----------
        Returns:
        --------
        """
        sc = nn.layers.AvgPool2D(strides, data_format=data_format)(input) if strides != (1, 1) else input
        if get_channels(input, data_format) != filters:
            pad = [(filters - get_channels(input, data_format)) // 2] * 2
            sc = tf.pad(sc, [[0,0], [0,0], [0,0], pad] if data_format=='channels_last' else [[0,0], pad, [0,0], [0,0]])

        return sc

    Block = BottleneckUnit
    def fwd(input):
        transform_input_fn = lambda x: pool_pad_input(input=x, filters=filters, strides=strides, data_format=data_format)

        x = StochResWrapper(
            Block(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  groups=groups,
                  expansion=expansion,
                  activation=activation,
                  data_format=data_format,
                  **kwargs),
            survival_rate=survival_fn(stage_start_pos),
            transform_input_fn=transform_input_fn,
            **kwargs
        )(input)
        
        for i in range(1,layers):
            x = StochResWrapper(
                Block(filters=filters,
                      kernel_size=kernel_size,
                      groups=groups,
                      expansion=expansion,
                      activation=activation,
                      data_format=data_format,
                      **kwargs),
                survival_rate=survival_fn(stage_start_pos + i),
                **kwargs
            )(x)
        
        return x
    
    return fwd


def ResNeXtSD(conv_per_stage,
              min_survival_p,
              img_size=(32,32),
              img_channels=3,
              classes=10,
              cardinality=16,
              filters=64,
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
    
    survival_fn = linear_decay_fn((0, 1), (sum(conv_per_stage), min_survival_p))
    layer_cnt = 1 # ...

    input = tf.keras.layers.Input(shape=input_shape)

    x = input
    if data_format == 'channels_last':
        x = tf.transpose(input, [0, 3, 1, 2])
        data_format = 'channels_first'

    # Initial Convolution
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               data_format=data_format,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    # Residual Stages
    for layers, strides in zip(conv_per_stage, strides):
        x = ResNeXtSD_Stage(layers=layers,
                            filters=filters * expansion,
                            groups=cardinality,
                            kernel_size=(3,3),
                            strides=strides,
                            survival_fn=survival_fn,
                            stage_start_pos=layer_cnt,
                            expansion=expansion,
                            data_format=data_format,
                            activation=activation,
                            **kwargs)(x)
        filters *= 2
        layer_cnt += layers

    x = tf.keras.layers.BatchNormalization(-1 if data_format=='channels_last' else 1)(x)
    x = get_activation_layer(activation)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'ResNeXtSD{sum(conv_per_stage) * 3 + 2}_{cardinality}x{filters // 16 // cardinality}d')


############## Predefined Nets ##############
def ResNeXtSD35(min_survival_p=0.8,
                img_size=(32,32),
                img_channels=3,
                classes=10,
                activation='relu',
                data_format='channels_last',
                **kwargs):
    """
    ResNeXtSD35 model for CIFAR/SVHN
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
    return ResNeXtSD(conv_per_stage=[2, 3, 4, 2],
                     min_survival_p=min_survival_p,
                     img_size=img_size,
                     img_channels=img_channels,
                     classes=classes,
                     activation=activation,
                     data_format=data_format,
                     **kwargs)

def ResNeXtSD50(min_survival_p=0.65,
                img_size=(32,32),
                img_channels=3,
                classes=10,
                activation='relu',
                data_format='channels_last',
                **kwargs):
    """
    ResNeXtSD50 model for CIFAR/SVHN
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
    return ResNeXtSD(conv_per_stage=[3, 4, 6, 3],
                     min_survival_p=min_survival_p,
                     img_size=img_size,
                     img_channels=img_channels,
                     classes=classes,
                     activation=activation,
                     data_format=data_format,
                     **kwargs)

def ResNeXtSD101(min_survival_p=0.5,
                 img_size=(32,32),
                 img_channels=3,
                 classes=10,
                 activation='relu',
                 data_format='channels_last',
                 **kwargs):
    """
    ResNeXtSD101 model for CIFAR/SVHN
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
    return ResNeXtSD(conv_per_stage=[3, 4, 23, 3],
                     min_survival_p=min_survival_p,
                     img_size=img_size,
                     img_channels=img_channels,
                     classes=classes,
                     activation=activation,
                     data_format=data_format,
                     **kwargs)

def ResNeXtSD152(min_survival_p=0.35,
                 img_size=(32,32),
                 img_channels=3,
                 classes=10,
                 activation='relu',
                 data_format='channels_last',
                 **kwargs):
    """
    ResNeXtSD152 model for CIFAR/SVHN
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
    return ResNeXtSD(conv_per_stage=[3, 8, 36, 3],
                     min_survival_p=min_survival_p,
                     img_size=img_size,
                     img_channels=img_channels,
                     classes=classes,
                     activation=activation,
                     data_format=data_format,
                     **kwargs)

