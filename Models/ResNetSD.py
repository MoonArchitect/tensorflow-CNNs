import tensorflow as tf
import tensorflow.keras as nn
from .ResNetV2 import BottleneckUnit
from ..Layers import get_activation_layer, get_channels, linear_decay_fn

""" 
    Implementation of ResNet with Stochastic Depth for CIFAR/SVHN/32x32

    From: Deep Networks with Stochastic Depth, https://arxiv.org/abs/1603.09382
    By: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
"""

############## Building Blocks ##############
class StochResWrapper(nn.layers.Layer):
    """
    Arguments:
    survival_rate: int
        -
    """
    def __init__(self, 
                 layer,
                 survival_rate,
                 transform_input_fn = (lambda x: x),
                 name='StochasticResBlock_',
                 **kwargs):
        super(StochResWrapper, self).__init__(name=name + str(nn.backend.get_uid(name)), **kwargs)
        self.layer = layer
        self.p = tf.constant(survival_rate, dtype=tf.float32)
        self.transform_input_fn = transform_input_fn

    def call(self, input):
        survived = nn.backend.in_test_phase(
            tf.constant(True, dtype=tf.bool),       # if in_test_phase
            tf.random.uniform([1], 0, 1) < self.p   # if not in_test_phase
        )

        return tf.cond(
            survived,
            lambda: self.transform_input_fn(input) + self.layer(input),
            lambda: self.transform_input_fn(input)
        )

def StochasticDepthStage(layers,
                         filters,
                         survival_fn,
                         stage_start_pos,
                         kernel_size=(3,3),
                         strides=(1,1),
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
    survival_fn: function
        Function to calculate survival rate for each layer given its idx
    stage_position: tuple/list of 2 integers
        -
    kernel_size: int, tuple/list of 2 integers
        Central 2D convolution window's Height and Width
    strides: int, tuple/list of 2 integers
        Specifying the strides of the central convolution along the height and width
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    activation: String or keras.Layer
        Activation function to use after each convolution.
    """
    Unit = BottleneckUnit

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
    
    def fwd(input):
        transform_input_fn = lambda x: pool_pad_input(input=x, filters=filters, strides=strides, data_format=data_format)

        x = StochResWrapper(
            Unit(
                filters,
                kernel_size,
                strides=strides,
                activation=activation,
                data_format=data_format,
                **kwargs
            ),
            survival_rate=survival_fn(stage_start_pos),
            transform_input_fn=transform_input_fn,
            **kwargs
        )(input)
        
        for i in range(1, layers):
            x = StochResWrapper(
                Unit(
                    filters,
                    kernel_size,
                    strides=(1,1),
                    activation=activation,
                    data_format=data_format,
                    **kwargs
                ),
                survival_rate=survival_fn(stage_start_pos + i),
                **kwargs
            )(x)

        return x
    
    return fwd

def ResNetSD(conv_per_stage,
             min_survival_p,
             img_size=(32,32),
             img_channels=3,
             classes=10,
             filters=16,
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
    activation: string, keras.Layer
        Activation function to use after each convolution.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """

    input_shape = (*img_size, img_channels) if data_format=='channels_last' else (img_channels, *img_size)
    strides = [(1,1)] + [(2,2)]*3
    expansion = 4
    
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
        x = StochasticDepthStage(layers=layers,
                                 filters=filters * expansion,
                                 survival_fn=survival_fn,
                                 stage_start_pos=layer_cnt,
                                 kernel_size=(3,3),
                                 strides=strides,
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
                                 name=f'{f"Wide{filters}" if filters != 256 else ""}ResNet{sum(conv_per_stage) * 3 + 2}SD_p{min_survival_p}')

############## Predefined Nets ##############
def ResNet50SD(img_size=(32,32),
               img_channels=3,
               min_survival_p=0.7,
               classes=10,
               activation='relu',
               data_format='channels_last',
               **kwargs):
    """
    ResNet50SD model for CIFAR/SVHN
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
    return ResNetSD(conv_per_stage=[3, 4, 6, 3],
                    min_survival_p=min_survival_p,
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet101SD(img_size=(32,32),
                min_survival_p=0.5,
                img_channels=3,
                classes=10,
                activation='relu',
                data_format='channels_last',
                **kwargs):
    """
    ResNet101SD model for CIFAR/SVHN
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
    return ResNetSD(conv_per_stage=[3, 4, 23, 3],
                    min_survival_p=min_survival_p,
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet152SD(img_size=(32,32),
                min_survival_p=0.35,
                img_channels=3,
                classes=10,
                activation='relu',
                data_format='channels_last',
                **kwargs):
    """
    ResNet152SD model for CIFAR/SVHN
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
    return ResNetSD(conv_per_stage=[3, 8, 36, 3],
                    min_survival_p=min_survival_p,
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

