import tensorflow as tf
import tensorflow.keras as nn
from ..Layers import PreActConv, get_activation_layer, get_channels

""" 
    Implementation of ResNet V2 for CIFAR/SVHN/32x32

    From: Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027.
    By: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
"""

############## Building Blocks ##############
class BasicUnit(nn.layers.Layer):
    """
    Basic Unit from ResNetV2
    Arguments:
    ---------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: int, tuple/list of 2 integers 
        Height and width of the 2D convolution window
    strides: int, tuple/list of 2 integers
        Specifying the strides of the convolution along the height and width
    groups: int
        The number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters
    activation: String, keras.Layer
        Activation function to use. If you don't specify anything, no activation is applied.
    data_format: String
        The ordering of the dimensions in the inputs. 
        'channels_last' = (batch_size, height, width, channels)
        'channels_first' = (batch_size, channels, height, width).
    Architecture:
    ------------
    input->[PreActConv3x3]->[PreActConv3x3] + input
    """
    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 strides=(1,1),
                 groups=1,
                 activation='RELu',
                 data_format='channels_last',
                 **kwargs):
        super(BasicUnit, self).__init__(**kwargs)
        self.layer = nn.Sequential()
        self.layer.add(
            PreActConv(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                data_format=data_format,
                groups=groups,
                activation=activation,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                **kwargs
            )
        )
        self.layer.add(
            PreActConv(
                filters=filters,
                kernel_size=kernel_size,
                strides=(1,1),
                padding='same',
                data_format=data_format,
                groups=groups,
                activation=activation,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                **kwargs
            )
        )

    def call(self, input):
        return self.layer(input)


class BottleneckUnit(nn.layers.Layer):
    """
    Bottleneck Unit from ResNetV2
    Arguments:
    ---------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: int, tuple/list of 2 integers 
        Height and width of the 2D convolution window
    strides: int, tuple/list of 2 integers
        Specifying the strides of the convolution along the height and width
    groups: int
        The number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters
    activation: String, keras.Layer
        Activation function to use. If you don't specify anything, no activation is applied.
    data_format: String
        The ordering of the dimensions in the inputs. 
        'channels_last' = (batch_size, height, width, channels)
        'channels_first' = (batch_size, channels, height, width).
    Architecture:
    ------------
    input->[PreActConv1x1]->[PreActConv3x3]->[PreActConv1x1] + input
    """
    def __init__(self,
                filters,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                activation='RELu',
                data_format='channels_last',
                **kwargs):
        super(BottleneckUnit, self).__init__(**kwargs)
        self.layer = nn.Sequential()
        self.layer.add(
            PreActConv(
                filters=filters // 4,
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                data_format=data_format,
                groups=1,
                activation=activation,
                use_bias=False,
                kernel_regularizer=nn.regularizers.l2(0.0001),
                **kwargs
            )
        )
        self.layer.add(
            PreActConv(
                filters=filters // 4,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                data_format=data_format,
                groups=groups,
                activation=activation,
                use_bias=False,
                kernel_regularizer=nn.regularizers.l2(0.0001),
                **kwargs
            )
        )
        self.layer.add(
            PreActConv(
                filters=filters,
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                data_format=data_format,
                groups=1,
                activation=activation,
                use_bias=False,
                kernel_regularizer=nn.regularizers.l2(0.0001),
                **kwargs
            )
        )
    
    def call(self, input):
        return self.layer(input) 


def ResStage(layers,
             filters,
             kernel_size=(3,3),
             strides=(1,1),
             data_format='channels_last',
             activation='relu',
             bottleneck=False,
             **kwargs):
    """
    Arguments:
    ----------
    layers: int
        Number of residual Units in that stage
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: int, tuple/list of 2 integers
        Central 2D convolution window's Height and Width
    strides: int, tuple/list of 2 integers
        Specifying the strides of the central convolution along the height and width
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    activation: String or keras.Layer
        Activation function to use after each convolution.
    bottleneck: Boolean
        Whether to use bottleneck type unit
    """

    Unit = BottleneckUnit if bottleneck else BasicUnit
    def fwd(input):
        sc = nn.layers.AvgPool2D(strides, data_format=data_format)(input) if strides != (1, 1) else input
        if get_channels(input, data_format) != filters:
            pad = [(filters - get_channels(input, data_format)) // 2] * 2
            sc = tf.pad(sc, [[0,0], [0,0], [0,0], pad] if data_format=='channels_last' else [[0,0], pad, [0,0], [0,0]])
        
        x = Unit(filters,
                 kernel_size,
                 strides=strides,
                 activation=activation,
                 data_format=data_format,
                 **kwargs)(input)
        input = nn.layers.Add()([x, sc])
        
        for i in range(layers-1):
            x = Unit(filters,
                     kernel_size,
                     activation=activation,
                     data_format=data_format,
                     **kwargs)(input)
            input = nn.layers.Add()([x, input])
        
        return input
    
    return fwd


def ResNetV2(conv_per_stage,
             img_size=(32,32),
             img_channels=3,
             classes=10,
             bottleneck=False,
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
    bottleneck: bool
        Whether to use Bottleneck Residual unit
    filters: int
        Number of filters in stem layer
    activation: string, keras.Layer
        Activation function to use after each convolution.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs. 
    """

    input_shape = (*img_size, img_channels) if data_format=='channels_last' else (img_channels, *img_size)
    strides = [(1,1)] + [(2,2)]*3
    expansion = 4 if bottleneck else 1

    input = tf.keras.layers.Input(shape=input_shape)
    # Initial Convolution
    x = tf.keras.layers.Conv2D(filters=filters, 
                               kernel_size=(3, 3), 
                               strides=(1, 1), 
                               padding='same', 
                               use_bias=False, 
                               kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    # Residual Stages
    for layers, strides in zip(conv_per_stage, strides):
        x = ResStage(layers=layers,
                     filters=filters * expansion,
                     kernel_size=(3,3),
                     strides=strides,
                     data_format=data_format,
                     activation=activation,
                     bottleneck=bottleneck,
                     **kwargs)(x)
        filters *= 2

    x = tf.keras.layers.BatchNormalization(-1 if data_format=='channels_last' else 1)(x)
    x = get_activation_layer(activation)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input, 
                                 outputs=output, 
                                 name=f'{f"Wide{filters}" if filters != 256 else ""}ResNet{sum(conv_per_stage) * 3 + 2}b')


############## Predefined Nets ##############
def ResNet18(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet18 model for CIFAR/SVHN
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
    return ResNetV2(conv_per_stage=[2, 2, 2, 2],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    bottleneck=False,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet34(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet34 model for CIFAR/SVHN
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
    return ResNetV2(conv_per_stage=[3, 4, 6, 3],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    bottleneck=False,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet35(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet35b model for CIFAR/SVHN
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
    return ResNetV2(conv_per_stage=[2, 3, 4, 2],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    bottleneck=True,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet50(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet50b model for CIFAR/SVHN
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
    return ResNetV2(conv_per_stage=[3, 4, 6, 3],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    bottleneck=True,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet101(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet101 model for CIFAR/SVHN
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
    return ResNetV2(conv_per_stage=[3, 4, 23, 3],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    bottleneck=True,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

def ResNet152(img_size=(32,32),
             img_channels=3,
             classes=10,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet152b model for CIFAR/SVHN
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
    return ResNetV2(conv_per_stage=[3, 8, 36, 3],
                    img_size=img_size,
                    img_channels=img_channels,
                    classes=classes,
                    bottleneck=True,
                    activation=activation,
                    data_format=data_format,
                    **kwargs)

