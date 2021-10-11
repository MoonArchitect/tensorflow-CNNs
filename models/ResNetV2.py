import tensorflow as tf
import tensorflow.keras as nn

from utils.registry import register_model
from .layers import PreActConv, AntiAliasDownsampling, get_activation_layer, _make_divisible

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
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 groups=1,
                 activation='RELu',
                 data_format='channels_last',
                 **kwargs):
        super(BasicUnit, self).__init__(**kwargs)

        self.filters = filters
        self.input_pool = None
        self.pad = None

        if strides != (1, 1):
            self.input_pool = nn.layers.AvgPool2D(strides, data_format=data_format)

        self.block1 = PreActConv(
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
        self.block2 = PreActConv(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            groups=groups,
            activation=activation,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            **kwargs
        )


    def build(self, input_shape):  # assumed data_format == 'channels_last'
        if input_shape[3] != self.filters:
            self.pad = [(self.filters - input_shape[3]) // 2] * 2

        super().build(input_shape)


    def call(self, inputs):
        x = self.block1(inputs)
        outputs = self.block2(x)
        
        if self.input_pool:
            inputs = self.input_pool(inputs)

        if self.pad:
            inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], self.pad])

        return outputs + inputs


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
    expansion: int
        -
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
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 groups=1,
                 expansion=4,
                 activation='RELu',
                 data_format='channels_last',
                 **kwargs):
        super(BottleneckUnit, self).__init__(**kwargs)
        assert data_format == 'channels_last'
        assert len(strides) == 2 and strides[0] == strides[1]
        assert filters // expansion % groups == 0 and filters // expansion // groups > 0
        
        self.filters = filters
        
        self.input_pool = None
        self.downsampler = None
        self.pad = None

        if strides != (1, 1):
            self.input_pool = nn.layers.AvgPool2D(strides, data_format=data_format)
            self.downsampler = AntiAliasDownsampling(
                filters // expansion,
                data_format=data_format
            )

        self.block1 = PreActConv(
            filters=filters // expansion,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            groups=1,
            activation=activation,
            use_bias=False,
            kernel_regularizer=nn.regularizers.l2(0.0001),
            **kwargs
        )

        
        self.block2 = PreActConv(
            filters=filters // expansion,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            groups=groups,
            activation=activation,
            use_bias=False,
            kernel_regularizer=nn.regularizers.l2(0.0001),
            **kwargs
        )
        
        
        self.block3 = PreActConv(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            groups=1,
            activation=activation,
            use_bias=False,
            kernel_regularizer=nn.regularizers.l2(0.0001),
            **kwargs
        )


    def build(self, input_shape):  # assumed data_format == 'channels_last'
        if input_shape[3] != self.filters:
            self.pad = [(self.filters - input_shape[3]) // 2] * 2

        super().build(input_shape)

    
    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)

        if self.downsampler:
            x = self.downsampler(x)
        
        outputs = self.block3(x)
        
        if self.input_pool:
            inputs = self.input_pool(inputs)
        
        if self.pad:
            inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], self.pad])

        return outputs + inputs


def ResStage(layers,
             filters,
             kernel_size=(3, 3),
             strides=(1, 1),
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

    model_unit = BottleneckUnit if bottleneck else BasicUnit
    
    def fwd(input):
        x = model_unit(
            filters,
            kernel_size,
            strides=strides,
            activation=activation,
            data_format=data_format,
            **kwargs
        )(input)
        
        for _ in range(layers - 1):
            x = model_unit(
                filters,
                kernel_size,
                activation=activation,
                data_format=data_format,
                **kwargs
            )(x)
        
        return x
    
    return fwd


def ResNetV2(conv_per_stage,
             width_factor=1,
             bottleneck=False,
             activation='relu',
             data_format='channels_last',
             input_shape=(32, 32, 3),
             classes=10,
             **kwargs):
    """
    Template for Bottleneck ResNet with 4 stages
    -----------
    Parameters:
    conv_per_stage: list, tuple
        Number of residual blocks in each stage
    width_factor: float
        Width coefficient of the network's layers
    bottleneck: bool
        Whether to use Bottleneck Residual units
    activation: string, keras.Layer
        Activation function to use after each convolution
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs
    input_shape: list, tuple
        Shape of an input image
    classes: int
        Number of classification classes
    """
    
    strides = [(1, 1)] + [(2, 2)] * 3
    expansion = 4 if bottleneck else 1
    filters = _make_divisible(16 * width_factor, 8)


    input = tf.keras.layers.Input(shape=input_shape)
    
    x = input
    # if data_format == 'channels_last':
    #    x = tf.transpose(input, [0, 3, 1, 2])
    #    data_format = 'channels_first'
    
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
        x = ResStage(layers=layers,
                     filters=filters * expansion,
                     kernel_size=(3, 3),
                     strides=strides,
                     data_format=data_format,
                     activation=activation,
                     bottleneck=bottleneck,
                     **kwargs)(x)
        filters *= 2

    x = tf.keras.layers.BatchNormalization(-1 if data_format == 'channels_last' else 1)(x)
    x = get_activation_layer(activation)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'{"Wide" if filters != 256 else ""}ResNet{sum(conv_per_stage) * (3 if bottleneck else 2) + 2}{f"-{filters // 256}" if filters != 256 else ""}')


############## Predefined Nets ##############
@register_model
def ResNet18(width_factor=1,
             activation='relu',
             **kwargs):
    """
    ResNet18 model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network

    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[2, 2, 2, 2],
                    width_factor=width_factor,
                    bottleneck=False,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet34(width_factor=1,
             activation='relu',
             **kwargs):
    """
    ResNet34 model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network

    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[3, 4, 6, 3],
                    width_factor=width_factor,
                    bottleneck=False,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet35(width_factor=1,
             activation='relu',
             **kwargs):
    """
    ResNet35b model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network

    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[2, 3, 4, 2],
                    width_factor=width_factor,
                    bottleneck=True,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet50(width_factor=1,
             activation='relu',
             **kwargs):
    """
    ResNet50b model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network

    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[3, 4, 6, 3],
                    width_factor=width_factor,
                    bottleneck=True,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet101(width_factor=1,
              activation='relu',
              **kwargs):
    """
    ResNet101 model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network

    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[3, 4, 23, 3],
                    width_factor=width_factor,
                    bottleneck=True,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet152(width_factor=1,
              activation='relu',
              **kwargs):
    """
    ResNet152b model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network
    
    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[3, 8, 36, 3],
                    width_factor=width_factor,
                    bottleneck=True,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet170(width_factor=1,
              activation='relu',
              **kwargs):
    """
    ResNet170b model for CIFAR/SVHN
    ----------
    Parameters:
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network

    ----------
    Returns:
    keras.Model
    """
    return ResNetV2(conv_per_stage=[5, 9, 37, 5],  # alternative [3, 10, 40, 3]
                    width_factor=width_factor,
                    bottleneck=True,
                    activation=activation,
                    **kwargs)

