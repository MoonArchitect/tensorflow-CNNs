import tensorflow as tf
import tensorflow.keras as nn

from models.ResNetV2 import AA_downsampling
from models.layers.pre_act_conv import PreActConv
from utils.registry import register_model
from .layers import get_activation_layer, get_channels, _make_divisible


"""
    Implementation of SeNet for CIFAR10/32x32
    
    From: Squeeze-and-Excitation Networks, https://arxiv.org/pdf/1709.01507.pdf.
    By: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
"""


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
        self.downsampler = None

        if strides != (1, 1):
            self.downsampler = AA_downsampling(
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


    def build(self, input_shape):
        super().build(input_shape)

    
    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)

        if self.downsampler:
            x = self.downsampler(x)
        
        outputs = self.block3(x)

        return outputs


class SEBlock(nn.layers.Layer):
    """
    Arguments:
    ----------
    in_channels: int
        -
    reduction: int
        -
    data_formt: str, 'channels_last' or 'channels_first'
        -
    """
    def __init__(self,
                 in_channels,
                 reduction = 4,
                 internal_activation='relu',
                 final_activation='sigmoid',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(**kwargs)

        assert data_format in ['channels_last', 'channels_first']
        self.axis = [1, 2] if data_format == 'channels_last' else [2, 3]
        
        self.pool = nn.layers.GlobalAvgPool2D(keepdims=True)
        self.dense1 = nn.layers.Dense(_make_divisible(in_channels // reduction, 8))
        self.relu = nn.layers.ReLU()
        self.dense2 = nn.layers.Dense(in_channels, activation='sigmoid')

        # self.conv1 = nn.layers.Conv2D(in_channels // reduction, 1, kernel_initializer=nn.initializers.he_normal(), data_format=data_format)
        # self.internal_act = get_activation_layer(internal_activation)
        # self.conv2 = nn.layers.Conv2D(in_channels, 1, kernel_initializer=nn.initializers.he_normal(), data_format=data_format)
        # self.final_act = get_activation_layer(final_activation)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = tf.reduce_mean(inputs, self.axis, keepdims=True)
        # x = self.conv1(x)
        # x = self.internal_act(x)
        # x = self.conv2(x)
        # x = self.final_act(x)
        return x * inputs


class SEBottleneck(nn.layers.Layer):
    """
    """
    def __init__(self,
                 filters,
                 reduction=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 expansion=4,
                 activation='RELu',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(**kwargs)

        self.filters = filters * expansion
        self.input_pool = None
        self.pad = None

        if strides != (1, 1):
            self.input_pool = nn.layers.AvgPool2D(strides, data_format=data_format)

        self.bottleneck = BottleneckUnit(filters=filters * expansion,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         groups=1,
                                         expansion=expansion,
                                         activation=activation,
                                         data_format=data_format)

        self.se_block = SEBlock(filters * expansion,
                                reduction=reduction,
                                data_format=data_format)


    def build(self, input_shape):  # assumed data_format == 'channels_last'
        if input_shape[3] != self.filters:
            self.pad = [(self.filters - input_shape[3]) // 2] * 2

        super().build(input_shape)


    def call(self, inputs):
        x = self.bottleneck(inputs)
        x = self.se_block(x)
        outputs = x

        if self.input_pool:
            inputs = self.input_pool(inputs)
        
        if self.pad:
            inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], self.pad])

        return outputs + inputs


def SEStage(layers,
            filters,
            reduction=16,
            kernel_size=(3, 3),
            strides=(1, 1),
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
    reduction: int
        Reduction value for SE layers
    kernel_size: int, tuple/list of 2 integers
        Central 2D convolution window's Height and Width
    strides: int, tuple/list of 2 integers
        Specifying the strides of the central convolution along the height and width
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs.
    activation: String or keras.Layer
        Activation function to use after each convolution.
    """

    def fwd(input):
        x = SEBottleneck(
            filters=filters,
            reduction=reduction,
            kernel_size=kernel_size,
            strides=strides,
            expansion=4,
            activation=activation,
            data_format=data_format,
            **kwargs
        )(input)

        for _ in range(layers - 1):
            x = SEBottleneck(
                filters=filters,
                reduction=reduction,
                kernel_size=kernel_size,
                strides=(1, 1),
                expansion=4,
                activation=activation,
                data_format=data_format,
                **kwargs
            )(x)
        
        return x
    
    return fwd


def SeNet(conv_per_stage,
          reduction=16,
          width_factor=1.0,
          activation='relu',
          data_format='channels_last',
          input_shape=(32, 32, 3),
          classes=10,
          **kwargs):
    """
    Template for Bottleneck SeNet with 4 stages
    Parameters:
    -----------
    conv_per_stage: list, tuple
        Number of residual blocks in each stage
    reduction: int
        Reduction value for SE layers
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Activation function to use after each convolution
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs
    input_shape: list, tuple
        Shape of an input image
    classes: int
        Number of classification classes
    """
    assert len(conv_per_stage) == 4, "conv_per_stage should have 4 stages"

    strides = [(1, 1)] + [(2, 2)] * 3
    filters = _make_divisible(16 * width_factor, 8)

    input = tf.keras.layers.Input(shape=input_shape)
    
    x = input
    # if data_format == 'channels_last':
    #     x = tf.transpose(input, [0, 3, 1, 2])
    #     data_format = 'channels_first'
    
    # Initial Convolution
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=data_format,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)

    # Residual Stages
    for layers, strides in zip(conv_per_stage, strides):
        x = SEStage(
            layers=layers,
            filters=filters,
            reduction=reduction,
            kernel_size=(3, 3),
            strides=strides,
            data_format=data_format,
            activation=activation,
            **kwargs
        )(x)
        
        filters *= 2

    x = tf.keras.layers.BatchNormalization(-1 if data_format == 'channels_last' else 1)(x)
    x = get_activation_layer(activation)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'{f"Wide{filters}" if filters != 256 else ""}SeNet{sum(conv_per_stage) * 3 + 2}')

############## Predefined Nets ##############


@register_model
def SeNet26(width_factor=1,
            activation='relu',
            **kwargs):
    """
    SeNet18 model for CIFAR/SVHN
    Parameters:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network
    Returns:
    ----------
    keras.Model
    """
    return SeNet(conv_per_stage=[2, 2, 2, 2],
                 width_factor=width_factor,
                 activation=activation,
                 **kwargs)


@register_model
def SeNet35(width_factor=1,
            activation='relu',
            **kwargs):
    """
    SeNet35b model for CIFAR/SVHN
    Parameters:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network
    Returns:
    ----------
    keras.Model
    """
    return SeNet(conv_per_stage=[2, 3, 4, 2],
                 width_factor=width_factor,
                 activation=activation,
                 **kwargs)


@register_model
def SeNet50(width_factor=1,
            activation='relu',
            **kwargs):
    """
    SeNet50b model for CIFAR/SVHN
    Parameters:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network
    Returns:
    ----------
    keras.Model
    """
    return SeNet(conv_per_stage=[3, 4, 6, 3],
                 width_factor=width_factor,
                 activation=activation,
                 **kwargs)


@register_model
def SeNet101(width_factor=1,
             activation='relu',
             **kwargs):
    """
    SeNet101 model for CIFAR/SVHN
    Parameters:
    ----------
    width_factor: float
        Width coefficient of the network's layers
    activation: string, keras.Layer
        Main activation function of the network
    Returns:
    ----------
    keras.Model
    """
    return SeNet(conv_per_stage=[3, 4, 23, 3],
                 width_factor=width_factor,
                 activation=activation,
                 **kwargs)

