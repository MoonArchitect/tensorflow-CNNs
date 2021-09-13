import tensorflow as tf
import tensorflow.keras as nn
from .layers import get_activation_layer, get_channels
# from utils.registry import register_model


"""
    Implementation of ResNetSE for CIFAR10/32x32
    
    From: Squeeze-and-Excitation Networks, https://arxiv.org/pdf/1709.01507.pdf.
    By: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
"""


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
        self.layer = nn.Sequential()
        self.layer.add(
            nn.layers.Conv2D(
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
        self.layer.add(nn.layers.BatchNormalization(-1 if data_format == 'channels_last' else 1))
        self.layer.add(nn.layers.ReLU())
        self.layer.add(
            nn.layers.Conv2D(
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
        )
        self.layer.add(nn.layers.BatchNormalization(-1 if data_format == 'channels_last' else 1))

    def call(self, inputs):
        return self.layer(inputs)


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
        
        self.conv1 = nn.layers.Conv2D(in_channels // reduction, 1, kernel_initializer=nn.initializers.he_normal(), data_format=data_format)
        self.internal_act = get_activation_layer(internal_activation)
        self.conv2 = nn.layers.Conv2D(in_channels, 1, kernel_initializer=nn.initializers.he_normal(), data_format=data_format)
        self.final_act = get_activation_layer(final_activation)

    def call(self, inputs):
        x = tf.reduce_mean(inputs, self.axis, keepdims=True)
        x = self.conv1(x)
        x = self.internal_act(x)
        x = self.conv2(x)
        x = self.final_act(x)
        return x * inputs


class SEBottleneck(nn.layers.Layer):
    """
    """
    def __init__(self,
                 filters,
                 reduction=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 # expansion=4,
                 activation='RELu',
                 data_format='channels_last',
                 input_fn=lambda x: x,
                 **kwargs):
        super().__init__(**kwargs)

        self.Bottleneck = BasicUnit(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            groups=1,
            activation=activation,
            data_format=data_format
        )

        self.se_block = SEBlock(filters, reduction=reduction, data_format=data_format)

        self.input_fn = input_fn


    def call(self, inputs):
        x = self.Bottleneck(inputs)
        x = self.se_block(x)
        return self.input_fn(inputs) + x


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
            sc = tf.pad(sc, [[0, 0], [0, 0], [0, 0], pad] if data_format == 'channels_last' else [[0, 0], pad, [0, 0], [0, 0]])
        return sc
    

    def fwd(input):
        x = SEBottleneck(
            filters,
            reduction,
            kernel_size,
            strides,
            4,
            activation,
            data_format,
            lambda x: pool_pad_input(input=x, filters=filters, strides=strides, data_format=data_format),
            **kwargs
        )(input)

        for _ in range(layers - 1):
            x = SEBottleneck(
                filters=filters,
                reduction=reduction,
                kernel_size=kernel_size,
                strides=(1, 1),
                activation=activation,
                data_format=data_format,
                **kwargs
            )(x)
        
        return x
    
    return fwd


def SeNet(conv_per_stage,
          input_shape=(32, 32, 3),
          reduction=16,
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
    input_shape: list, tuple
        Shape of an input image
    reduction: int
        Reduction value for SE layers
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

    strides = [(1, 1)] + [(2, 2)] * 3

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
        if layers == 0:
            continue
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
                                 name=f'{f"Wide{filters}" if filters != 256 else ""}SeNet{sum(conv_per_stage) * 2  + 2}')

############## Predefined Nets ##############

