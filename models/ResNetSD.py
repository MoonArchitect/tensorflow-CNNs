import tensorflow as tf
import tensorflow.keras as nn
from .ResNetV2 import AA_downsampling
from .layers import get_activation_layer, get_channels, linear_decay_fn, PreActConv
from utils.registry import register_model

"""
    Implementation of ResNet with Stochastic Depth for CIFAR/SVHN/32x32

    From: Deep Networks with Stochastic Depth, https://arxiv.org/abs/1603.09382
    By: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
"""


class StochasticBottleneckUnit(nn.layers.Layer):
    """
    Stochastic Bottleneck Unit from ResNetSD
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
    main_path:
        input->[PreActConv1x1]->[PreActConv3x3]->[PreActConv1x1] + input->output
    alternate_path:
        input->output
    inference_path:
        input->([PreActConv1x1]->[PreActConv3x3]->[PreActConv1x1])*p + input->output

    """
    def __init__(self,
                 filters,
                 main_probability,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 groups=1,
                 expansion=4,
                 activation='RELu',
                 data_format='channels_last',
                 **kwargs):
        super(StochasticBottleneckUnit, self).__init__(**kwargs)
        assert data_format == 'channels_last'
        assert len(strides) == 2 and strides[0] == strides[1]
        assert filters // expansion % groups == 0 and filters // expansion // groups > 0
        
        self.main_probability = tf.constant(main_probability, dtype=self._compute_dtype)
        self.filters = filters
        
        self.input_pool = None
        self.downsampler = None
        self.pad = None

        if strides != (1, 1):
            self.input_pool = nn.layers.AvgPool2D(strides, data_format=data_format)
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


    def build(self, input_shape):  # assumed data_format == 'channels_last'
        if input_shape[3] != self.filters:
            self.pad = [(self.filters - input_shape[3]) // 2] * 2

        return super().build(input_shape)

    
    def call(self, inputs, training=None):

        def inputs_transform(inputs):
            if self.input_pool:
                inputs = self.input_pool(inputs)
            
            if self.pad:
                inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], self.pad])
            
            return inputs
        
        def training_path(inputs):
            x = self.block1(inputs)
            x = self.block2(x)

            if self.downsampler:
                x = self.downsampler(x)
            
            outputs = self.block3(x)

            return outputs
        

        if training:
            return tf.cond(
                tf.random.uniform([1], 0, 1, dtype=self._compute_dtype) < self.main_probability,
                lambda: training_path(inputs) + inputs_transform(inputs),
                lambda: inputs_transform(inputs)
            )
        else:
            return tf.scalar_mul(self.main_probability, training_path(inputs)) + inputs_transform(inputs)


def StochasticDepthStage(layers,
                         filters,
                         survival_fn,
                         stage_start_pos,
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
    def fwd(input):
        x = StochasticBottleneckUnit(
            filters=filters,
            kernel_size=kernel_size,
            main_probability=survival_fn(stage_start_pos),
            strides=strides,
            activation=activation,
            data_format=data_format,
            **kwargs
        )(input)
        
        for i in range(1, layers):
            x = StochasticBottleneckUnit(
                filters=filters,
                kernel_size=kernel_size,
                main_probability=survival_fn(stage_start_pos + i),
                strides=(1, 1),
                activation=activation,
                data_format=data_format,
                **kwargs
            )(x)

        return x
    
    return fwd


def ResNetSD(conv_per_stage,
             min_survival_p,
             input_shape=(32, 32, 3),
             classes=10,
             filters=16,
             activation='relu',
             data_format='channels_last',
             **kwargs):
    """
    ResNet with Stochastic Depth
    -----------
    Parameters:
    conv_per_stage: list, tuple
        Number of residual blocks in each stage
    input_shape: list, tuple
        Shape of an input image
    classes: int
        Number of classification classes.
    filters: int
        Number of filters in stem layer
    activation: string, keras.Layer
        Activation function to use after each convolution.
    data_format: 'channels_last' or 'channels_first'
        The ordering of the dimensions in the inputs.
    """

    strides = [(1, 1)] + [(2, 2)] * 3
    expansion = 4
    
    survival_fn = linear_decay_fn((0, 1), (sum(conv_per_stage), min_survival_p))
    layer_cnt = 1  # ...
    

    input = tf.keras.layers.Input(shape=input_shape)

    x = input
    # if data_format == 'channels_last':
    #     x = tf.transpose(input, [0, 3, 1, 2])
    #     data_format = 'channels_first'
        
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
                                 kernel_size=(3, 3),
                                 strides=strides,
                                 data_format=data_format,
                                 activation=activation,
                                 **kwargs)(x)
        filters *= 2
        layer_cnt += layers

    x = tf.keras.layers.BatchNormalization(-1 if data_format == 'channels_last' else 1)(x)
    x = get_activation_layer(activation)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input,
                                 outputs=output,
                                 name=f'{f"Wide{filters}" if filters != 256 else ""}ResNet{sum(conv_per_stage) * 3 + 2}SD_p{min_survival_p}')


############## Predefined Nets ##############
@register_model
def ResNet50SD(min_survival_p=0.7,
               activation='relu',
               **kwargs):
    """
    ResNet50b model for CIFAR/SVHN
    Parameters:
    ----------
    min_survival_p: float
        last layer's survival probability
    activation: string, keras.Layer
        Main activation function of the network.
    Returns:
    ----------
    keras.Model
    """
    return ResNetSD(conv_per_stage=[3, 4, 6, 3],
                    min_survival_p=min_survival_p,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet101SD(min_survival_p=0.45,
                activation='relu',
                **kwargs):
    """
    ResNet101 model for CIFAR/SVHN
    Parameters:
    ----------
    min_survival_p: float
        last layer's survival probability
    activation: string, keras.Layer
        Main activation function of the network.
    Returns:
    ----------
    keras.Model
    """
    return ResNetSD(conv_per_stage=[3, 4, 23, 3],
                    min_survival_p=min_survival_p,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet152SD(min_survival_p=0.35,
                activation='relu',
                **kwargs):
    """
    ResNet152b model for CIFAR/SVHN
    Parameters:
    ----------
    min_survival_p: float
        last layer's survival probability
    activation: string, keras.Layer
        Main activation function of the network.
    Returns:
    ----------
    keras.Model
    """
    return ResNetSD(conv_per_stage=[3, 8, 36, 3],
                    min_survival_p=min_survival_p,
                    activation=activation,
                    **kwargs)


@register_model
def ResNet170SD(min_survival_p=0.35,
                activation='relu',
                **kwargs):
    """
    ResNet170b model for CIFAR/SVHN
    Parameters:
    ----------
    min_survival_p: float
        last layer's survival probability
    activation: string, keras.Layer
        Main activation function of the network.
    Returns:
    ----------
    keras.Model
    """
    return ResNetSD(conv_per_stage=[4, 10, 36, 6],
                    min_survival_p=min_survival_p,
                    activation=activation,
                    **kwargs)
