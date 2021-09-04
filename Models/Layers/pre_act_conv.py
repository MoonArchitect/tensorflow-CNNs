import tensorflow.keras as nn
from .utils import get_activation_layer


class PreActConv(nn.layers.Layer):
    """
    PreActivation Convolution
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
    padding: 'valid' or 'same'
        -
    use_bias: bool
        Whether the layer uses a bias vector.
    kernel_regularizer: tf.keras.regularizers.Regularizer
        Kernel Regularizer for Convolutional Layer
    data_format: 'channels_last' (default) or 'channels_first'
        The ordering of the dimensions in the inputs.
        'channels_last' = (batch_size, height, width, channels)
        'channels_first' = (batch_size, channels, height, width).
    Architecture:
    -------------
    BN + Activation + Convolution
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 groups=1,
                 activation='RELu',
                 use_bias=False,
                 kernel_regularizer=nn.regularizers.l2(0.0001),
                 # shape=None,
                 **kwargs):
        super(PreActConv, self).__init__(**kwargs)
        
        assert(data_format in ['channels_last', 'channels_first'])

        self.layer = nn.Sequential()
        self.layer.add(nn.layers.BatchNormalization(-1 if data_format == 'channels_last' else 1))
        self.layer.add(get_activation_layer(activation))
        self.layer.add(
            nn.layers.Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=data_format,
                             groups=groups,
                             use_bias=use_bias,
                             kernel_regularizer=kernel_regularizer,
                             **kwargs)
        )
    
    # def build(self, input_shape):
    #    self.z = tf.zeros(input_shape[1:])
    #    return super().build(input_shape)

    def call(self, inputs):
        return self.layer(inputs)
