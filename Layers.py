import tensorflow as tf
import tensorflow.keras as nn
from tensorflow.keras.backend import in_test_phase


@tf.function
@tf.custom_gradient
def Mish_fn(x):
    """
    """
    sx = tf.sigmoid(x)
    th = tf.tanh(tf.math.softplus(x))
    k = (th + x * sx * (1 - th * th))
    def grad(dy):
        return dy * k
    
    return x * th, grad

class Mish(nn.layers.Layer):
    """
    Mish - Self regularized non-monotonic activation function, f(x) = x*tanh(softplus(x)).
    From "Mish: A Self Regularized Non-Monotonic Activation Function", https://arxiv.org/abs/1908.08681
    """
    def __init__(self, *kwargs):
        super(Mish, self).__init__(kwargs)

    def call(self, x):
        return Mish_fn(x)


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
                 **kwargs):
        super(PreActConv, self).__init__(**kwargs)
        
        assert(data_format in ['channels_last', 'channels_first'])

        self.layer = nn.Sequential()
        self.layer.add(nn.layers.BatchNormalization(-1 if data_format=='channels_last' else 1))
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
    
    def call(self, input):
        return self.layer(input)


def get_activation_layer(activation, **kwargs):
    """
    """
    assert (activation is not None)

    if isinstance(activation, str):
        activation = activation.lower()
        if activation == "relu":
            return nn.layers.ReLU(**kwargs)
        elif activation == "prelu":
            return nn.layers.PReLU(**kwargs)
        elif activation == "mish":
            return Mish(**kwargs)
        elif activation == "swish":
            NotImplementedError("Swish is not implemented")
            #return nn.layers.Lambda(lambda x: nn.activations.swish(x))
        elif activation == "hswish":
            NotImplementedError("Hswish is not implemented")
            return None
        elif activation == "sigmoid":
            return nn.layers.Lambda(lambda x: tf.nn.sigmoid(x))
        elif activation == "tanh":
            return nn.layers.Lambda(lambda x: tf.nn.tanh(x))
        else:
            raise NotImplementedError(f"{activation} is not implemented")
    else:
        assert (isinstance(activation, nn.layers.Layer))
        return activation


def get_channels(x, data_format='channels_last'):
    return x.shape[3] if data_format=='channels_last' else x.shape[1]

#region Decay Functions
def linear_decay_fn(start_pos_val,
                    end_pos_val,
                    name="Linear Decay"):
    """
    Returns function to generate values with linear decay corresponding to (start/end)_pos_val
    Arguments:
    ----------
    start_pos_val: tuple/list of 2 integers
        -
    end_pos_val: tuple/list of 2 integers
        -
    Returns:
    --------
    Python function
        -> fn(x), takes 1 argument, position:float
        -> Returns value:float
    """
    # Swap for convenience if 
    if start_pos_val[0] > end_pos_val[0]:
        temp = end_pos_val
        end_pos_val = start_pos_val
        start_pos_val = temp

    def fn(x):
        if not start_pos_val[0] <= x <= end_pos_val[0]:
            raise ValueError(f"{name} | Position {x} is not in the range of specified positions: {start_pos_val[0]}->{end_pos_val[1]}")

        return start_pos_val[1] + (x - start_pos_val[0]) / (end_pos_val[0] - start_pos_val[0]) * (end_pos_val[1] - start_pos_val[1])
    return fn

def linear_decay(x, start_pos_val,
                 end_pos_val):
    return linear_decay_fn(start_pos_val, end_pos_val)(x)

#endregion


