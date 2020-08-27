import tensorflow as tf
from tensorflow.keras.backend import in_test_phase
import tensorflow.keras as nn 


class ShakeDrop(nn.layers.Layer):
    def __init__(self, p = 1, alpha=(0, 0), beta=(0, 1), *kwargs):
        super(ShakeDrop, self).__init__(kwargs)
        self.p = tf.constant([p], dtype='float32', name='p')
        self.alpha = tf.constant(alpha, dtype='float32', name='alpha')
        self.beta = tf.constant(beta, dtype='float32', name='beta')
        pass
    
    #@tf.function
    @tf.custom_gradient
    def _ShakeDrop(self, input):
        b = tf.cond(tf.random.uniform([1], 0, 1)[0] <= self.p, lambda: tf.constant(1.0), lambda: tf.constant(0.0))
        def grad(dy):
            beta = tf.random.uniform([128, 1, 1, 1], self.beta[0], self.beta[1])
            return (b + beta - beta * b) * dy
        #alpha = tf.random.uniform([1], self.alpha[0], self.alpha[1])
        #return input, grad
        return tf.scalar_mul(b, input), grad
        #in_test_phase((self.p + (self.alpha[1] + self.alpha[0]) * (1 - self.p) / 2)[0], (b + alpha - b*alpha)[0])

    def call(self, input):
        #return tf.cond(tf.random.uniform([1], 0, 1) <= self.p, lambda: tf.scalar_mul(tf.constant(1.0), input), lambda: tf.scalar_mul(tf.constant(0.0), input))
        return self._ShakeDrop(input)


class Mish(nn.layers.Layer):
    def __init__(self, *kwargs):
        super(Mish, self).__init__(kwargs)

    @tf.function
    @tf.custom_gradient
    def MishFn(self, x):
        
        def grad(dy):
            sp = tf.math.log1p(tf.exp(x)) if x < 20 else x
            grad_sp = 1 - tf.exp(-sp)
            tsp = tf.tanh(sp)
            grad_tsp = (1 - tsp*tsp) * grad_sp
            grad = x * grad_tsp + tsp
            return dy * grad
        
        return x * tf.tanh(tf.math.softplus(x)), grad

    def call(self, x):
        return MishFn(x)#x * tf.tanh(tf.math.softplus(x))


def PreActConv(filters,
               kernel_size,
               strides=1,
               padding='same',
               data_format='channels_last',
               groups=1,
               activation='RELu',
               use_bias=False,
               kernel_regularizer=nn.regularizers.l2(0.0001),
               **kwargs):
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
    def f(input):
        assert(data_format in ['channels_last', 'channels_first'])

        x = nn.layers.BatchNormalization(-1 if data_format=='channels_last' else 1)(input)
        x = get_activation_layer(activation)(x)
        x = nn.layers.Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=data_format,
                             groups=groups,
                             use_bias=use_bias,
                             kernel_regularizer=kernel_regularizer,
                             **kwargs)(x)
        return x

    return f


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
            NotImplementedError("Mish is not implemented")
            return None
        elif activation == "swish":
            NotImplementedError("Swish is not implemented")
            return None
        elif activation == "hswish":
            NotImplementedError("Hswish is not implemented")
            return None
        elif activation == "sigmoid":
            return tf.nn.sigmoid
        elif activation == "tanh":
            return tf.nn.tanh
        else:
            raise NotImplementedError(f"{activation} is not implemented")
    else:
        assert (isinstance(activation, nn.layers.Layer))
        return activation


def get_channels(x, data_format='channels_last'):
    return x.shape[3] if data_format=='channels_last' else x.shape[1]



