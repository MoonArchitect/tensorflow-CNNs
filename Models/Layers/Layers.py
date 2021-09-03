import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as nn 
from tensorflow.keras.backend import in_test_phase

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

class AA_downsampling(nn.layers.Layer):
    """
    """
    def __init__(self, in_channels, data_format, **kwargs):
        super().__init__(**kwargs)
        self.data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'

        a = tf.constant([1., 2., 1.], dtype=self._compute_dtype)
        filter = (a[:, None] * a[None, :])
        filter = filter / tf.reduce_sum(filter)
        self.filter = tf.repeat(filter[:, :, None, None], [in_channels], axis=2)
        self.strides = [1,2,2,1] if data_format == 'channels_last' else [1,1,2,2]

    def call(self, input):
        return tf.nn.depthwise_conv2d(input, self.filter, self.strides, "SAME", data_format=self.data_format)

@tf.function
@tf.custom_gradient
def Mish_fn(x):
    """
    """
    sx = tf.sigmoid(x)
    th = tf.tanh(tf.math.softplus(x))
    k = (th + x * sx * (1 - th * th))
    def grad(dy):
        #sx = tf.sigmoid(x)
        #return dy * (th + x * sx * (1 - th * th))
        return dy * k
    
    return x * th, grad

class Mish(nn.layers.Layer):
    """
    Mish - Self regularized non-monotonic activation function, f(x) = x*tanh(softplus(x)).
    From "Mish: A Self Regularized Non-Monotonic Activation Function", https://arxiv.org/abs/1908.08681
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

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
                 shape=None, **kwargs):
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
    
    #def build(self, input_shape):
    #    self.z = tf.zeros(input_shape[1:])
    #    return super().build(input_shape)

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
            return nn.layers.Activation(tf.nn.swish, **kwargs)
        elif activation == "hswish":
            print("hswish is not implemented efficiently yet, using swish instead")
            return nn.layers.Activation(tf.nn.swish, **kwargs)
        elif activation == "sigmoid":
            return nn.layers.Activation('sigmoid', **kwargs)
        elif activation == "tanh":
            return nn.layers.Activation('tanh', **kwargs)
        else:
            raise NotImplementedError(f"{activation} is not implemented")
    else:
        assert (isinstance(activation, nn.layers.Layer))
        return activation


def get_channels(x, data_format='channels_last'):
    return x.shape[3] if data_format=='channels_last' else x.shape[1]

#region Decay Functions
def decay_fn(pos_val_pairs, exponent = 1):
    raise NotImplementedError("decay_fn is not implemented")

    assert all(
            [pos_val_pairs[0][i] > pos_val_pairs[0][i - 1] for i in range(1, len(pos_val_pairs[0]))]
        ), f"Positions must be in increasing order, recieved {[i[0] for i in pos_val_pairs]}"
    
    def fn(x):
        #for i in range(len(pos_val_pairs), 0):
        #    if x > pos_val_pairs[i][0]:
        #        return x        
        
        # If value has not been returned yet
        raise ValueError(f"Position {x} is not it range of specified positions: {[i[0] for i in pos_val_pairs]}")
    return fn

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

