import tensorflow as tf

__all__ = ['Mish']


@tf.function
@tf.custom_gradient
def Mish_fn(x):
    sx = tf.sigmoid(x)
    th = tf.tanh(tf.math.softplus(x))
    k = (th + x * sx * (1 - th * th))  # TODO write proper kernel
    
    def grad(dy):
        # sx = tf.sigmoid(x)
        # return dy * (th + x * sx * (1 - th * th))
        return dy * k
    
    return x * th, grad


class Mish(tf.keras.layers.Layer):
    """
    Mish - Self regularized non-monotonic activation function, f(x) = x*tanh(softplus(x)).
    From "Mish: A Self Regularized Non-Monotonic Activation Function", https://arxiv.org/abs/1908.08681
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs):
        return Mish_fn(inputs)
