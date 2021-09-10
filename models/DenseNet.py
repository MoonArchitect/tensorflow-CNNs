import tensorflow as tf
from utils.registry import register_model

"""
    Implementation of DenseNet for CIFAR10/32x32

    From: Densely Connected Convolutional Networks, https://arxiv.org/abs/1608.06993.
    By: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
"""


def BNRConv(filters, kernel_size, strides, kernel_regularizer=tf.keras.regularizers.l2(0.0001)):
    """BN + RELu + Convolution"""
    def f(input):
        x = tf.keras.layers.BatchNormalization()(input)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=False, kernel_regularizer=kernel_regularizer)(x)
        return x
    return f

############## Dense Net ##############


def DenseNetUnit(filters):
    def f(input):
        x = BNRConv(filters * 4, (1, 1), (1, 1))(input)
        x = BNRConv(filters, (3, 3), (1, 1))(x)
        return x
    return f


def DenseNetBlock(units, filters):
    def f(input):
        for _ in range(units):
            x = DenseNetUnit(filters)(input)
            input = tf.concat([x, input], axis=3)

        return input
    return f


def DenseNet(input_shape = (32, 32, 3),
             classes = 10,
             reduction=0.5,
             growth_rate = 12,
             layers = 100):
    """
    DenseNet model for CIFAR10/SVHN/32x32 images
    Parameters:
    ----------
    Returns
    -------
        Keras Model
    """
    N = (layers - 4) // 3 // 2

    input = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(growth_rate * 2, (3, 3), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)

    x = DenseNetBlock(N, growth_rate)(x)
    x = BNRConv(int(x.shape[3] * reduction), (1, 1), (1, 1))(x)
    x = tf.keras.layers.AvgPool2D((2, 2), (2, 2))(x)

    x = DenseNetBlock(N, growth_rate)(x)
    x = BNRConv(int(x.shape[3] * reduction), (1, 1), (1, 1))(x)
    x = tf.keras.layers.AvgPool2D((2, 2), (2, 2))(x)

    x = DenseNetBlock(N, growth_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output = tf.keras.layers.Dense(classes)(x)
    return tf.keras.models.Model(inputs = input, outputs = output, name = f'DenseNet{layers}')

############## Nets ##############


@register_model
def DenseNet100k12(growth_rate = 12,
                   reduction = 0.5):
    """
    Parameters:
    ----------
    Returns
    -------
    """
    return DenseNet(reduction = reduction,
                    growth_rate = growth_rate,
                    layers=100)


@register_model
def DenseNet100k16(growth_rate = 16,
                   reduction = 0.5):
    """
    Parameters:
    ----------
    Returns
    -------
    """
    return DenseNet(reduction = reduction,
                    growth_rate = growth_rate,
                    layers=100)


@register_model
def DenseNet160k12(growth_rate = 12,
                   reduction = 0.5):
    """
    Parameters:
    ----------
    Returns
    -------
    """
    return DenseNet(reduction = reduction,
                    growth_rate = growth_rate,
                    layers=160)






