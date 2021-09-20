import tensorflow as tf
import tensorflow.keras as nn

from .layers import PreActConv
from utils.registry import register_model



"""
    Implementation of DenseNet for CIFAR10/32x32

    From: Densely Connected Convolutional Networks, https://arxiv.org/abs/1608.06993.
    By: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
"""


class DenseNetUnit(nn.layers.Layer):
    """
    Basic DenseNet unit
    """
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = PreActConv(filters * 4, (1, 1), (1, 1))
        self.conv2 = PreActConv(filters, (3, 3), (1, 1))

    def call(self, inputs):
        x = self.conv1(inputs)
        outputs = self.conv2(x)
        return outputs


class DenseNetBlock(nn.layers.Layer):
    """
    Dense Block consisting of 'n_units' basic units
    """
    def __init__(self, n_units, filters, **kwargs):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.units = []
        for _ in range(self.n_units):
            self.units.append( DenseNetUnit(filters) )

    def call(self, inputs):
        for i in range(self.n_units):
            x = self.units[i](inputs)
            inputs = tf.concat([x, inputs], axis=3)
        
        return inputs


# TODO unify arguments
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

    input = nn.Input(input_shape)

    x = nn.layers.Conv2D(filters=growth_rate * 2,
                         kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                         )(input)

    x = DenseNetBlock(N, growth_rate)(x)
    x = PreActConv(int(x.shape[3] * reduction), (1, 1), (1, 1))(x)
    x = nn.layers.AvgPool2D((2, 2), (2, 2))(x)

    x = DenseNetBlock(N, growth_rate)(x)
    x = PreActConv(int(x.shape[3] * reduction), (1, 1), (1, 1))(x)
    x = nn.layers.AvgPool2D((2, 2), (2, 2))(x)

    x = DenseNetBlock(N, growth_rate)(x)
    x = nn.layers.BatchNormalization()(x)
    x = nn.layers.ReLU()(x)

    x = nn.layers.GlobalAveragePooling2D()(x)

    output = nn.layers.Dense(classes)(x)

    return nn.models.Model(inputs=input,
                           outputs=output,
                           name=f'DenseNet{layers}')


############## Predefined Models ##############
@register_model
def DenseNet52k12(growth_rate = 12,
                  reduction = 0.5):
    """
    Parameters:
    ----------
    Returns
    -------
    """
    return DenseNet(reduction = reduction,
                    growth_rate = growth_rate,
                    layers=52)


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
