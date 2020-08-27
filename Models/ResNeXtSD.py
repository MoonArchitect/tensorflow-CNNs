import tensorflow as tf
from tensorflow.keras.backend import in_test_phase
from numpy import sum, max

""" 
    Implementation of ResNetXt with Stochastic Depth for CIFAR10/32x32

    From: Deep Networks with Stochastic Depth, https://arxiv.org/abs/1603.09382
    By: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
    And
    From: Aggregated Residual Transformations for Deep Neural Networks, https://arxiv.org/abs/1611.05431
    By: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
"""

class StochasticResUnit(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, current_layer, total_layers, p, k):
        super().__init__()
        self.survival_p = tf.Variable(1 - (current_layer**k/total_layers**k) * (1 - p), dtype=tf.dtypes.float32)

        self.pool = tf.keras.layers.AvgPool2D(strides, strides=strides, data_format='channels_first') if strides != (1, 1) else tf.keras.layers.Lambda(lambda a: a)
        self.filters = filters

        self.BN1 = tf.keras.layers.BatchNormalization(1)
        self.ReLU1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(filters // 2, (1, 1), (1, 1), padding='same', data_format='channels_first', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        self.BN2 = tf.keras.layers.BatchNormalization(1)
        self.ReLU2 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters // 2, kernel_size, strides, groups=32, padding='same', data_format='channels_first', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        self.BN3 = tf.keras.layers.BatchNormalization(1)
        self.ReLU3 = tf.keras.layers.ReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters, (1, 1), (1,1), padding='same', data_format='channels_first', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        
        self.add = tf.keras.layers.Add()
        pass

    def call(self, input):
        sc = self.pool(input)
        if input.shape[1] != self.filters:
            sc = tf.pad(sc, [[0,0], [(self.filters - input.shape[1]) // 2, (self.filters - input.shape[1]) // 2], [0,0], [0,0]])

        def ConvLayer(dIn, dSc):
            x = self.BN1(dIn)
            x = self.ReLU1(x)
            x = self.conv1(x)
            x = self.BN2(x)
            x = self.ReLU2(x)
            x = self.conv2(x)
            x = self.BN3(x)
            x = self.ReLU3(x)
            x = self.conv3(x)
            x = in_test_phase(tf.scalar_mul(self.survival_p, x), x)
            return self.add([x, dSc])
        
        rand = tf.random.uniform([1], 0, 1)
        x = tf.cond(
            rand <= in_test_phase(1.0, self.survival_p),
            lambda: ConvLayer(input, sc),
            lambda: sc
        )

        return x

def Stage(layers, filters, strides, layerIdx, total_layers, p, k):
    def f(input):
        x = StochasticResUnit(filters, (3,3), strides, layerIdx, total_layers, p, k)(input)
        for i in range(layers - 1):
            x = StochasticResUnit(filters, (3,3), (1, 1), layerIdx + i + 1, total_layers, p, k)(x)
        return x
    return f

def BottleneckResNet(conv_per_stage, input_shape, classes, filters = 16, filter_multiplier = [1, 1, 1, 1], p=0.5, k=1):
    N = conv_per_stage
    total_conv_layers = sum(conv_per_stage)
    current_conv_layer = 1

    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda a: tf.transpose(a, [0, 3, 1, 2]))(input)
    # Conv 1
    x = tf.keras.layers.Conv2D(filters * 2, (3, 3), (1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001), data_format='channels_first')(x)
    # Stage 1
    x = Stage(N[0], filters * 1 * 4, strides=(1,1), layerIdx=current_conv_layer, total_layers=total_conv_layers, p=p, k=k)(x)
    current_conv_layer += N[0]
    # Stage 2
    x = Stage(N[1], filters * 2 * 4, strides=(2,2), layerIdx=current_conv_layer, total_layers=total_conv_layers, p=p, k=k)(x)
    current_conv_layer += N[1]
    # Stage 3
    x = Stage(N[2], filters * 4 * 4, strides=(2,2), layerIdx=current_conv_layer, total_layers=total_conv_layers, p=p, k=k)(x)
    current_conv_layer += N[2]
    # Stage 4
    x = Stage(N[3], filters * 8 * 4, strides=(2,2), layerIdx=current_conv_layer, total_layers=total_conv_layers, p=p, k=k)(x)
    current_conv_layer += N[3]

    x = tf.keras.layers.BatchNormalization(1)(x) 
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input, outputs=output, name=f'{f"Wide{filter_multiplier}" if max(filter_multiplier) > 1 else ""}ResNeXtSD{sum(conv_per_stage) * 3 + 2}')

############## Nets ##############
def ResNeXtSD50(input_shape, classes, p = 0.5, k = 1):
    return BottleneckResNet([3, 4, 6, 3], input_shape, classes)

def ResNeXtSD101(input_shape, classes, p = 0.5, k = 1):
    return BottleneckResNet([3, 4, 23, 3], input_shape, classes, p = p, k = k)

def ResNeXtSD152(input_shape, classes, p = 0.35, k = 1):
    return BottleneckResNet(conv_per_stage = [3, 8, 33, 6], input_shape = input_shape, classes = classes, p = p, k = k)




