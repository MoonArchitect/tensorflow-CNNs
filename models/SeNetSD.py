import tensorflow as tf
from tensorflow.keras.backend import in_test_phase
from numpy import sum, max
from utils.registry import register_model

"""
    Implementation of ResNetSE for CIFAR10/32x32

    From: , .
    By:
"""


class SEBlock(tf.keras.layers.Layer):
    """
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or nn.Layer, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Layer, default 'sigmoid'
        Activation function after the last convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 # round_mid=False,
                 use_conv=False,
                 mid_activation="relu",
                 out_activation="sigmoid",
                 data_format="channels_first",
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.use_conv = use_conv
        self.data_format = data_format
        mid_channels = channels // reduction  # if not round_mid else round_channels(float(channels) / reduction)

        self.pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.fc1 = tf.keras.layers.Dense(
            units=mid_channels,
            input_dim=channels,
            name="fc1")
        self.activ = tf.keras.layers.ReLU()  # get_activation_layer(mid_activation, name="activ")
        self.fc2 = tf.keras.layers.Dense(
            units=channels,
            input_dim=mid_channels,
            name="fc2")
        self.sigmoid = tf.keras.layers.Activation(tf.keras.activations.sigmoid)  # get_activation_layer(out_activation, name="sigmoid")

    def call(self, x, training=None):
        w = self.pool(x)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        axis = -1
        w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        x = x * w
        return x


class SqueezeExcitationUnit(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, current_layer, total_layers, p, k, r = 8):
        super().__init__()

        self.survival_p = tf.Variable(1 - (current_layer ** k / total_layers ** k) * (1 - p), dtype=tf.dtypes.float32)

        self.pool = tf.keras.layers.AvgPool2D(strides, strides=strides, data_format='channels_first') if strides != (1, 1) else tf.keras.layers.Lambda(lambda a: a)
        self.filters = filters

        self.BN1 = tf.keras.layers.BatchNormalization(1)
        self.ReLU1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(filters // 4, (1, 1), (1, 1), padding='same', data_format='channels_first', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        self.BN2 = tf.keras.layers.BatchNormalization(1)
        self.ReLU2 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters // 4, kernel_size, strides, padding='same', data_format='channels_first', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        self.BN3 = tf.keras.layers.BatchNormalization(1)
        self.ReLU3 = tf.keras.layers.ReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters, (1, 1), (1, 1), padding='same', data_format='channels_first', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        
        self.SE = SEBlock(filters)
        # self.SE_GlobalPool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')
        # self.SE_Dense1 = tf.keras.layers.Dense(filters // r, use_bias=False)
        # self.SE_ReLU = tf.keras.layers.ReLU()
        # self.SE_Dense2 = tf.keras.layers.Dense(filters, use_bias=False)
        # self.SE_Sigmoid = tf.keras.layers.Activation(tf.keras.activations.sigmoid)

        self.add = tf.keras.layers.Add()

    def call(self, input):
        sc = self.pool(input)
        if input.shape[1] != self.filters:
            sc = tf.pad(sc, [[0, 0], [(self.filters - input.shape[1]) // 2, (self.filters - input.shape[1]) // 2], [0, 0], [0, 0]])

        def SEstage():
            x = self.BN1(input)
            x = self.ReLU1(x)
            x = self.conv1(x)
            x = self.BN2(x)
            x = self.ReLU2(x)
            x = self.conv2(x)
            x = self.BN3(x)
            x = self.ReLU3(x)
            x = self.conv3(x)
            x = self.SE(x)

            x = in_test_phase(tf.scalar_mul(self.survival_p, x), x)
            return self.add([x, sc])

        rand = tf.random.uniform([1], 0, 1)
        x = tf.cond(
            rand <= in_test_phase(1.0, self.survival_p),
            lambda: SEstage(),
            lambda: sc
        )

        return x


def SeNetStage(layers, filters, kernel_size, strides, layerIdx, total_layers, p, k):
    def f(input):
        x = SqueezeExcitationUnit(filters, kernel_size, strides, layerIdx, total_layers, p, k)(input)
        for i in range(1, layers):
            x = SqueezeExcitationUnit(filters, kernel_size, (1, 1), layerIdx + i, total_layers, p, k)(x)
        return x
    return f


def SeNetSD(conv_per_stage, input_shape, classes, filters = 16, filter_multiplier = [1, 1, 1, 1], p = 0.5, k = 1):
    N = conv_per_stage
    total_conv_layers = sum(conv_per_stage)
    current_conv_layer = 1

    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda a: tf.transpose(a, [0, 3, 1, 2]))(input)

    # Conv 1
    x = tf.keras.layers.Conv2D(filters, (3, 3), (1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001), data_format='channels_first')(x)

    # Stage 1
    x = SeNetStage(N[0], filters * 4 * 1 * filter_multiplier[0], (3, 3), (1, 1), current_conv_layer, total_conv_layers, p, k)(x)
    current_conv_layer += N[0]
    # Stage 2
    x = SeNetStage(N[1], filters * 4 * 2 * filter_multiplier[1], (3, 3), (2, 2), current_conv_layer, total_conv_layers, p, k)(x)
    current_conv_layer += N[1]
    # Stage 3
    x = SeNetStage(N[2], filters * 4 * 4 * filter_multiplier[2], (3, 3), (2, 2), current_conv_layer, total_conv_layers, p, k)(x)
    current_conv_layer += N[2]
    # Stage 4
    x = SeNetStage(N[3], filters * 4 * 8 * filter_multiplier[3], (3, 3), (2, 2), current_conv_layer, total_conv_layers, p, k)(x)
    current_conv_layer += N[3]

    x = tf.keras.layers.BatchNormalization(1)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input, outputs=output, name=f'{f"Wide{filter_multiplier}" if max(filter_multiplier) > 1 else ""}SeNet{sum(conv_per_stage) * 3 + 2}SD_p{p}')


############## Nets ##############
@register_model
def SeNetSD50(input_shape, classes, p = 0.7, k = 1):
    return SeNetSD(conv_per_stage = [3, 4, 6, 3], input_shape = input_shape, classes = classes, p = p, k = k)


@register_model
def SeNetSD101(input_shape, classes, p = 0.5, k = 0.9):
    return SeNetSD(conv_per_stage = [3, 4, 23, 3], input_shape = input_shape, classes = classes, p = p, k = k)


@register_model
def SeNetSD152(input_shape, classes, p = 0.35, k = 0.75):
    return SeNetSD(conv_per_stage = [3, 8, 36, 3], input_shape = input_shape, classes = classes, p = p, k = k)
