import tensorflow as tf
from numpy import sum, max

""" 
    Implementation of ResNet V2 for CIFAR10/32x32

    From: Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027.
    By: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
"""

def BNRConv(filters, kernel_size, strides, kernel_regularizer=tf.keras.regularizers.l2(0.0001)):
    """BN + RELu + Convolution"""
    def f(input):
        x = tf.keras.layers.BatchNormalization()(input) 
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=False, kernel_regularizer=kernel_regularizer)(x) #, kernel_initializer='he_normal'
        return x

    return f

############## Residual Net ##############
def ResUnit(filters, kernel_size=(3,3), strides=(2,2)):
    """
    Simple Residiul Unit
    
    [ BN+RELu+Conv3x3 ]x2 + input
    """
    def f(input):
        sc = tf.keras.layers.AvgPool2D(strides, strides=strides)(input) if strides != (1, 1) else input
        # pad main dim, width/height if after avgpool dims != after conv
        if input.shape[3] != filters:
            sc = tf.pad(sc, [[0,0], [0,0], [0,0], [(filters - input.shape[3]) // 2, (filters - input.shape[3]) // 2]])
        
        x = BNRConv(filters, kernel_size, strides)(input)
        x = BNRConv(filters, kernel_size, (1, 1))(x) #ShakeDrop

        x = tf.keras.layers.Add()([x, sc])
        return x
    return f

def ResStage(layers, filters, kernel_size=(3,3), strides=(2,2)):
    def f(input):
        x = ResUnit(filters, kernel_size, strides)(input)
        for i in range(layers - 1):
            x = ResUnit(filters, kernel_size, strides=(1,1))(x)
        return x
    return f

def ResNet(conv_per_stage, input_shape, classes, filters = 16, filter_multiplier = [1, 1, 1, 1]):
    """"""
    N = conv_per_stage
    
    input = tf.keras.layers.Input(shape=input_shape)
    #Conv 1
    x = tf.keras.layers.Conv2D(filters, (3, 3), (1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    # Stage 1
    x = ResStage(N[0], filters * 1 * filter_multiplier[0], strides=(1,1))(x)
    # Stage 2
    x = ResStage(N[1], filters * 2 * filter_multiplier[1])(x)
    # Stage 3
    x = ResStage(N[2], filters * 4 * filter_multiplier[2])(x)
    # Stage 4
    x = ResStage(N[3], filters * 8 * filter_multiplier[3])(x)

    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input, outputs=output, name=f'{f"Wide{filter_multiplier}" if max(filter_multiplier) > 1 else ""}ResNet{sum(conv_per_stage) * 2 + 2}')

############## Bottleneck Residual Net ##############
def BottleneckResUnit(filters, kernel_size=(3,3), strides=(2,2)):
    """
    Simple Residiul Unit
    
    [ BN+RELu+Conv3x3 ]x2 + input
    """
    def f(input):
        sc = tf.keras.layers.AvgPool2D(strides, strides=strides)(input) if strides != (1, 1) else input
        if input.shape[3] != filters:
            sc = tf.pad(sc, [[0,0], [0,0], [0,0], [(filters - input.shape[3]) // 2, (filters - input.shape[3]) // 2]])

        x = BNRConv(filters // 4, (1, 1), (1, 1))(input)
        x = BNRConv(filters // 4, kernel_size, strides)(x)
        x = BNRConv(filters, (1, 1), (1, 1))(x)
        #tf.add()
        x = tf.keras.layers.Add()([x, sc])
        return x
    return f

def BottleneckResStage(layers, filters, kernel_size=(3,3), strides=(2,2)):
    def f(input):
        x = BottleneckResUnit(filters, kernel_size, strides)(input)
        for i in range(layers - 1):
            x = BottleneckResUnit(filters, kernel_size, (1, 1))(x)
        return x
    return f

def BottleneckResNet(conv_per_stage, input_shape, classes, filters = 16, filter_multiplier = [1, 1, 1, 1]):
    """"""
    N = conv_per_stage
    
    input = tf.keras.layers.Input(shape=input_shape)
    # Conv 1
    x = tf.keras.layers.Conv2D(filters, (3, 3), (1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    # Stage 1
    x = BottleneckResStage(N[0], filters * 4 * 1 * filter_multiplier[0], strides=(1,1))(x)
    # Stage 2
    x = BottleneckResStage(N[1], filters * 4 * 2 * filter_multiplier[1])(x)
    # Stage 3
    x = BottleneckResStage(N[2], filters * 4 * 4 * filter_multiplier[2])(x)
    # Stage 4
    x = BottleneckResStage(N[3], filters * 4 * 8 * filter_multiplier[3])(x)

    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(classes)(x)

    return tf.keras.models.Model(inputs=input, outputs=output, name=f'{f"Wide{filter_multiplier}" if max(filter_multiplier) > 1 else ""}ResNet{sum(conv_per_stage) * 3 + 2}b')

############## Nets ##############
def ResNet18(input_shape, classes):
    return ResNet([2, 2, 2, 2], input_shape, classes)

def ResNet34(input_shape, classes):
    return ResNet([3, 4, 6, 3], input_shape, classes)

def ResNet50(input_shape, classes):
    return BottleneckResNet([3, 4, 6, 3], input_shape, classes)

def ResNet101(input_shape, classes):
    return BottleneckResNet([3, 4, 23, 3], input_shape, classes)

def ResNet152(input_shape, classes):
    return BottleneckResNet([3, 8, 36, 3], input_shape, classes)

