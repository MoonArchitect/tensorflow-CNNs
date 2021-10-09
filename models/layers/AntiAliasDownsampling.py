import tensorflow as tf
import tensorflow.keras as nn


class AntiAliasDownsampling(nn.layers.Layer):
    """
    TResNet version of AntiAlias downsampling
    """
    def __init__(self, in_channels, data_format, **kwargs):
        super().__init__(**kwargs)
        self.data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
        
        filter = tf.constant([1., 2., 1.], dtype=self._compute_dtype)
        filter = (filter[:, None] * filter[None, :])
        filter = filter / tf.reduce_sum(filter)

        self.filter = tf.repeat(filter[:, :, None, None], [in_channels], axis=2)
        self.strides = [1, 2, 2, 1] if data_format == 'channels_last' else [1, 1, 2, 2]


    def call(self, inputs):
        return tf.nn.depthwise_conv2d(inputs, self.filter, self.strides, "SAME", data_format=self.data_format)
