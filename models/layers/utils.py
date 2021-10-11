import tensorflow as tf
import tensorflow.keras as nn

from .mish import Mish

__all__ = ['get_activation_layer', 'get_channels', 'linear_decay_fn', 'linear_decay', '_make_divisible']


def get_activation_layer(activation, **kwargs):
    """
    """
    assert activation is not None
    
    if isinstance(activation, str):
        activation = activation.lower()
        if activation == "relu":
            return nn.layers.ReLU(**kwargs)
        elif activation == "relu6":
            return nn.layers.ReLU(max_value=6.0, **kwargs)
        elif activation == "prelu":
            return nn.layers.PReLU(**kwargs)
        elif activation == "mish":
            return Mish(**kwargs)
        elif activation == "swish":
            return nn.layers.Activation(tf.nn.swish, **kwargs)
        elif activation == "hswish":
            # print("hswish is not implemented efficiently yet, using swish instead")
            return nn.layers.Activation(tf.nn.swish, **kwargs)
        elif activation == "sigmoid":
            return nn.layers.Activation('sigmoid', **kwargs)
        elif activation == "tanh":
            return nn.layers.Activation('tanh', **kwargs)
        else:
            raise NotImplementedError(f"{activation} is not implemented")
    else:
        assert isinstance(activation, nn.layers.Layer)
        return activation


def get_channels(x, data_format='channels_last'):
    if isinstance(x, tf.TensorShape):
        return x[3] if data_format == 'channels_last' else x[1]
    
    return x.shape[3] if data_format == 'channels_last' else x.shape[1]




def linear_decay_fn(start_pos_val,
                    end_pos_val,
                    name="Linear Decay"):
    """
    Returns function to generate values with linear decay corresponding to (start/end)_pos_val
    
    ----------
    Arguments:
    start_pos_val: tuple/list of 2 integers
        -
    end_pos_val: tuple/list of 2 integers
        -
    
    --------
    Returns:
    Python function
        -> fn(x), takes 1 argument, position:float
        -> Returns value:float
    """
    # Swap for convenience if
    assert start_pos_val[0] < end_pos_val[0], ""

    def fn(x):
        if not start_pos_val[0] <= x <= end_pos_val[0]:
            raise ValueError(f"{name} | Position {x} is not in the range of specified positions: {start_pos_val[0]}->{end_pos_val[1]}")

        return start_pos_val[1] + (x - start_pos_val[0]) / (end_pos_val[0] - start_pos_val[0]) * (end_pos_val[1] - start_pos_val[1])
    return fn


def linear_decay(x, start_pos_val, end_pos_val):
    return linear_decay_fn(start_pos_val, end_pos_val)(x)


def _make_divisible(value, divisor, msg_on_change = None):
    """
    Makes 'value' divisible by divisor
    If given, logs 'msg_on_change' if value is changed
    """
    initial = value

    extra = value % divisor
    value -= extra
    if extra >= divisor / 2:
        value += divisor

    if isinstance(divisor, int):
        value = int(value)

    if value == 0:
        print(f"\tWARNING (_make_divisible): value is rounded to 0, given {initial} w/ divisor {divisor}")
    
    if msg_on_change and initial != value:
        values_to_log = {}

        if "{initial}" in msg_on_change:
            values_to_log["initial"] = initial
        
        if "{final}" in msg_on_change:
            values_to_log["final"] = value
        
        print("\tWARNING (_make_divisible): " + msg_on_change.format(**values_to_log))

    return value
