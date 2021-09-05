import sys
import tensorflow as tf


models = {}

lr_schedules = {}

optimizers = {
    "sgd": tf.keras.optimizers.SGD
}


def register_model(fn):
    name = fn.__name__
    module = sys.modules[fn.__module__]

    if hasattr(module, '__all__'):
        module.__all__.append(name)
    else:
        module.__all__ = [ name ]
    
    name = name.lower()
    if name in models:
        raise ValueError(f"Model '{name}' is already registered in '{module}'")
    models[name] = fn

    return fn


def get_all_models():
    return list(models.keys())


def is_model(name):
    return name in models




def register_schedule(fn):
    name = fn.__name__.lower()
    module = sys.modules[fn.__module__]
    
    if name in lr_schedules:
        raise ValueError(f"lr_schedule '{name}' is already registered in '{module}'")
    lr_schedules[name] = fn

    return fn

