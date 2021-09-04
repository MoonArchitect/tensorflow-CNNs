# TODO import it
# import models
# import utils.LearningSchedules
from utils.registry import is_model, optimizers, lr_schedules, models



def create_model(name, **kwargs):
    name = name.lower()
    if is_model(name):
        return models[name](**kwargs)
    else:
        raise ValueError(f"Model '{name}' is not registered. Available: {list(models.keys())}")


def create_optimzer(name, **kwargs):
    name = name.lower()
    if name in optimizers:
        return optimizers[name](**kwargs)
    else:
        raise ValueError(f"Optimzer '{name}' is not registered. Available: {list(optimizers.keys())}")


def create_lr_schedule(name, **kwargs):
    name = name.lower()
    if name in lr_schedules:
        return lr_schedules[name](**kwargs)
    else:
        raise ValueError(f"lr_schedule '{name}' is not registered. Available: {list(lr_schedules.keys())}")
