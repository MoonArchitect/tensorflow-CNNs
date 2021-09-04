import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
from utils.registry import register_schedule

""" Learning rate schedules """


@register_schedule
def StepDecaySchedule(schedule, verbose=0):
    """
    Parameters:
    ----------
    schedule: array of N (epoch,value) pairs, in increasing order of epoch
    Example:
    ----------
    schedule=[(0, 0.1), (150, 0.01), (225, 0.001)]
    """
    def f(epoch):
        for t, lr in reversed(schedule):
            if epoch >= t:
                return lr
        # If value has not been returned yet
        raise ValueError(f"No suitable lr is found from {schedule} for epoch:{epoch}")
    return LearningRateScheduler(f, verbose)


@register_schedule
def LinearDecaySchedule():
    # interpolate step schedule
    raise NotImplementedError("")


@register_schedule
def HTD(L, U, totalEpochs, lrmax, lrmin, warmup=0, verbose=0):
    """
    Hyperbolic-Tangent Decay schedule. From: Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification, https://arxiv.org/abs/1806.01593
    
    Parameters:
    ----------
    L: Lower Bound
    U: Upper Bound
    totalEpochs: Number of epochs to train
    lrmax: Maximum lr (at the start of the training)
    lrmin: Minimum lr (at the end of the training)
    """
    def f(epoch):
        if epoch >= warmup:
            return lrmin + (lrmax - lrmin) / 2 * (1 - np.math.tanh(L + (U - L) * epoch / totalEpochs))
        else:
            return (lrmax - lrmin) / warmup * (epoch + 0.5)
    return LearningRateScheduler(f, verbose)


@register_schedule
def CosineDecay(totalEpochs, lrmax, lrmin, warmup=0, verbose=0):
    """
    Cosine Decay schedule. From: Stochastic Gradient Descent with Warm Restarts, https://arxiv.org/abs/1608.03983
    
    Arguments:
    ----------
    totalEpochs: Number of epochs to train
    lrmax: Maximum lr (at the start of the training)
    lrmin: Minimum lr (at the end of the training)
    """
    def f(epoch):
        if epoch >= warmup:
            return lrmin + (lrmax - lrmin) / 2 * (1 + np.math.cos(epoch / totalEpochs * np.math.pi))
        else:
            return (lrmax - lrmin) / warmup * (epoch + 0.5)
    return LearningRateScheduler(f, verbose)

