from tensorflow.keras.callbacks import ModelCheckpoint


__all__ = ['BestModelCheckpoint']


class BestModelCheckpoint(ModelCheckpoint):
    """
    BestModelCheckpoint is not implemented yet
    """
    def __init__(self, delay, filepath, save_weights_only = True, **kwargs):
        raise NotImplementedError("BestModelCheckpoint is not implemented yet")
        super().__init__(filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=save_weights_only, mode='max', save_freq='epoch', **kwargs)
        self.delay = delay
