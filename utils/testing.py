import os
from time import perf_counter

import numpy as np
import tensorflow as tf
import tensorflow.keras as nn

from .registry import get_all_models
from .creator import create_model


__all__ = ['initialize_and_run_all_models', 'performance_benachmark']


def initialize_and_run_all_models():
    """
    Initializes, compiles and runs all models on random data
    """
    inputs = tf.random.stateless_uniform((16, 32, 32, 3), seed=[1, 2])
    print("\n\nInitializing and running all models\n")

    for model_name in get_all_models():
        print(f"Testing {model_name}")
        model: nn.Model
        model = create_model(model_name)
        model.compile()
        model(inputs)
        
        tf.keras.backend.clear_session()
    
    # just to be sure
    tf.keras.backend.clear_session()


# Measure Performance

class PerformanceMeasurer(nn.callbacks.Callback):
    """
    """
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None
        self.train_batch = 0
        self.test_batch = 0
        self.skip = 10

    def count_params(self, weights):
        """
        Method is taken from tensorflow->keras->utils->layer_utils->count_params
        
        Count the total number of scalars composing the weights.
        
        Args:
            weights: An iterable containing the weights on which to compute params

        Returns:
            The total number of scalars composing the weights
        """
        unique_weights = {id(w): w for w in weights}.values()
        weight_shapes = [w.shape.as_list() for w in unique_weights]
        standardized_weight_shapes = [
            [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
        ]
        return int(sum(np.prod(p) for p in standardized_weight_shapes))

    def log_params_count(self):
        """
        Snippet taken from tensorflow->keras->utils->layer_utils->print_summary method

        Logs number of total, trainable and non-trainable parameters in the model
        """
        if hasattr(self.model, '_collected_trainable_weights'):
            trainable_count = self.count_params(self.model._collected_trainable_weights)
        else:
            trainable_count = self.count_params(self.model.trainable_weights)

        non_trainable_count = self.count_params(self.model.non_trainable_weights)

        print('\tTotal params: {:,}'.format(trainable_count + non_trainable_count))
        print('\tTrainable params: {:,}'.format(trainable_count))
        print('\tNon-trainable params: {:,}'.format(non_trainable_count))

    def on_train_begin(self, logs=None):
        print("\t2/3 Measuring training performance . . .")

    def on_train_batch_begin(self, batch, logs=None):
        self.train_batch = batch
        if batch == self.skip:
            self.train_start = perf_counter()

        if batch > self.skip and batch % 5 == 0 and perf_counter() - self.train_start > 30:
            self.model.stop_training = True

    def on_test_begin(self, logs=None):
        print("\t3/3 Measuring inference performance . . .")
        self.train_end = perf_counter()

    def on_test_batch_begin(self, batch, logs=None):
        self.test_batch = batch
        if batch == self.skip * 2:
            self.test_start = perf_counter()
    
    def on_train_end(self, logs=None):
        self.test_end = perf_counter()

        train_perf = self.batch_size * (self.train_batch - self.skip) / (self.train_end - self.train_start)
        test_perf = self.batch_size * (self.test_batch - self.skip * 2) / (self.test_end - self.test_start)

        print(f"\tTraining: {train_perf:.2f} imgs/sec")  # TODO display std
        print(f"\tInference: {test_perf:.2f} imgs/sec")
        self.log_params_count()


def parse_code(code):
    try:
        if "(" in code:
            name, params = code.split("(", 1)
            name = name.split(".")[-1]
            kwargs = eval("dict(" + params)
        else:
            name = code.split(".")[-1]
            kwargs = {}

    except Exception as e:
        print(f"\nERROR     -     {e.__class__} exception occured during an attempt to parse {code}")
        print("Expected syntax: \n")
        raise e

    return name, kwargs


def performance_benachmark(models=None,
                           batch_size = 256,
                           fp16 = False,
                           xla = False,
                           verbose = '2'):
    """
    Benchmarks model training and inference performance
    
    Parameters:
    models: str
        models' name
        if None -> benchmarks all registered models
    """
    
    assert isinstance(verbose, str), "verbose should be one of ['0', '1', '2', '3']"
    assert isinstance(fp16, bool), "fp16 should be either True or False"
    assert isinstance(xla, bool), "xla should be either True or False"

    if models is None:
        models = get_all_models()
    elif not isinstance(models, list):
        models = [ models ]

    assert all([ isinstance(model, str) for model in models ]), f"Each entry should be a registered model_name of type str, recieved {models}"
    
    print("Clearing session . . .")
    nn.backend.clear_session()

    if verbose:
        print(f"Verbose: {verbose}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = verbose
    
    print(f"Batch-size: {batch_size}")
    
    if xla:
        print("XLA: ON")
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=3"
    
    if fp16:
        print("FP16: ON")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def reshape_data(x):
        x = tf.random.uniform((32, 32, 3), 0., 1., tf.float32, 42)
        y = tf.one_hot(indices=tf.random.uniform(( ), 0, 10, tf.int32, 42),
                       depth=10,
                       on_value=1,
                       off_value=0,
                       dtype=tf.int32)

        return (x, y)

    print("Building dataset . . .\n")
    test_dataset = tf.data.Dataset.random(42) \
                                  .map( reshape_data, num_parallel_calls=tf.data.AUTOTUNE ) \
                                  .batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
                                  .prefetch(12)

    for model_name in models:
        print(f" - {model_name}:")
        
        model_name, model_kwargs = parse_code(model_name)
        model = create_model(model_name, **model_kwargs)
        
        model.compile(optimizer=nn.optimizers.SGD(),
                      loss=nn.losses.CategoricalCrossentropy(from_logits=True))

        try:
            print("\t1/3 Warming up the model . . .")
            model.fit(
                x=test_dataset,
                batch_size=batch_size,
                steps_per_epoch=20,
                verbose=0
            )

            model.fit(
                x=test_dataset,
                batch_size=batch_size,
                steps_per_epoch=150,
                validation_data=test_dataset,
                validation_steps=200,
                verbose=0,
                callbacks=[
                    PerformanceMeasurer(batch_size=batch_size)
                ]
            )
        except Exception as e:
            print(e)
        
        nn.backend.clear_session()
        print()
    
