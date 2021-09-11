import os
from time import perf_counter

import tensorflow as tf
import tensorflow.keras as nn

from .registry import get_all_models
from .creator import create_model


__all__ = ['initialize_and_run_all_models', 'perf_benchmark']


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


# TODO change benchmark length based on model
# TODO custom callback that stops training when performance is stable
def perf_benchmark(models = None,
                   batch_size = 256,
                   fp16 = False,
                   xla = False,
                   verbose = '3'):
    """
    Benchmarks model training and inference performance
    
    Parameters:
    models: str
        models' name
        if None -> benchmarks all registered models
    """
    
    assert isinstance(verbose, str)
    assert isinstance(fp16, bool)
    assert isinstance(xla, bool)

    if models is None:
        models = get_all_models()

    if not isinstance(models, list):
        models = [ models ]

    assert all([ isinstance(model, str) for model in models ])
    
    print("Clearing session")
    nn.backend.clear_session()

    if verbose:
        print(f"Verbose set to {verbose}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = verbose

    if xla:
        print("XLA is ON")
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=3"
    
    if fp16:
        print("FP16 is ON")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def reshape_data(x):
        x = tf.random.uniform((32, 32, 3), 0., 1., tf.float32, 42)
        y = tf.one_hot(
            tf.random.uniform(( ), 0, 10, tf.int32, 42),
            depth=10,
            on_value=1,
            off_value=0,
            dtype=tf.int32
        )
        return (x, y)

    test_dataset = tf.data.Dataset.random(42) \
                                  .map( reshape_data, num_parallel_calls=tf.data.AUTOTUNE ) \
                                  .batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
                                  .prefetch(12)

    # Tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir="logs/testing",
    #     profile_batch='15, 30'
    # )

    print()

    for model_name in models:
        print(f" - Testing {model_name}:")
        model: nn.Model
        model = create_model(model_name)
        model.compile(
            optimizer=nn.optimizers.SGD(),
            loss=nn.losses.CategoricalCrossentropy(from_logits=True)
        )

        print("\tWarming up the model")
        model.fit(
            x=test_dataset,
            batch_size=batch_size,
            steps_per_epoch=50,
            # callbacks=[Tensorboard_callback]
        )

        start_time = perf_counter()
        model.fit(
            x=test_dataset,
            batch_size=batch_size,
            steps_per_epoch=200,
        )
        print(f"\tTraining: {batch_size * 200 / (perf_counter() - start_time):.2f} imgs/sec")  # TODO display std

        start_time = perf_counter()
        model.evaluate(
            x=test_dataset,
            batch_size=batch_size,
            steps=200
        )
        print(f"\tInference: {batch_size * 200 / (perf_counter() - start_time):.2f} imgs/sec")
        
        nn.backend.clear_session()
        print()
    
