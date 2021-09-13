import os
from datetime import datetime

import tensorflow as tf

import models
from utils import LearningSchedules
from data import read_dataset, prepare_cifar10


tf.keras.mixed_precision.set_global_policy('mixed_float16')     # enable mixed precision
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=3"              # enable XLA


epochs = 50
batch_size = 512
max_lr = 0.4


print("Dataset prep")
train_ds, val_ds = read_dataset()
train_ds, val_ds = prepare_cifar10(
    train_ds, val_ds,
    batch_size=batch_size,
    adv_augment="cutmix"
)


print("Model compile")
model = models.ResNetV2.ResNet34()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(True),
    optimizer=tf.keras.optimizers.SGD(momentum=0.9, nesterov=True),
    metrics=['accuracy']
)

model.summary()


model_filename = model.name + datetime.now().strftime("@%d^%H^%M")
log_dir = os.path.join("logs", model_filename)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, profile_batch = '100, 110')
lr_callback = LearningSchedules.CosineDecay(epochs, max_lr, 0.003, 5)


print("Fitting")
model.fit(
    train_ds,
    epochs = epochs,
    steps_per_epoch = 50000 // batch_size,
    validation_data = val_ds,

    callbacks = [
        lr_callback,
        tensorboard_callback,
    ]
)

# Epoch 50/50
# 97/97 [==============================] - 6s 66ms/step - loss: 0.7921 - accuracy: 0.9002 - val_loss: 0.3692 - val_accuracy: 0.9235
