import os
import tensorflow as tf

__all__ = ["readDatasets", "prepareDatasets", "display_dataset"]


def readDatasets(path = "tfrecords/cifar10"):
    train_dataset = tf.data.TFRecordDataset([os.path.join(path, "train.tfrecord")])
    val_dataset = tf.data.TFRecordDataset([os.path.join(path, "val.tfrecord")])

    feature_map = {
        "imgs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_data(example):
        example = tf.io.parse_single_example(example, feature_map)

        imgs = tf.io.parse_tensor(example['imgs'], tf.uint8)
        labels = tf.io.parse_tensor(example['labels'], tf.uint8)

        imgs.set_shape([None, 32, 32, 3])  # tf.shape(imgs)
        labels.set_shape([None, 1])

        return tf.data.Dataset.from_tensor_slices((imgs, labels))

    train_dataset = train_dataset.interleave(parse_data)
    val_dataset = val_dataset.interleave(parse_data)

    return train_dataset, val_dataset



def prepareDatasets(train_dataset,
                    val_dataset,
                    batch_size,
                    adv_augment = None):
    # normalize
    mean = [125.306918046875, 122.950394140625, 113.86538318359375]
    std = [62.99321927813685, 62.088707640014405, 66.70489964063101]
    train_dataset = train_dataset.map(
        lambda x, y: (( tf.cast(x, tf.float32) - mean) / std, tf.one_hot(y[0], 10, 1., 0.)),
        num_parallel_calls = 8, deterministic = False).cache()

    val_dataset = val_dataset.map(
        lambda x, y: (( tf.cast(x, tf.float32) - mean) / std, tf.one_hot(y[0], 10, 1., 0.)),
        num_parallel_calls = 8, deterministic = False).cache()


    train_dataset = train_dataset.repeat().shuffle(1024, reshuffle_each_iteration=True).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    """ flip left-right, random 4px shift """
    def augment(x, y):
        img_shape = x.shape[1:]
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [ [0, 0], [4, 4], [4, 4], [0, 0] ], mode='SYMMETRIC', )
        x = tf.map_fn(lambda i: tf.image.random_crop(i, img_shape), x)
        return x, y
    
    train_dataset = train_dataset.map(augment, num_parallel_calls = 8, deterministic = False)
    
    # adv_augment
    if adv_augment:
        adv_augment.lower()

        # adv_fn = lambda x, y: (x, y)  # not necessary
        img_size = 32

        if adv_augment == "cutmix":
            adv_fn = _cutmix_fn(img_size, batch_size)
        elif adv_augment == "cutout":
            adv_fn = _cutout_fn(img_size, batch_size, 16)
        elif adv_augment == "mixup":
            adv_fn = _mixup_fn(batch_size, 0.2)
        else:
            raise ValueError(f"Value of advanced_augment has to be one of ['cutout', 'cutmix', 'mixup', 'HardPatchUp', 'SoftPatchUp'], recieved {adv_augment}")

        train_dataset = train_dataset.map(adv_fn, num_parallel_calls = 8)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset



def _cutmix_fn(imgSize, batch_size):
    def map_fn(x, y):
        def make_mask(id):
            lam = tf.random.uniform([1], 0, 1)
            pos = tf.random.uniform([2], 0, imgSize + 1, tf.int32)
            dim = tf.math.minimum([imgSize, imgSize] - pos, tf.cast([imgSize, imgSize] * tf.math.sqrt(1 - lam[0]), dtype=tf.int32))

            mask = tf.pad(tf.zeros(dim), [[pos[0], imgSize - pos[0] - dim[0]], [pos[1], imgSize - pos[1] - dim[1]]], mode='CONSTANT', constant_values=1)
            mask = tf.expand_dims(mask, -1)
            mask = tf.ensure_shape(mask, [32, 32, 1])

            return mask, tf.expand_dims(tf.cast(1 - dim[0] * dim[1] / 1024, dtype=tf.float32), -1)
        
        masks = tf.map_fn(
            lambda i: make_mask(i),
            tf.range(0, batch_size, dtype=tf.int32),
            fn_output_signature=(tf.TensorSpec([32, 32, 1]), tf.TensorSpec([1]))
        )
        
        pairedX = tf.random.shuffle(x, seed=14723)
        pairedY = tf.random.shuffle(y, seed=14723)
        
        x = x * masks[0] + (1 - masks[0]) * pairedX
        y = y * masks[1] + (1 - masks[1]) * pairedY

        return x, y
    return map_fn



def _mixup_fn(batch_size, alpha=1):
    def map_fn(x, y):
        lam = tf.compat.v1.distributions.Beta(alpha, alpha).sample((batch_size, 1))
        pairedY = tf.random.shuffle(y, 15036)
        y = y * lam + (1 - lam) * pairedY
        lam = tf.expand_dims(tf.expand_dims(lam, [-1]), [-1])
        pairedX = tf.random.shuffle(x, 15036)
        x = x * lam + (1 - lam) * pairedX

        return x, y
    return map_fn



def _cutout_fn(imgSize, batch_size, cutSize):
    def map_fn(x, y):
        cutout = tf.zeros([batch_size, cutSize, cutSize])
        
        randPos = tf.cast(tf.random.uniform([batch_size, 2], 0, imgSize + 1, tf.int32), dtype=tf.float32)
        cutout = tf.pad(
            cutout,
            [ [0, 0], [imgSize - cutSize // 2, imgSize - cutSize // 2], [imgSize - cutSize // 2, imgSize - cutSize // 2] ],
            constant_values=1, mode='CONSTANT'
        )

        randPos = tf.reshape(
            tf.stack(
                [randPos, randPos + tf.constant([[imgSize, imgSize]] * batch_size, dtype='float32')], axis=1
            ), (batch_size, 4)
        ) / ((imgSize - cutSize // 2) * 2 + cutSize)

        cutout = tf.expand_dims(cutout, [-1])  # data_format=='channels_last'
        cutout = tf.image.crop_and_resize(cutout, randPos, tf.range(batch_size), [imgSize, imgSize], method='nearest')
        x = tf.multiply(x, cutout)

        return x, y
        
    return map_fn


def display_dataset(ds):
    from matplotlib import pyplot as plt
    import numpy as np

    imgs, labels = next(ds.as_numpy_iterator())
    imgs = np.array(imgs)
    imgs *= [62.99321927813685, 62.088707640014405, 66.70489964063101]
    imgs += [125.306918046875, 122.950394140625, 113.86538318359375]
    imgs = np.clip(imgs / 255, 0.0, 1.0)
    
    f = plt.figure(figsize=(8, 8))
    f.subplots_adjust(0.05, 0, 0.95, 0.9, 0.1, 0.1)
    f.tight_layout()
    n = imgs.shape[0]

    for i in range(n):
        ax = f.add_subplot(n // 8 + 1, 8, i + 1)
        ax.axis('off')
        plt.imshow(imgs[i])

    plt.show(block=True)

    
