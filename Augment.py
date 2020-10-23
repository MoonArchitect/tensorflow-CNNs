import tensorflow as tf

def from_image_slices(images,
                      labels,
                      batch_size,
                      advanced_augment=None,
                      flip_up_down=False,
                      flip_left_right=False,
                      #random_hue_range=None,
                      random_shift_delta=None,
                      random_contrast_delta=None,
                      random_brightness_delta=None,
                      #random_saturation_range=None
                      num_parallel_calls=12,
                      ):
    """
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)
                    ).repeat(
                    ).shuffle(512, reshuffle_each_iteration=True
                    ).batch(batch_size)
    
    if advanced_augment:
        if advanced_augment == 'Cutout':
            advanced_fn = Cutout(imgSize = images[0].shape[0],
                                 cutSize = 16,
                                 batch_size = batch_size,
                                 returnDataset=True)
        elif advanced_augment == 'Cutmix':
            advanced_fn = Cutmix(imgSize = images[0].shape[0], 
                                 batch_size = batch_size, 
                                 returnDataset=True)
        elif advanced_augment == 'Mixup':
            advanced_fn = Mixup(alpha=0.2)
        elif advanced_augment == 'HardPatchUp':
            #advanced_fn = HardPatchUp()
            raise NotImplementedError()
        elif advanced_augment == 'SoftPatchUp':
            #advanced_fn = SoftPatchUp()
            raise NotImplementedError()
        else:
            raise ValueError(f"Value of advanced_augment has to be one of ['Cutout', 'Cutmix', 'Mixup', 'HardPatchUp', 'SoftPatchUp'], recieved {advanced_augment}")

        dataset = dataset.interleave(advanced_fn, num_parallel_calls=num_parallel_calls)
    
    if flip_up_down or flip_left_right or random_shift_delta or random_contrast_delta or random_brightness_delta:
        augment_fn = Augment(batch_size = batch_size,
                             flip_up_down = flip_up_down,
                             flip_left_right=flip_left_right,
                             random_shift_delta=random_shift_delta,
                             random_contrast_delta=random_contrast_delta,
                             random_brightness_delta=random_brightness_delta,
                             returnDataset=True)
        dataset = dataset.interleave(augment_fn, num_parallel_calls=num_parallel_calls)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

def Augment(batch_size,
            flip_up_down=None, 
            flip_left_right=None,
            #random_hue_range=None,
            random_shift_delta=None,
            random_contrast_delta=None,
            random_brightness_delta=None,
            #random_saturation_range=None
            returnDataset=True
            ):
    """
    """
    def map_fn(x, y):
        img_shape = x.shape[1:]
        if random_shift_delta:
            x = tf.pad(x, [[0,0], random_shift_delta, random_shift_delta, [0,0]], mode='SYMMETRIC')
            x = tf.map_fn(lambda i: tf.image.random_crop(i, img_shape), x)

        #if random_brightness_delta:
        #    x = tf.map_fn(lambda x: tf.image.random_brightness(x, random_brightness_delta), x)

        #if random_contrast_delta:
        #    x = tf.map_fn(lambda x: tf.image.random_contrast(x, random_contrast_delta[0], random_contrast_delta[1]), x)
        
        if flip_left_right:
            x = tf.image.random_flip_left_right(x)

        if flip_up_down:
            x = tf.image.random_flip_up_down(x)
        
        if returnDataset:
            return tf.data.Dataset.from_tensors((x, y))
        else:
            return x, y
    return map_fn

# To fix:
#   profiler doesn't works
def Cutout(imgSize, 
           cutSize,
           batch_size,
           returnDataset=True,
           data_format='channels_last'):
    """
    """
    def map_fn(x, y):
        cutout = tf.zeros([batch_size, cutSize, cutSize])
        
        randPos = tf.cast(tf.random.uniform([batch_size, 2], 0, imgSize+1, dtype=tf.int32), dtype=tf.float32)
        cutout = tf.pad(
            cutout, 
            [[0,0], [imgSize-cutSize//2,imgSize-cutSize//2], [imgSize-cutSize//2,imgSize-cutSize//2]], 
            constant_values=1, mode='CONSTANT'
        )
        randPos = tf.reshape(
            tf.stack(
                [randPos, randPos + tf.constant([[imgSize, imgSize]]*batch_size, dtype='float32')], axis=1
            ), (batch_size,4)
        ) / ((imgSize-cutSize//2) * 2 + cutSize)

        cutout = tf.expand_dims(cutout, [-1] if data_format=='channels_last' else [1])
        cutout = tf.image.crop_and_resize(cutout, randPos, tf.range(batch_size), [imgSize,imgSize], method='nearest')
        x = tf.multiply(x, cutout)

        if returnDataset:
            return tf.data.Dataset.from_tensors((x, y))#  else x, y
        else:
            return x, y
        
    return map_fn

# To fix:
#   seed is the same
#   profiler doesn't works
def Mixup(alpha=1):
    def map_fn(x, y):
        lam = tf.compat.v1.distributions.Beta(alpha, alpha).sample((256, 1))
        pairedY = tf.random.shuffle(y, 15036)
        y = y * lam + (1 - lam) * pairedY
        lam = tf.expand_dims(tf.expand_dims(lam, [-1]), [-1])
        pairedX = tf.random.shuffle(x, 15036)
        x = x * lam + (1 - lam) * pairedX

        return tf.data.Dataset.from_tensors((x, y))
    return map_fn

# To fix:
#   seed is the same
#   profiler doesn't works
def Cutmix(imgSize,
           batch_size,
           returnDataset=True,
           data_format='channels_last'):
    """
    """
    def map_fn(x, y):
        def make_mask(id):
            lam = tf.random.uniform([1], 0, 1)
            pos = tf.random.uniform([2], 0, imgSize+1, dtype=tf.int32)
            dim = tf.math.minimum([imgSize, imgSize] - pos, tf.cast([imgSize, imgSize] * tf.math.sqrt(1 - lam[0]), dtype=tf.int32))

            mask = tf.pad(tf.zeros(dim), [[pos[0], imgSize-pos[0]-dim[0]], [pos[1], imgSize-pos[1]-dim[1]]], mode='CONSTANT', constant_values=1)
            mask = tf.expand_dims(mask, -1)
            mask = tf.ensure_shape(mask, [32,32,1])

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

        return tf.data.Dataset.from_tensors((x, y))
    return map_fn


def SoftPatchUp():
    def map_fn(x, y):
        return x, y
    return map_fn


def HardPatchUp():
    def map_fn(x, y):
        return x, y
    return map_fn

