import os
import argparse
import time
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["cifar10"], required=True, help="dataset to create")  # , "imagenet1k"
    # parser.add_argument("-p", "--path", help="path to imagenet")
    parser.add_argument("-w", "--write-dir", default="tfrecords/", help="path to write TFRecords")
    
    cfg = parser.parse_args()

    print(cfg)

    wpath = os.path.join(cfg.write_dir, cfg.dataset)

    if not os.path.isdir(wpath):
        os.makedirs(wpath)

    if cfg.dataset == "cifar10":
        cifar10(wpath)
    elif cfg.dataset == "imagenet1k":
        raise NotImplementedError("imagenet1k not implemented")


def cifar10(path):
    print("Loading CIFAR10")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    tf.constant([2])


    start = time.perf_counter()
    print("Serializing Train Batch")

    with tf.io.TFRecordWriter(os.path.join(path, "train.tfrecord")) as writer:
        data = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "imgs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ tf.io.serialize_tensor(train_images).numpy() ])),
                    "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ tf.io.serialize_tensor(train_labels).numpy() ]))
                }
            )
        )

        print("Writing to TFRecord file")
        writer.write(
            data.SerializeToString()
        )

    
    with tf.io.TFRecordWriter(os.path.join(path, "val.tfrecord")) as writer:
        print("Serializing Validation Batch")
        data = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "imgs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ tf.io.serialize_tensor(test_images).numpy() ])),
                    "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ tf.io.serialize_tensor(test_labels).numpy() ])),
                }
            )
        )

        print("Writing to TFRecord file")
        writer.write(
            data.SerializeToString()
        )

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    main()
