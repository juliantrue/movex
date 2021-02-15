import os
from absl import flags
from absl.flags import FLAGS
import tensorflow as tf


feature_description = {
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
}


def transform_and_augment_images(x_train, size):
    re_size = [size[0] + 5, size[1] + 5]
    x_train = tf.image.resize(x_train, re_size, method="bicubic")
    x_train = x_train / 255
    x_train = tf.image.random_flip_left_right(x_train)
    x_train = tf.image.random_jpeg_quality(x_train, 50, 95)
    x_train = tf.image.random_crop(x_train, size=[size[0], size[1], 3])
    return x_train


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, size, method="bicubic")
    x_train = x_train / 255
    return x_train


def parse_tfrecord(example_proto, feature_description):
    # Parse the input tf.Example proto using the dictionary above.
    example = tf.io.parse_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(example["image/encoded"])
    label = example["label"]
    return image, label


def load_train_dataset(train_dir_path, batch_size):
    train_filenames = [
        os.path.join(train_dir_path, path) for path in os.listdir(train_dir_path)
    ]
    train_dataset = tf.data.TFRecordDataset(filenames=train_filenames)
    train_dataset = train_dataset.map(lambda x: parse_tfrecord(x, feature_description))
    train_dataset = train_dataset.shuffle(buffer_size=8192)
    train_dataset = train_dataset.map(
        lambda x, y: (
            transform_and_augment_images(x, [FLAGS.img_height, FLAGS.img_width]),
            y,
        )
    )
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset


def load_test_dataset(test_dir_path, batch_size):
    test_filenames = [
        os.path.join(test_dir_path, path) for path in os.listdir(test_dir_path)
    ]
    test_dataset = tf.data.TFRecordDataset(filenames=test_filenames)
    test_dataset = test_dataset.map(lambda x: parse_tfrecord(x, feature_description))
    test_dataset = test_dataset.map(
        lambda x, y: (transform_images(x, [FLAGS.img_height, FLAGS.img_width]), y)
    )
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset
