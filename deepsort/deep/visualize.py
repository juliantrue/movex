import os
from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)

from model import Model


flags.DEFINE_string(
    "test_dataset", "data/test", "Path to testing dataset",
)
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_integer("img_width", 64, "image width")
flags.DEFINE_integer("img_height", 128, "image height")


def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


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


def main(argv):
    feature_description = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
    }

    logging.info("Loading testing dataset...")
    test_dataset = tf.data.TFRecordDataset(
        filenames=[
            os.path.join(FLAGS.test_dataset, path)
            for path in os.listdir(FLAGS.test_dataset)
        ]
    )
    test_dataset = test_dataset.shuffle(buffer_size=4096)
    test_dataset = test_dataset.map(lambda x: parse_tfrecord(x, feature_description))
    test_dataset = test_dataset.map(
        lambda x, y: (transform_images(x, [FLAGS.img_height, FLAGS.img_width]), y)
    )
    test_dataset = test_dataset.batch(FLAGS.batch_size)
    logging.info("Done!")

    logging.info("Creating model and starting training.")
    model = Model((FLAGS.img_height, FLAGS.img_width))
    model.load_weights("checkpoints/best_model/cml_10.tf")

    # Validation step.
    for step, (x_batch_val, y_batch_val) in enumerate(test_dataset.take(1000)):

        # Inference
        inference = model.predict(x_batch_val)

        img0 = x_batch_val[0].numpy() * 255
        img1 = x_batch_val[1].numpy() * 255

        img0 = cv2.resize(img0.astype(np.uint8), (128, 256))
        img1 = cv2.resize(img1.astype(np.uint8), (128, 256))

        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

        feature0 = np.array([inference[0]])
        feature1 = np.array([inference[1]])

        sim = cosine_distance(feature0, feature1)
        logging.info("Similarity: {}".format(sim))

        img = np.concatenate((img0, img1), axis=1)
        cv2.imshow("test", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    app.run(main)
