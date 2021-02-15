import os
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from model import Model


flags.DEFINE_string(
    "test_dataset", "data/test", "Path to testing dataset",
)
flags.DEFINE_string(
    "weights_path", "checkpoints/cml_3.tf", "Path to weights checkpoint."
)
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("img_width", 64, "image width")
flags.DEFINE_integer("img_height", 128, "image height")


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, size)
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

    logging.info("Creating model and starting testing.")
    model = tf.keras.models.load_model(
        "~/deepsort/models/original_2020-11-06 02:01:33.498292"
    )

    # model = Model((FLAGS.img_height, FLAGS.img_width), num_classes=1500, training=True)
    # model.load_weights(FLAGS.weights_path)

    @tf.function
    def loss_fn(label, inference):
        label = tf.squeeze(label)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(label, inference)

    model.compile(
        # optimizer=Adam(),
        # loss=SparseCategoricalCrossentropy(from_logits=True),
        loss=loss_fn,
        metrics=[SparseCategoricalAccuracy()],
    )
    score = model.evaluate(test_dataset)

    print(score)


if __name__ == "__main__":
    app.run(main)
