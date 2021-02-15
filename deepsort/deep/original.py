import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Input,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    ELU,
    MaxPooling2D,
    Multiply,
    BatchNormalization,
    LayerNormalization,
)
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import unit_norm


def conv(x, filters, strides=1, weight_decay=1e-8, bn=True):
    initializer = TruncatedNormal(stddev=1e-3)
    regularizer = l2(weight_decay)
    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=False,
    )(x)

    if bn:
        x = BatchNormalization()(x)

    x = ELU()(x)
    return x


def residual(x, filters, strides=1, dropout_rate=0.6):
    incoming = x
    x = conv(x, filters, strides=strides)
    x = Dropout(dropout_rate)(x)
    x = conv(x, filters, bn=False)

    if not incoming.shape == x.shape:
        # Upsample via convolution to match the shape
        incoming = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            use_bias=False,
        )(incoming)

    x = Add()([incoming, x])

    return x


def Model(size, channels=3, num_classes=1500, feature_dim=128, training=False):
    x = inputs = Input([size[0], size[1], channels], name="input")
    x = conv(x, 32)
    x = conv(x, 32)
    x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)
    x = residual(x, 32)
    x = residual(x, 32)
    x = residual(x, 64, strides=2)
    x = residual(x, 64)
    x = residual(x, 128, strides=2)
    x = residual(x, 128)
    x = Flatten()(x)
    x = Dropout(0.6)(x)
    x = BatchNormalization()(x)
    x = Dense(
        feature_dim,
        kernel_regularizer=l2(1e-8),
        kernel_initializer=TruncatedNormal(stddev=1e-3),
    )(x)
    features = tf.nn.l2_normalize(x, axis=1)

    if training:
        outputs = Dense(num_classes)(x)

    else:
        outputs = features

    return tf.keras.Model(inputs, outputs, name="model")
