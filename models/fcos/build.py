import tensorflow as tf
import numpy as np
import os


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), **reg):

        super(ConvBlock, self).__init__(name="")

        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation=None, **reg,
        )
        self.act = tf.keras.layers.Activation("relu")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inp, training=False):
        x = self.conv(inp)
        x = self.act(x)
        x = self.bn(x, training=training)
        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), **reg):

        super(ResidualBlock, self).__init__(name="")

        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation=None, **reg,
        )
        self.act1 = tf.keras.layers.Activation("relu")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation=None, **reg,
        )

        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.Activation("relu")
        self.bn = tf.keras.layers.BatchNormalization()

    def __call__(self, inp, training=False):

        x = self.conv1(inp)
        x = self.act1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)

        x = self.add([x, inp])
        x = self.act(x)
        x = self.bn(x, training=training)

        return x


def build_model(**kwargs):

    # Default arguments
    kwargs["num_classes"] = kwargs.get("num_classes", 1)
    kwargs["input_h"] = kwargs.get("input_h", 300)
    kwargs["input_w"] = kwargs.get("input_w", 400)
    kwargs["l2_reg"] = kwargs.get("l2_reg", 0.001)

    reg = {
        "kernel_regularizer": tf.keras.regularizers.l2(kwargs["l2_reg"]),
        "bias_regularizer": tf.keras.regularizers.l2(kwargs["l2_reg"]),
    }

    img_input = tf.keras.layers.Input(
        shape=(kwargs["input_h"], kwargs["input_w"], 3), name="img"
    )

    # Residual feature extraction

    x = ConvBlock(64, **reg)(img_input)
    x = ResidualBlock(64, **reg)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = ConvBlock(128, **reg)(x)
    x = ResidualBlock(128, **reg)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = ConvBlock(256, **reg)(x)
    x = ResidualBlock(256, **reg)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = ConvBlock(256, **reg)(x)
    feature_map = ConvBlock(256, **reg)(x)

    # Classification (detection) branch

    det = tf.keras.layers.Conv2D(
        kwargs["num_classes"],
        kernel_size=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(kwargs["l2_reg"]),
    )(feature_map)
    det = tf.keras.layers.Activation("sigmoid")(det)

    # Regression branch

    reg = tf.keras.layers.Conv2D(
        4,
        kernel_size=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(10 * kwargs["l2_reg"]),
        bias_regularizer=tf.keras.regularizers.l2(5 * kwargs["l2_reg"]),
    )(feature_map)
    reg = tf.keras.layers.Activation("linear")(reg)

    # Output

    output = tf.keras.layers.Concatenate()([det, reg])

    return tf.keras.Model(img_input, output)
