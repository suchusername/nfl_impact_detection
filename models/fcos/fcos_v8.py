import tensorflow as tf
import numpy as np
import os

from data_processing.Processor import Processor
from data_processing import image, bboxes
from models.fcos.loss import fcos_detection_loss, fcos_regression_loss


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), dropout=0, **reg):

        super(ResidualBlock, self).__init__(name="")

        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation=None, **reg,
        )
        self.act1 = tf.keras.layers.Activation("relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)

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
        x = self.dropout1(x, training=training)
        x = self.conv2(x)

        x = self.add([x, inp])
        x = self.act(x)
        x = self.bn(x, training=training)

        return x


def build_model(**kwargs):

    # Default arguments
    kwargs["num_classes"] = kwargs.get("num_classes", 1)
    kwargs["input_h"] = kwargs.get("input_h", 296)
    kwargs["input_w"] = kwargs.get("input_w", 400)
    kwargs["l2_reg"] = kwargs.get("l2_reg", 0.001)
    kwargs["dropout"] = kwargs.get("dropout", 0.1)

    reg = {
        "kernel_regularizer": tf.keras.regularizers.l2(kwargs["l2_reg"]),
        "bias_regularizer": tf.keras.regularizers.l2(kwargs["l2_reg"]),
    }

    # Input

    img_input = tf.keras.layers.Input(
        shape=(kwargs["input_h"], kwargs["input_w"], 3), name="img"
    )

    # Backbone: residual feature extraction

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", **reg)(
        img_input
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(kwargs["dropout"])(x)
    x = ResidualBlock(32)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", **reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(kwargs["dropout"])(x)
    x = ResidualBlock(64)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", **reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(kwargs["dropout"])(x)
    x = ResidualBlock(128)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", **reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(kwargs["dropout"])(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", **reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(kwargs["dropout"])(x)

    feature_map = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", **reg
    )(x)

    # Head: detection (classification) branch

    det = tf.keras.layers.Conv2D(
        kwargs["num_classes"],
        kernel_size=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(kwargs["l2_reg"]),
    )(feature_map)
    det = tf.keras.layers.Activation("sigmoid")(det)

    # Head: regression branch

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

    return tf.keras.Model(inputs=[img_input], outputs=[output])


def build_loss(**kwargs):

    # Default arguments
    kwargs["regression_weight"] = kwargs.get("regression_weight", 0.05)

    def loss(y_true, y_pred):
        ret = fcos_detection_loss(y_true, y_pred)
        ret += kwargs["regression_weight"] * fcos_regression_loss(y_true, y_pred)
        return tf.identity(ret, name="loss")

    return loss


def build_metrics(**kwargs):
    loss = build_loss(**kwargs)
    return [loss, fcos_detection_loss, fcos_regression_loss]


def build_train_processor(**kwargs):

    # Default arguments
    kwargs["num_classes"] = kwargs.get("num_classes", 1)
    kwargs["n_bboxes"] = kwargs.get("n_bboxes", 80)
    kwargs["input_h"] = kwargs.get("input_h", 296)
    kwargs["input_w"] = kwargs.get("input_w", 400)
    kwargs["strides"] = kwargs.get("strides", [8])

    transformations = [
        image.LoadImage(),
        bboxes.LoadBboxes(n_bboxes=kwargs["n_bboxes"]),
        image.HFlip(),
        image.ResizeKeepRatio(kwargs["input_h"], kwargs["input_w"]),
        image.Normalize(),
        bboxes.BuildFCOSTarget(
            (kwargs["input_h"], kwargs["input_w"]), kwargs["strides"]
        ),
    ]

    processor = Processor(transformations, feature_keys=["img"], label_keys=["target0"])

    return processor


def build_processor(**kwargs):

    # Default arguments
    kwargs["num_classes"] = kwargs.get("num_classes", 1)
    kwargs["n_bboxes"] = kwargs.get("n_bboxes", 80)
    kwargs["input_h"] = kwargs.get("input_h", 296)
    kwargs["input_w"] = kwargs.get("input_w", 400)
    kwargs["strides"] = kwargs.get("strides", [8])

    transformations = [
        image.LoadImage(),
        bboxes.LoadBboxes(n_bboxes=kwargs["n_bboxes"]),
        image.ResizeKeepRatio(kwargs["input_h"], kwargs["input_w"]),
        image.Normalize(),
        bboxes.BuildFCOSTarget(
            (kwargs["input_h"], kwargs["input_w"]), kwargs["strides"]
        ),
    ]

    processor = Processor(transformations, feature_keys=["img"], label_keys=["target0"])

    return processor
