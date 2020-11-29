import tensorflow as tf
import tensorflow.keras.backend as K


def fcos_detection_loss(y_true, y_pred):
    y_detection_gt = y_true[:, :, :, 0:-4]
    y_detection_predicted = y_pred[:, :, :, 0:-4]
    pt = 1 - tf.math.abs(y_detection_gt - y_detection_predicted)
    eps = 0.000001
    gamma = 2
    ret = -K.mean(tf.math.pow((1 - pt), gamma) * tf.math.log(pt + eps))
    return ret


def fcos_regression_loss(y_true, y_pred):
    y_detection_gt = tf.convert_to_tensor(tf.reduce_sum(y_true[:, :, :, 0:-4], axis=-1))
    y_regression_gt = tf.convert_to_tensor(y_true[:, :, :, -4:])
    y_regression_predicted = tf.convert_to_tensor(y_pred[:, :, :, -4:])
    diff = (y_regression_predicted - y_regression_gt) * (
        y_regression_predicted - y_regression_gt
    )
    diff = K.mean(diff, axis=-1)
    gt_weighted = diff * y_detection_gt
    ret = K.sum(gt_weighted) / (K.sum(y_detection_gt) + 0.0001)
    return ret
