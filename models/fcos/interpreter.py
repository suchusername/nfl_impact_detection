import tensorflow as tf
import numpy as np
import yaml
import os.path as osp



class Interpreter:
    def __init__(self, config):
        """
        Builds Interpreter

        config: path to yaml config or preloaded config,
        e.g. fcos/interpreter.yaml
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self._maxpool_count = config["maxpool_count"]
        self._confidence_threshold = config["confidence_threshold"]
        self._restore_size = config["restore_size"]

    def process_one_img(self, img_pred):
        bboxes = process_raw_fcos_output(
            ans = np.expand_dims(img_pred, axis=0),
            maxpool_count = self._maxpool_count,
            detection_threshold=self._confidence_threshold
        )
        all_bboxes = []
        for class_id, class_bboxes in enumerate(bboxes):
            if len(class_bboxes) == 0:
                continue
            class_bboxes = np.array(class_bboxes)
            class_id_vec = np.zeros_like(class_bboxes[:, 0:1]) + class_id
            class_bboxes = np.concatenate(
                [class_bboxes[:, 0:4], class_id_vec, class_bboxes[:, 4:5]], axis=-1
            )
            all_bboxes.append(class_bboxes)
        if len(all_bboxes) == 0:
            return np.zeros((1, 4+len(bboxes)))
        all_bboxes = np.concatenate(all_bboxes, axis=0)
        return sorted(all_bboxes, key=lambda detection: detection[1], reverse=True)
    
    def __call__(self, pred, src_image_bbox, original_shape):
        """
        Builds FCOS interpreter

        inputs - pred: raw FCOS prediction for an image batch

        returns - bboxes: detected bboxes of shape [n,m,6]
        """
        img_bboxes = []
        for img_pred in pred:
            img_bboxes.append(self.process_one_img(img_pred))

        n_img = len(img_bboxes)
        max_bboxes_per_img = np.max([len(b) for b in img_bboxes])
        bbox_size = len(img_bboxes[0][0])
        placeholder = np.zeros((n_img, max_bboxes_per_img, bbox_size))

        for img_idx, bboxes in enumerate(img_bboxes):
            n_bboxes = len(bboxes)
            if n_bboxes != 0:
                placeholder[img_idx, :n_bboxes] = bboxes
        if self._restore_size:
            placeholder = resize(
                placeholder,
                src_image_bbox,
                original_shape
            )

        return placeholder

def process_raw_fcos_output(
    ans, maxpool_count, detection_threshold
):
    detection_channels_count = ans.shape[-1] - 4
    detections = []
    for channel in range(detection_channels_count):
        d = raw_predict_to_bboxes(ans, channel, detection_threshold, maxpool_count)
        detections.append(d)
    return detections


def raw_predict_to_bboxes(ans, channel, detection_threshold, maxpool_count):
    """Prepares sorted detections for NMS"""
    raw_detections = np.argwhere(ans[0, :, :, channel] > detection_threshold)

    detections = []
    for y, x in raw_detections:
        score, prediction = ans[0][y, x, channel], ans[0][y, x, -4:]
        coords = to_coords(prediction, x, y, maxpool_count)
        coords.append(score)
        detections.append(coords)

    return sorted(detections, key=lambda detection: detection[-1], reverse=True)


def to_coords(prediction, x, y, maxpool_count, predictions_in_log_scale=True):
    """Transforms NN's prediction to markup coordinates"""
    """It runs on one particular x,y cell of prediction"""
    fm_stride = 2 ** maxpool_count
    t = 2 ** prediction[-4]
    r = 2 ** prediction[-3]
    b = 2 ** prediction[-2]
    l = 2 ** prediction[-1]
    y0 = int(fm_to_image_coords(y, fm_stride) - t)
    x0 = int(fm_to_image_coords(x, fm_stride) - l)

    h = int(t + b)
    w = int(l + r)
    return [x0, y0, w, h]


def fm_to_image_coords(fm_coord, stride):
    return stride * fm_coord + stride / 2.0 - 0.5

def resize(bboxes, src_bboxes, original_shape, batch=True):
    if batch:
        src_bboxes = np.expand_dims(src_bboxes, axis=1)
        original_shape = np.expand_dims(original_shape, axis=1)
        #print(original_shape, src_bboxes.shape, bboxes.shape)
        bboxes[:,:,0:2] -= src_bboxes[:,:,0:2]
        w_scale = original_shape[:,:,1:2] / src_bboxes[:,:,2:3]
        h_scale = original_shape[:,:,0:1] / src_bboxes[:,:,3:4]
        bboxes[:,:,:4] *= np.concatenate([w_scale, h_scale, w_scale, h_scale], axis=2)
        return bboxes
    else:
        bboxes[:,0:2] -= src_bboxes[0:2]
        w_scale = original_shape[1] / src_bboxes[2]
        h_scale = original_shape[0] / src_bboxes[3]
        bboxes[:,:4] *= [w_scale, h_scale, w_scale, h_scale]
        return bboxes