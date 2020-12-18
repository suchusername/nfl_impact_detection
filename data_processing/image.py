import os
import numpy as np
import cv2
import tensorflow as tf
from .nms import filter_phantoms


class LoadImage:
    """
    Load RGB image as a numpy array.
    
    Required keys: img_path
    Added keys: img
    """

    def __init__(self):
        pass

    def process(self, img_path, byte=True):
        """
        Numpy processing.
        
        Arguments:
        img_path: str or bytes, path to an image
        
        Returns:
        np.array, 3d-array with raw image data (as np.float32) in RGB
        """
        if byte:
            img_path = img_path.decode("utf-8")
        img = cv2.imread(img_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    def __call__(self, sample):
        """
        Tensorflow processing.
        
        Arguments:
        sample: dict, element of tf.data.Dataset
        
        Returns:
        transformed sample
        """
        sample["img"] = tf.numpy_function(
            self.process, [sample["img_path"]], [tf.float32]
        )[0]
        return sample


class ResizeKeepRatio:
    """
    Resizes an image while preserving an aspect ratio.
    
    Required keys: img, bboxes
    
    Arguments:
    h   : int, new height
    w   : int, new width
    mode: str, "center" or "topleft"
    """

    def __init__(self, h, w, mode="center"):
        self.h = h
        self.w = w
        if mode not in ["center", "topleft"]:
            raise ValueError("Invalid mode.")
        self.mode = mode

    def resize_bboxes(self, bboxes, old_img_bbox, new_img_bbox):
        """
        Resize bboxes.
        
        Arguments:
        bboxes      : np.array of shape (n,5), [x_min, y_min, width, height, label]
        old_img_bbox: np.array of shape (4,), bbox of original image
        new_img_bbox: np.array of shape (4,), bbox of resized image
        
        Returns:
        np.array of shape (n,5), resized bboxes
        """
        resized_bboxes = np.zeros_like(bboxes)
        valid_bboxes = bboxes[bboxes.sum(axis=1) > 0]
        valid_bboxes[:, 0:2] -= old_img_bbox[0:2]
        valid_bboxes[:, 0:4] *= np.tile(new_img_bbox[2:4] / old_img_bbox[2:4], 2)
        valid_bboxes[:, 0:2] += new_img_bbox[0:2]
        resized_bboxes[: len(valid_bboxes)] = valid_bboxes
        return resized_bboxes

    def process(self, img, bboxes):
        """
        Numpy processing.
        
        Arguments:
        img: np.array, 3d-array with image data
        
        Returns:
        np.array, 3d-array with transformed image data
        """
        output = np.zeros_like(img, shape=(self.h, self.w, img.shape[2]))
        scale = min(self.h / img.shape[0], self.w / img.shape[1])
        scaled = cv2.resize(img, None, fx=scale, fy=scale)

        n_bboxes = len(bboxes)
        new_bboxes = np.zeros_like(bboxes)

        if self.mode == "topleft":
            image_bbox = np.array([0, 0, scaled.shape[1], scaled.shape[0]])
            delta_x = 0
            delta_y = 0

        elif self.mode == "center":
            image_bbox = np.array(
                [
                    (output.shape[1] - scaled.shape[1]) // 2,
                    (output.shape[0] - scaled.shape[0]) // 2,
                    scaled.shape[1],
                    scaled.shape[0],
                ]
            )
            delta_x = (output.shape[1] - scaled.shape[1]) // 2
            delta_y = (output.shape[0] - scaled.shape[0]) // 2

        output[
            image_bbox[1] : image_bbox[1] + image_bbox[3],
            image_bbox[0] : image_bbox[0] + image_bbox[2],
        ] = scaled

        bboxes = self.resize_bboxes(
            bboxes,
            old_img_bbox=np.array([0, 0, img.shape[1], img.shape[0]]),
            new_img_bbox=image_bbox,
        )

        return output, bboxes, np.array((delta_x, delta_y, scaled.shape[1], scaled.shape[0]), dtype=np.float32)
        # return output, bboxes, np.array((0, 0, scaled.shape[1], scaled.shape[0]), dtype=np.float32)

    def __call__(self, sample):
        """
        Tensorflow processing.
        
        Arguments:
        sample: dict, element of tf.data.Dataset
        
        Returns:
        transformed sample
        """
        sample["img"], sample["bboxes"], sample["src_img_bbox"] = tf.numpy_function(
            self.process, [sample["img"], sample["bboxes"]], [tf.float32, tf.float32, tf.float32],
        )
        return sample


class Normalize:
    """
    Linear transformation Y = a*X + b of image values.
    
    Required keys: img
    
    Arguments:
    a: float
    b: float
    """

    def __init__(self, a=1 / 255.0, b=0.0):
        self.a = a
        self.b = b

    def process(self, img):
        """
        Numpy processing.
        
        Arguments:
        img: np.array, 3d-array with image data
        
        Returns:
        np.array, 3d-array with transformed image data
        """
        return img.astype(np.float32) * self.a + self.b

    def __call__(self, sample):
        """
        Tensorflow processing.
        
        Arguments:
        sample: dict, element of tf.data.Dataset
        
        Returns:
        transformed sample
        """
        sample["img"] = tf.numpy_function(self.process, [sample["img"]], [tf.float32])[
            0
        ]
        return sample

class HFlip:
    """
    Random flip for image and bboxes
    
    Required keys: img, bboxes
    """

    def __init__(self):
        pass

    def process(self, img, bboxes):
        # print(np.random.randint(2))
        if np.random.randint(2) > 0: # 0.5 prob of flip
            img = img[:, ::-1]
            result = np.zeros_like(bboxes)
            bboxes = filter_phantoms(bboxes)
            bboxes[:, 0] = img.shape[1] - bboxes[:, 0] - bboxes[:, 2]
            result[: len(bboxes)] = bboxes	
            bboxes = result
        
        return img, bboxes

    def __call__(self, sample):
        
        if "bboxes" in sample.keys():
            sample["img"], sample["bboxes"] = tf.numpy_function(
                self.process, [sample["img"], sample["bboxes"]], [tf.float32, tf.float32],
            )

        return sample