import os
import numpy as np
import cv2
import tensorflow as tf


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
    
    Required keys: img (and bboxes)
    
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

    def process(self, img):
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

        if self.mode == "topleft":
            output[: scaled.shape[0], : scaled.shape[1]] = scaled
        elif self.mode == "center":
            cx = (output.shape[0] - scaled.shape[0]) // 2
            cy = (output.shape[1] - scaled.shape[1]) // 2
            output[cx : cx + scaled.shape[0], cy : cy + scaled.shape[1]] = scaled

        return output

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
