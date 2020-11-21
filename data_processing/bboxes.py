import os
import numpy as np
import tensorflow as tf

from .io import load_gmc


class LoadBboxes:
    """
    Load bboxes from markup.
    
    Arguments:
    tags    : list, bboxes with these tags will be loaded
    n_bboxes: int, the list of bboxes is extended to this length 
    
    Required keys: markup_path
    Added keys: bboxes
    """

    def __init__(self, tags=["h"], n_bboxes=None):
        self.tags = tags
        self.n_bboxes = n_bboxes

    def process(self, markup_path, byte=True):
        """
        Numpy processing.
        
        Arguments:
        markup_path: str or bytes, path to a markup file
        
        Returns:
        np.array of shape (n_bboxes, 5) with bboxes [x_min, y_min, width, height, label]
        """
        if byte:
            markup_path = markup_path.decode("utf-8")
        markup = load_gmc(markup_path, tags=self.tags, n_bboxes=self.n_bboxes)
        return markup.astype(np.float32)

    def __call__(self, sample):
        """
        Tensorflow processing.
        
        Arguments:
        sample: dict, element of tf.data.Dataset
        
        Returns:
        transformed sample
        """
        sample["bboxes"] = tf.numpy_function(
            self.process, [sample["markup_path"]], [tf.float32]
        )[0]
        return sample
