import json
import os
import os.path as osp
import pandas as pd
import numpy as np
import tensorflow as tf

from .walk_directory import walk_directory

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = "data"  # path to directory with data


def load_detection_ds(
    images_dir="data/images", markup_dir="data/images_markup", n_bboxes=None
):
    """
    Load raw dataset for helmet detection.
    
    Arguments:
    images_dir: str, path to directory with images relative to root of project
    markup_dir: str, path to directory with markup relative to root of project
    n_bboxes  : int, length of bboxes array for each sample (to make every sample have the same shape)
    
    Returns:
    tf.data.Dataset with keys:
        img_path   : str, absolute path to image
        markup_path: str, absolute path to markup file
    """
    full_images_dir = osp.join(ROOT_DIR, images_dir)
    full_markup_dir = osp.join(ROOT_DIR, markup_dir)

    ret = {"img_path": [], "markup_path": []}

    config = {"white_list": [images_dir]}
    for img_path in walk_directory(config, mode="images"):
        markup_path = osp.join(
            full_markup_dir, osp.relpath(img_path, full_images_dir) + ".json"
        )
        ret["img_path"].append(img_path)
        ret["markup_path"].append(markup_path)

    return tf.data.Dataset.from_tensor_slices(ret)


class Processor:
    """
    Builds a simple pipeline of transformations.
    
    Arguments (and attributes):
    transformations: list, contains callable transformations
    feature_keys   : list, contains keys that should be returned as a tuple
    label_key      : str, key of label
    
    Methods:
    __call__
    """

    def __init__(self, transformations=None, feature_keys=["img"], label_key="bboxes"):
        self.transformations = transformations
        self.feature_keys = feature_keys
        self.label_key = label_key

    def __call__(self, sample):
        """
        Applies a pipeline of transformations to one sample.
        
        Arguments:
        sample: dict, element of tf.data.Dataset
        
        Returns:
        if self.keys is None:
            transformed sample
        else:
            a tuple (x, y)
        """
        for f in self.transformations:
            sample = f(sample)

        if self.feature_keys is None:
            return sample
        return (
            tuple([sample[k] for k in self.feature_keys]),
            sample[self.label_key],
        )
