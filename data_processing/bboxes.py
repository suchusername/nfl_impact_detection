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


class BuildFCOSTarget:
    """
    Prepare feature maps for bounding box regression, like in FCOS.
    
    Arguments:
    image_shape: array, [height, width]
    strides    : list, list of strides on the current layer
    num_classes: int, number of classes to detect
    
    Required keys: "bboxes"
    Added keys: "target0", "target1", ... (depends of len(strides))
    
    Reference:
    https://arxiv.org/abs/2006.09214
    """

    def __init__(self, image_shape, strides, num_classes=1):
        self.image_shape = image_shape
        self.strides = strides
        self.num_classes = num_classes

    def bbox_to_points(self, bbox, s):
        """
        Convert each bounding box to a list of pairs (x,y) and (t,r,b,l), where
            x,y: coordinates on the feature map
            t,r,b,l: values to regress at coordinates (x,y)
            
        Arguments:
        bbox: array of shape (5,), has format [x_min, y_min, width, height, label]
        s   : int, stride on the current feature map
        
        Returns:
        list of pairs (x,y), (t,r,b,l)
        """
        ret = []
        x0, y0, w, h = tuple(bbox[0:4])

        def get_first_point(x0, s):
            start_x = x0 + 0.5 - s / 2
            return np.ceil(start_x / s).astype(int)

        def get_last_point(x0, s):
            start_x = x0 + 0.5 - s / 2
            return np.floor(start_x / s).astype(int)

        start_x = get_first_point(x0, s)
        end_x = get_last_point(x0 + w, s)
        start_y = get_first_point(y0, s)
        end_y = get_last_point(y0 + h, s)

        def fm_to_image_coords(fm_coord, s):
            return s * fm_coord + s / 2.0 - 0.5

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                img_x = fm_to_image_coords(x, s)
                img_y = fm_to_image_coords(y, s)
                t = max(0, img_y - y0)
                r = max(0, x0 + w - img_x)
                b = max(0, y0 + h - img_y)
                l = max(0, img_x - x0)
                ret.append(((x, y), (t, r, b, l)))

        return ret

    def process(self, bboxes):
        """
        Numpy processing.
        
        Arguments:
        bboxes: np.array of shape (n,5), bboxes have format [x_min, x_max, width, height, label]
        
        Returns:
        np.array of shape (?, ?, 4 + self.num_classes), where first two dimensions same as
            dimensions of current feature map
            
            Slices of the last dimension:
                0:C: one-hot encoding of a class a pixel belongs to
                C+0: top boundary regression
                C+1: right boundary regression
                C+2: bottom boundary regression
                C+3: left boundary regression
        """
        # filtering phantom bboxes
        bboxes = bboxes[bboxes.sum(axis=1) > 0]

        # sorting bboxes by area (descending)
        bboxes = np.array(sorted(bboxes, key=lambda x: x[2] * x[3], reverse=True))

        fmaps = []

        for s in self.strides:

            # generating a feature map
            fmap = np.zeros(
                shape=(
                    np.floor(self.image_shape[0] / s).astype(int),
                    np.floor(self.image_shape[1] / s).astype(int),
                    4 + self.num_classes,
                )
            )

            # filling a feature map with info about bboxes
            for bbox in bboxes:

                fm_points = self.bbox_to_points(bbox, s)
                for p, t in fm_points:

                    if p[1] < fmap.shape[0] and p[0] < fmap.shape[1]:

                        # setting a class label
                        fmap[p[1], p[0], : self.num_classes] = 0
                        fmap[p[1], p[0], int(bbox[4])] = 1

                        # setting regression targets
                        for i in range(0, 4):
                            fmap[p[1], p[0], self.num_classes + i] = np.log(
                                t[i] + 1.001
                            ) / np.log(2)

            # adding feature map to a list
            fmaps.append(fmap.astype(np.float32))

        return tuple(fmaps)

    def __call__(self, sample):
        """
        Tensorflow processing.
        
        Arguments:
        sample: dict, element of tf.data.Dataset
        
        Returns:
        transformed sample
        """
        ret = tf.numpy_function(
            self.process, [sample["bboxes"]], [tf.float32] * len(self.strides)
        )

        for i in range(len(self.strides)):
            sample["target" + str(i)] = ret[i]

        return sample
