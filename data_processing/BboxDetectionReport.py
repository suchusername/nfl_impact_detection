import os.path as osp
import os
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import tqdm
import warnings
import json

from .nms import IoU
from .io import load_gmc, load_bbox

from .walk_directory import walk_directory


class BboxDetectionReport:
    true = "true"
    pred = "pred"

    def __init__(self, overwrite=False):
        self._report = dict()
        self._matched = False

    @property
    def images(self):
        return list(self._report.keys())

    def verify_obj_tags(self, true_file_path, obj_prop_filter=None):
        with open(true_file_path, "r") as file:
            json_file = json.load(file)
            if not obj_prop_filter:
                image_flag = 1
                return image_flag
            image_flag = 0
            for category in obj_prop_filter:
                if len(json_file) > 0:
                    for object_ in json_file["objects"]:
                        if "tags" in object_ and object_["type"] == "rect":
                            if obj_prop_filter[category] == "include":
                                if category in object_["tags"]:
                                    image_flag = 1
                                    break
                                else:
                                    image_flag = 0
                            elif obj_prop_filter[category] == "exclude":
                                if category not in object_["tags"]:
                                    image_flag = 1
                                else:
                                    image_flag = 0
                                    return image_flag
                    if image_flag == 0:
                        return image_flag
            return image_flag

    def verify_img_properties(self, true_file_path, img_prop_filter=None):
        with open(true_file_path, "r") as file:
            json_file = json.load(file)
            image_flag = 1
            if not img_prop_filter:
                return image_flag
            if "properties" in json_file:
                for category in img_prop_filter:
                    if category not in json_file["properties"]:
                        print(f"{true_file_path} haven't the {category}")
                        continue
                    if img_prop_filter[category] == "include":
                        if json_file["properties"][category]:
                            image_flag = 1
                        else:
                            image_flag = 0
                            break
                    elif img_prop_filter[category] == "exclude":
                        if not json_file["properties"][category]:
                            image_flag = 1
                        else:
                            image_flag = 0
                            break
            elif "properties" not in json_file:
                print(f"{true_file_path} haven't the 'properties'")
            return image_flag

    def from_serialized_report(
        self,
        dir_true,
        dir_pred,
        class_idx,
        img_prop_filter=None,
        obj_prop_filter=None,
        track_ids=False,
    ):
        for pred_file_path in walk_directory(None, mode="markup", start_root=dir_pred):
            true_file_path = osp.join(dir_true, osp.relpath(pred_file_path, dir_pred))
            path_split = true_file_path.split("/")
            true_file_path = "/".join(path_split)
            if not osp.exists(true_file_path):
                continue
            bboxes_true = load_gmc(
                true_file_path, tags=["h"], track_ids=track_ids
            )
            bboxes_true = bboxes_true[bboxes_true[:, 4] == class_idx]
            bboxes_pred = load_bbox(pred_file_path)
            bboxes_pred = bboxes_pred[bboxes_pred[:, 4] == class_idx]

            if self.verify_img_properties(
                true_file_path, img_prop_filter
            ) and self.verify_obj_tags(true_file_path, obj_prop_filter):

                self.add_pred(bboxes_pred, osp.relpath(pred_file_path, dir_pred))
                self.add_true(bboxes_true, osp.relpath(pred_file_path, dir_pred))

    def eval_on_ds(self, ds, model, interpreter, class_idx=None):
        """
        Build a report from a batched tf.data.Dataset, the model and its interpreter.
        
        Note: each list of bboxes in sample["bboxes"] contains phantom boxes, so filtering is applied here.
        
        Args:
        ds         : BATCHED tf.data.Dataset, each sample contains keys "img", "img_path" and "bboxes"
        model      : a callable model that can predict on batches (takes sample["img"] as input)
        interpreter: a callable interpreter that transforms raw batches of model predictions into final predictions
        class_idx  : only ground truth bounding boxes and predictions with these labels will be added to the report,
            if None, all are added
        
        Returns: nothing
            it just modifies self._report
        """
        warnings.warn(
            "deprecated, consider using from_serialized_report", DeprecationWarning
        )
        for sample_batch in tqdm.tqdm(ds):
            raw_bboxes_batch = model(sample_batch["img"])
            bboxes_batch = interpreter(raw_bboxes_batch, sample_batch)
            for i in range(len(bboxes_batch)):
                if class_idx is not None:
                    bboxes_batch[i] = bboxes_batch[i][
                        np.in1d(bboxes_batch[i][:, 4], class_idx)
                    ]
                self.add_pred(
                    bboxes_batch[i], sample_batch["img_path"][i].numpy().decode("utf-8")
                )
                gt_boxes = sample_batch["bboxes"][i].numpy()
                gt_boxes = gt_boxes[
                    np.sum(gt_boxes, axis=1) > 0
                ]  # filtering phantom ground truth bboxes
                if class_idx is not None:
                    gt_boxes = gt_boxes[
                        np.in1d(gt_boxes[:, 4], class_idx)
                    ]  # selecting bboxes of correct classes
                self.add_true(
                    gt_boxes, sample_batch["img_path"][i].numpy().decode("utf-8")
                )

    def eval_on_dirs(self, true_dir, pred_dir, iou_thres):
        warnings.warn(
            "deprecated, consider using from_serialized_report", DeprecationWarning
        )
        for sample in tqdm.tqdm(ds, "predicting"):
            if model_predicts_on_batch:
                raw_bboxes_pred = model(sample["img"])
                bboxes_pred = interpreter(raw_bboxes_pred, sample)[0]
            else:
                raw_bboxes_pred = model(sample["img"][0])
                sample_ = {k: sample[k][0] for k in sample.keys()}
                bboxes_pred = interpreter([raw_bboxes_pred], [sample_])[0]
            self.add_pred(bboxes_pred, sample["img_path"][0].numpy().decode("utf-8"))
            gt_bboxes = sample["bboxes"][0].numpy()
            self.add_true(gt_bboxes, sample["img_path"][0].numpy().decode("utf-8"))
        self.match(iou_thres=iou_thres)

    def add_pred(self, bboxes, image_tag):
        assert bboxes.shape[1] >= 6, "seems bboxes are w/o confidence"
        if image_tag not in self._report:
            self._report[image_tag] = dict()
        self._report[image_tag][self.pred] = bboxes
        self._matched = False

    def add_true(self, bboxes, image_tag):
        if image_tag not in self._report:
            self._report[image_tag] = dict()
        self._report[image_tag][self.true] = bboxes
        self._matched = False

    def filter_report(self, class_idx):
        """
        Remove unnecessary classes from self._report.
        
        Args:
        class_idx: a list of labels to retain
        
        Returns: nothing
            it just modifies self._report
        """
        for img in self.images:
            img_rep = self._report[img]
            pred_idx = np.in1d(img_rep["pred"][:, 4], class_idx)
            img_rep["pred"] = img_rep["pred"][pred_idx]
            true_idx = np.in1d(img_rep["true"][:, 4], class_idx)
            img_rep["true"] = img_rep["true"][true_idx]

    def match(self, iou_thres, verbose=False):
        if verbose:
            gen = tqdm.tqdm(self.images, desc="matching")
        else:
            gen = self.images
        for img in gen:
            img_rep = self._report[img]
            if (self.pred not in img_rep) or (self.true not in img_rep):
                raise ValueError(
                    f"attempt to match, but {img} does not have cnn and/or gt bboxes"
                )

            if (len(img_rep[self.pred]) > 0) and (len(img_rep[self.true]) > 0):
                img_rep["matches"] = self.match_impl(
                    objects_pred=img_rep[self.pred],
                    objects_true=img_rep[self.true],
                    distance_thres=iou_thres,
                )
            else:
                img_rep["matches"] = np.empty((0, 2), dtype=int)
        self._matched = True

    def precision_recall_curve(self, return_conf=False):
        if not self._matched:
            raise ValueError("not matched. Consider calling match(iou=0.5)")
        TotTrue = 0
        for img in self.images:
            img_rep = self._report[img]
            TotTrue += len(img_rep["true"])

        all_confidences = []
        good_detections_mask = []
        for img in self.images:
            img_rep = self._report[img]
            if len(img_rep["pred"]) > 0:
                all_confidences.extend(img_rep["pred"][:, 5])
                detections_mask = np.zeros(len(img_rep["pred"]), dtype=np.bool)
            else:
                continue
            if len(img_rep["matches"]) > 0:
                detections_mask[img_rep["matches"][:, 0]] = 1
            good_detections_mask.extend(detections_mask)
        asrt = np.argsort(all_confidences)[::-1]
        all_confidences = np.array(all_confidences)[asrt]
        good_detections_mask = np.array(good_detections_mask)[asrt]
        precisions = np.cumsum(good_detections_mask) / np.arange(
            1, len(good_detections_mask) + 1
        )
        recalls = np.cumsum(good_detections_mask) / TotTrue
        if return_conf:
            return precisions, recalls, all_confidences
        else:
            return precisions, recalls

    @staticmethod
    def greatest_x_for_given_y(x, y, y_target, eps=0.01):
        good_x = x[np.argwhere(np.abs(y - y_target) < eps)]
        if len(good_x) == 0:
            return 0
        else:
            return np.max(good_x)

    def pfr(self, fixed_recall):
        if not self._matched:
            raise ValueError("not matched. Consider calling match(iou=0.5)")
        precisions, recalls = self.precision_recall_curve()
        return self.greatest_x_for_given_y(precisions, recalls, fixed_recall)

    def rfp(self, fixed_precision):
        if not self._matched:
            raise ValueError("not matched. Consider calling match(iou=0.5)")
        precisions, recalls = self.precision_recall_curve()
        return self.greatest_x_for_given_y(recalls, precisions, fixed_precision)

    @staticmethod
    def match_impl(
        objects_true, objects_pred, distance_thres,
    ):
        if len(objects_pred) == 0:
            return np.empty((0, 2), dtype=int)

        # eval pairwise dists & sort (dist + conf)
        dist_mat = pairwise_distances(
            objects_pred[:, 0:4], objects_true[:, 0:4], metric=IoU
        )
        match_mat = dist_mat > distance_thres
        asrt_dist = np.argsort(dist_mat, axis=1)[:, ::-1]

        asrt_conf = np.argsort(objects_pred[:, 5])[::-1]

        # greedy matching
        used_true = list()
        matches = list()
        # for each detection
        for pred_idx in asrt_conf:
            for true_idx in asrt_dist[pred_idx]:
                # if ref sample is not used & close enough - match
                if (true_idx not in used_true) and match_mat[pred_idx, true_idx]:
                    used_true.append(true_idx)
                    matches.append((pred_idx, true_idx))
                    break
        matches = np.array(matches).astype(int)
        return matches

    @staticmethod
    def pr2convex01(precision, recall):
        p, r = precision, recall
        asrt = np.argsort(r)[::-1]
        print(p)
        print(r)
        print(asrt)
        r = r[asrt]
        p = p[asrt]
        r = np.concatenate([(1, r[0]), r, (0, r[-1])])
        p = np.concatenate([(0, 0), p, (1, 1)])
        p_convex = np.maximum.accumulate(p)
        return p_convex, r

    def ap(self, recall_interval):
        p, r = self.precision_recall_curve()
        p, r = self.pr2convex01(p, r)
        good_recalls = np.logical_and(r >= recall_interval[0], r <= recall_interval[1])
        if len(good_recalls) < 3:
            return 0
        p = p[good_recalls]
        r = r[good_recalls]
        steps = r[:-1] - r[1:]
        steps /= np.sum(steps)
        top = np.sum(p[1:] * steps)
        bot = np.sum(p[:-1] * steps)
        area = np.mean((top, bot))
        return area

    def ar(self, precision_interval):
        p, r = self.precision_recall_curve()
        p, r = self.pr2convex01(p, r)
        good_precisions = np.logical_and(
            p >= precision_interval[0], p <= precision_interval[1]
        )
        if len(good_precisions) < 3:
            return 0
        p = p[good_precisions]
        r = r[good_precisions]

        steps = p[1:] - p[:-1]
        steps /= np.sum(steps)
        top = np.sum(r[1:] * steps)
        bot = np.sum(r[:-1] * steps)
        area = np.mean((top, bot))
        return area

    def run_standard_evaluation(self):
        self.match(0.5)
        return {
            "precision_at_.3recall.5IoU": np.round(self.pfr(0.3), 3),
            "precision_at_.5recall.5IoU": np.round(self.pfr(0.5), 3),
            "precision_at_.8recall.5IoU": np.round(self.pfr(0.8), 3),
            "precision_at_.9recall.5IoU": np.round(self.pfr(0.9), 3),
            "recall_at_.3precision.5IoU": np.round(self.rfp(0.3), 3),
            "recall_at_.5precision.5IoU": np.round(self.rfp(0.5), 3),
            "recall_at_.8precision.5IoU": np.round(self.rfp(0.8), 3),
            "recall_at_.9precision.5IoU": np.round(self.rfp(0.9), 3)
        }
