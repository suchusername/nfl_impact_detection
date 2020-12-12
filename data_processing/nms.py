import os.path as osp
import numpy as np


def nms_impl(detections, threshold):
    filtered = []
    for d in detections:
        detection_is_different = True
        for f in filtered:
            if IoU(d, f) > threshold:
                detection_is_different = False
                break

        if detection_is_different:
            filtered.append(d)
    return np.array(filtered)

def soft_nms_impl(detections, threshold):
    filtered = []
    S = np.array([i[5] for i in detections])
    while len(detections) != 0:
        max_index = np.argmax(S)
        M = detections[max_index]
        filtered.append(M)
        detections = np.delete(detections, max_index, 0)
        S = np.delete(S, max_index, 0)
        remove_index = []
        for i, d in enumerate(detections):
            iou = IoU(d, M)
            if iou > threshold:
                d[5] = d[5] * (1 - iou)
                S[i] = d[5]
            if d[5] < 0.15:
                remove_index.append(i)
        detections = np.delete(detections, remove_index, 0)
        S = np.delete(S, remove_index, 0)  
    return np.array(filtered)

def nms_one_img(detections, threshold, independent_classes):
    detections = np.array(detections)
    if independent_classes:
        try:
            class_ids = np.unique(detections[:, 4])
        except:
            print(detections)
        class_bboxes = [detections[detections[:, 4] == c_id] for c_id in class_ids]
        class_bboxes_nmsed = [nms_impl(bbx, threshold) for bbx in class_bboxes]
        if class_bboxes_nmsed == []:
            return np.array(class_bboxes_nmsed)
        class_bboxes_nmsed = np.concatenate(class_bboxes_nmsed)
        return class_bboxes_nmsed
    else:
        return nms_impl(detections, threshold)

def soft_nms_one_img(detections, threshold, independent_classes):
    if independent_classes:
        class_ids = np.unique(detections[:, 4])
        class_bboxes = [detections[detections[:, 4] == c_id] for c_id in class_ids]
        class_bboxes_nmsed = [soft_nms_impl(bbx, threshold) for bbx in class_bboxes]
        if class_bboxes_nmsed == []:
            return np.array(class_bboxes_nmsed)
        class_bboxes_nmsed = np.concatenate(class_bboxes_nmsed)
        return class_bboxes_nmsed
    else:
        return soft_nms_impl(detections, threshold)

def IoU(d, f):
    area0 = d[2] * d[3]
    left0 = d[0]
    top0 = d[1]
    right0 = left0 + d[2]
    bot0 = top0 + d[3]

    area1 = f[2] * f[3]
    left1 = f[0]
    top1 = f[1]
    right1 = left1 + f[2]
    bot1 = top1 + f[3]

    vertical_intersection = max(0, min(right0, right1) - max(left0, left1))
    horizontal_intersection = max(0, min(bot0, bot1) - max(top0, top1))
    intersection_area = vertical_intersection * horizontal_intersection

    if intersection_area == 0:
        return 0

    return intersection_area / (area0 + area1 - intersection_area)

def filter_phantoms(bboxes):
    real_detections = bboxes[np.sum(bboxes, axis=-1) > 0]
    return real_detections


def nms(detections, threshold, independent_classes=True, nms_type='nms'):
    """
    NMS as it is

    Parameters
    ----------
    detections : numpy.ndarray of shape [n,m,k]
      k >= 6 - batch of detected bounding boxes
      NOTE: with phantom bboxes
    threshold : float
      IoU threshold.
      threshold = 0 - all overlapping bboxes are merged
      threshold = 1 - only exactly matching bboxes are merged
    independent_classes : bool, default : True
      if true, classes are nms'ed independently (otherwise not)
    
    Returns
    -------
    bboxes : numpy.ndarray of shape [n,m,k]
      bboxes after nms
      NOTE: array shape is not changed, phantom bboxes are added
    """
    result = np.zeros_like(detections)
    for img_idx, img_detections in enumerate(detections):
        real_detections = filter_phantoms(img_detections)
        if len(real_detections) == 0:
            continue
        if nms_type == 'nms':
            filtered = nms_one_img(real_detections, threshold, 
                                   independent_classes)
        elif nms_type == 'soft_nms':
            filtered = soft_nms_one_img(real_detections, threshold, 
                                        independent_classes)
        result[img_idx, : len(filtered)] = filtered
    return result
