import os
import os.path as osp
import json
import tqdm
import numpy as np
import pandas as pd
from .nms import filter_phantoms


def numpy_json_converter(obj):
    """
    Allows to dump numpy arrays as json.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def csv_to_json_markup(csv_path, markup_dir):
    """
    Converts a csv markup file to gmc-json format.
    
    Arguments:
    csv_path  : str, path to csv-file with bboxes and labels
    markup_dir: str, path to directory where to save markup
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("image", ignore_index=True)
    images = list(df["image"].unique())

    c = 0
    for img in tqdm.tqdm(images):

        json_path = os.path.join(markup_dir, img + ".json")
        d = {"objects": []}
        bboxes = []
        labels = []

        while (c < df.shape[0]) and (df.iloc[c]["image"] == img):
            entry = df.iloc[c]
            obj = {}
            obj["data"] = [entry["left"], entry["top"], entry["width"], entry["height"]]
            obj["tags"] = ["h", entry["label"]]
            obj["type"] = "rect"
            d["objects"].append(obj)
            c += 1

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as fd:
            json.dump(d, fd, indent=4, default=numpy_json_converter)


def convert_gmc_to_bbox(gmc_markup_path, tag_category=["h"], track_ids=False):
    """
    Convert gmc markup into list of lists of x,y,w,h.
    
    Arguments:
    gmc_markup_path: str, path to json-file
    tag_category   : list, bboxes with these tags will be loaded
    track_ids      : bool, whether there are track ids in markup    
    
    Returns:
    list of bboxes, each bbox is a list
        [x_min, y_min, width, height(, track_id)]
    """
    with open(gmc_markup_path, "r") as f:
        gmc_json = json.load(f)
    bboxes = []
    if len(gmc_json) > 0:
        for object_ in gmc_json["objects"]:
            if "tags" in object_ and object_["type"] == "rect":
                for tag_item in tag_category:
                    if tag_item in object_["tags"]:
                        bboxes.append(object_["data"])
                        if track_ids:
                            # adding track ids as well (if they exist)
                            # checking to see if there's a number tag
                            track_id = -1
                            for t in object_["tags"]:
                                try:
                                    track_id = int(t)
                                except:
                                    pass
                            bboxes[-1].extend([track_id])
                        break
    return bboxes


def load_gmc(markup_path, tags=["h"], n_bboxes=None, track_ids=False):
    """
    Load markup for an image.
    
    Arguments:
    markup_path: str, path to json-file with markup
    tags       : list, bboxes with these tags will be loaded
    n_bboxes   : int, the list of bboxes is extended to this length
    track_ids  : bool, whether there are track ids in markup 
    
    Note: classes are labelled in the same order that they appear in `tags` array
    
    Returns:
    np.array of shape (n, 5) with bboxes:
        [x_min, y_min, width, height, label] (if track_ids==False)
    """
    all_bboxes = []
    for tag_idx, tag in enumerate(tags):
        class_bboxes = convert_gmc_to_bbox(
            markup_path, tag_category=[tag], track_ids=track_ids
        )
        if len(class_bboxes) == 0:
            continue
        class_bboxes = np.array(class_bboxes).astype(np.float32)
        class_id_vec = np.zeros_like(class_bboxes[:, 0:1]) + tag_idx
        class_bboxes = np.concatenate(
            [class_bboxes[:, 0:4], class_id_vec, class_bboxes[:, 4:5]], axis=-1
        )
        all_bboxes.append(class_bboxes)
    if len(all_bboxes) == 0:
        bboxes_np = np.zeros((0, 5 + int(track_ids)))
    else:
        bboxes_np = np.concatenate(all_bboxes, axis=0)
    if n_bboxes is not None:
        if n_bboxes < bboxes_np.shape[0]:
            raise ValueError(
                f"Image has {bboxes_np.shape[0]} bboxes. Increase `n_bboxes`."
            )
        placeholder = np.zeros((n_bboxes,) + bboxes_np.shape[1:], dtype=np.float32)
        if bboxes_np.shape[0] > 0:
            placeholder[: bboxes_np.shape[0]] = bboxes_np
        return placeholder
    else:
        return bboxes_np


def dump(json_path, bboxes):
    with open(json_path, "w") as fd:
        json.dump(bboxes.tolist(), fd)


def serialize(img_paths, bboxes, img_path_prefix_to_cut, output_dir):
    """
    Serialize bboxes to file(s)

    Parameters
    ----------
    img_paths : iterable
      paths to images, on which the bboxes are detected
      Used to create output path
    
    bboxes : numpy.ndarray
      bboxes detected in the images
      NOTE: mb with phantom bboxes
    
    img_path_prefix_to_cut : str
      a common subpath of img_paths
      in the output_dir, file osp.relpath(img_path, img_path_prefix_to_cut) + '.json'
      would be created

    output_dir : str      

    """
    for img_path, img_bboxes in zip(img_paths, bboxes):
        try:
            img_path = img_path.decode("utf-8")
        except:
            pass
        img_name = osp.relpath(img_path, img_path_prefix_to_cut)
        out_path = osp.join(output_dir, img_name + ".json")
        os.makedirs(osp.dirname(out_path), exist_ok=True)

        valid_bboxes = filter_phantoms(img_bboxes)
        dump(out_path, valid_bboxes)



def load_bbox(markup_path, n_bboxes=None):
    with open(markup_path, "r") as fd:
        bboxes = json.load(fd)
    if len(bboxes) == 0:
        bboxes_np = np.zeros((0, 6))
    else:
        bboxes_np = np.array(bboxes)

    if n_bboxes is not None:
        placeholder = np.zeros((n_bboxes,) + bboxes_np.shape[1:], dtype=np.float32)
        if bboxes_np.shape[0] > 0:
            placeholder[: bboxes_np.shape[0]] = bboxes_np
        return placeholder
    else:
        return bboxes_np
