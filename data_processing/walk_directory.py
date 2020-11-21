import os
import os.path as osp
import yaml
import tqdm
import fnmatch
import warnings
from collections import deque

DATASETS_ROOT = ""
MARKUP_ROOT = ""

IMAGE_EXTS = [".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff"]
VIDEO_EXTS = [".avi", ".mp4", ".m4v", ".flv", ".MOV"]

DEFAULT_BLACK_LIST = ["*.ipynb_checkpoints*", "*__pycache__*"]


def parse_config(config):
    """
    Parse config argument.
    
    Args:
    config: None, dict or str, path to .yaml config
    
    Returns:
    white_list, black_list, strides,
        where strides - dict, {"pattern": [count, stride]}
    """
    if config is None:
        config = {"white_list": [""]}

    if isinstance(config, str):
        with open(config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

    white_list = []
    strides = {}
    for item in config["white_list"]:
        if isinstance(item, str):
            white_list.append(item)
        elif isinstance(item, list):
            white_list.append(item[0])
            if (len(item) > 1) and ("stride" in item[1]):
                strides[item[0]] = [0, item[1]["stride"]]
    black_list = config.get("black_list", [])
    black_list.extend(DEFAULT_BLACK_LIST)

    if len(white_list) == 0:
        white_list = [""]

    if sum([(("?" in s) or ("*" in s)) for s in white_list]) > 20:
        warnings.warn("White list has over 20 regular expressions. Could be slow.")
    if len(black_list) > 10:
        warnings.warn("Black list has over 10 regular expressions. Could be slow.")

    return white_list, black_list, strides


def walk_directory(config, mode="images", start_root=None, verbose=False):
    """
    Walk through a given folder and get files that match patterns in white list.
    
    Args:
    config    : None, dict or str, path to .yaml config
        example of .yaml config file:
        white_list:
          - balanced_dataset/11
          -
            - ITSK-LAB
            - stride: 10
          - balanced_dataset/2/GPN_180319_6103
    mode      : str, one of `images`, `markup` or `videos`, what sort of files to yield
    start_root: str, path to where to start searching from (if None, is set automatically)
    verbose   : bool, whether to print tqdm progress bar
          
    Returns:
    a generator object, that yields absolutes paths of an files that match patterns in config
    """
    if mode == "images":
        if start_root is None:
            start_root = DATASETS_ROOT
        good_exts = IMAGE_EXTS.copy()
    elif mode == "markup":
        if start_root is None:
            start_root = MARKUP_ROOT
        good_exts = [ext + ".json" for ext in IMAGE_EXTS]
    elif mode == "videos":
        if start_root is None:
            start_root = DATASETS_ROOT
        good_exts = VIDEO_EXTS.copy()
    else:
        raise ValueError("mode must be one of `images`, `markup` or `videos`.")

    white_list, black_list, strides = parse_config(config)
    # strides - dict
    # {"pattern": [count, stride], ...}

    def extension_check(path):
        if not any([path.endswith(ext) for ext in good_exts]):
            return False
        return True

    def blacklist_check(path):
        if any(
            [
                fnmatch.fnmatch(path, osp.join(start_root, pattern))
                for pattern in black_list
            ]
        ):
            return False
        return True

    def stride_check(entry):
        if entry in strides.keys():
            st = strides[entry]
            st[0] += 1
            if (st[0] - 1) % st[1] != 0:
                return False
        return True

    if verbose:
        gen = tqdm.tqdm(
            white_list, desc="walking through " + start_root, total=len(white_list)
        )
    else:
        gen = white_list

    for entry in gen:

        qm_pos = entry.find("?")
        ast_pos = entry.find("*")
        pattern_pos = min(qm_pos, ast_pos)
        if (qm_pos == -1) ^ (ast_pos == -1):
            pattern_pos = max(qm_pos, ast_pos)
        # pattern_pos - position of first "?" or "*"
        slash_pos = entry.find("/")

        if max(qm_pos, ast_pos) >= 0:
            # regular expression case

            start_dir = start_root
            if (slash_pos < pattern_pos) and (slash_pos >= 0):
                start_dir = osp.join(start_root, entry[:slash_pos])
            regexp = entry[slash_pos + 1 :]
            # start_dir - constant part of regular expression

            # Note: we assume that a regexp can match directory names only at one depth level
            # Example:
            #   regexp = "balanced*"
            #   "balanced_dataset" -> match
            #   "balanced_dataset/2" -> no match
            #
            # This is done to prevent duplicate images from appearing in dataset

            Q = deque()

            for d in next(os.walk(start_dir))[1]:
                Q.append(osp.join(start_dir, d))

            while Q:
                matched = [
                    el
                    for el in Q
                    if fnmatch.fnmatch(osp.relpath(el, start_dir), regexp)
                ]
                if len(matched) > 0:
                    break

                n = len(Q)
                for i in range(n):
                    el = Q.popleft()
                    for d in next(os.walk(el))[1]:
                        Q.append(osp.join(el, d))

            for el in matched:
                # all directories that match regexp

                for r, ds, fs in os.walk(el):
                    # yielding all files in them

                    fs.sort()
                    for f in fs:

                        if not extension_check(f):
                            continue
                        file_path = osp.abspath(osp.join(r, f))
                        if not blacklist_check(file_path):
                            continue
                        if not stride_check(entry):
                            continue

                        yield file_path

            if len(matched) == 0:
                # no directories matched => matching files

                for r, ds, fs in os.walk(start_dir):

                    fs.sort()
                    for f in fs:

                        if not extension_check(f):
                            continue
                        file_path = osp.abspath(osp.join(r, f))
                        if not fnmatch.fnmatch(
                            osp.relpath(file_path, start_dir), regexp
                        ):
                            continue
                        if not blacklist_check(file_path):
                            continue

                        yield file_path

        else:
            # path case
            full_path = osp.join(start_root, entry)

            if not any([entry.endswith(ext) for ext in good_exts]):
                # possibly a directory

                if not osp.isdir(full_path):
                    continue

                for r, ds, fs in os.walk(osp.join(start_root, entry)):
                    fs.sort()

                    for f in fs:

                        if not extension_check(f):
                            continue
                        file_path = osp.abspath(osp.join(r, f))
                        if not blacklist_check(file_path):
                            continue
                        if not stride_check(entry):
                            continue

                        yield file_path

            else:
                # possibly a file

                if not osp.exists(full_path):
                    continue
                if not blacklist_check(full_path):
                    continue

                yield full_path
