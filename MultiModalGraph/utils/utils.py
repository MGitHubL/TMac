import logging
import os
import pickle
import pprint
import sys
from collections.abc import Mapping

import cv2
import numpy as np
import psutil
import torch
from fvcore.common.checkpoint import Checkpointer
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

import MultiModalGraph.utils.comm as comm
from MultiModalGraph.utils.file_io import PathManager

__all__ = [
    "print_mem",
    "print_all_mem",
    "sizeof_fmt",
    "load_model",
    "neq_load_customized",
    "video_reader",
    "DetectionCheckpointer",
    "flatten_results_dict",
    "verify_results",
]


def print_mem(p):
    rss = p.memory_info().rss
    print(f"[{p.pid}] memory usage: {rss / 1e6:0.3} MB")


def print_all_mem():
    p = psutil.Process()
    procs = [p] + p.children(recursive=True)
    for p in procs:
        print_mem(p)


def sizeof_fmt(num, suffix="B"):
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def load_model(model, load_path):
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    state_dict = checkpoint["state_dict"]

    try:
        model.load_state_dict(state_dict)
    except:
        neq_load_customized(model, state_dict, verbose=False)

    return model


def neq_load_customized(model, pretrained_dict, verbose=False):
    """ load pre-trained model in a not-equal way,
    when new model has been partially modified """

    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print("\n=======Check Weights Loading======")
        print("Weights not used from pretrained file:")
        for k, v in pretrained_dict.items():  # only select the RGB subnetwork
            if "encoder_k.0." in k:
                # if k in model_dict:
                k = k.replace("encoder_k.0.", "")
                tmp[k] = v
            else:
                print(k)
        print("---------------------------")
        print("Weights not loaded into new model:")
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print("===================================\n")
    else:
        for k, v in pretrained_dict.items():  # only select the RGB subnetwork
            if "encoder_k.0." in k:
                # if k in model_dict:
                k = k.replace("encoder_k.0.", "")
                tmp[k] = v
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def imshow(img, name="image"):
    try:
        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow(name, img)

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
    finally:
        # closing all open windows
        cv2.destroyAllWindows()


def video_reader(video_path, transform=None, transform_cuda=None):

    img_array = []
    try:
        cap = cv2.VideoCapture(video_path)
        success, img = cap.read()

        while success:
            img_array.append(Image.fromarray(img))
            # read next frame
            success, img = cap.read()
        cap.release()
        if transform:
            img_array = transform(img_array)
        img_array = torch.stack(img_array, 1).unsqueeze(0)
        if transform_cuda:
            img_array = transform_cuda(img_array)
        return img_array
    except:
        cap.release()
        print(f"failed processing {video_path}")
        return torch.zeros(1, 3, 20, 128, 128)


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager

    def load(self, path, *args, **kwargs):
        need_sync = False

        if path and isinstance(self.model, DistributedDataParallel):
            logger = logging.getLogger(__name__)
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {
                    "model": data,
                    "__author__": "Caffe2",
                    "matching_heuristics": True,
                }
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {
                "model": model_state,
                "__author__": "pycls",
                "matching_heuristics": True,
            }

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        # if checkpoint.get("matching_heuristics", False):
        #     self._convert_ndarray_to_tensor(checkpoint["model"])
        #     # convert weights by name-matching heuristics
        #     checkpoint["model"] = align_and_update_state_dicts(
        #         self.model.state_dict(),
        #         checkpoint["model"],
        #         c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
        #     )
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task].get(metric, None)
        if actual is None:
            ok = False
            continue
        if not np.isfinite(actual):
            ok = False
            continue
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def atten_dive_score(attn_vec, edge_index=None):
    """
    Args:
        attn_vec (torch.Tensor): (1, N)
        edge_index (torch.Tensor): (2, E)
    """
    if edge_index is None:
        return torch.std(attn_vec, dim=1).mean()
    else:
        attn_w = [
            attn_vec[edge_index[1] == idx]
            for idx in torch.unique(edge_index[1])
            if len(attn_vec[edge_index[1] == idx]) != 1
        ]
        return torch.stack(
            [torch.std(attn_w[idx]).mean() for idx in range(len(attn_w))]
        ).mean()

