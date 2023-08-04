import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import focal_loss
from sklearn import metrics

from .evaluator import DatasetEvaluator


class AudioSetEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self, dataset_name, num_class, prefix="", loss="CrossEntropyLoss", classes_name=[], cfg=None
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        """
        self._logger = logging.getLogger(__name__)

        self.num_class = num_class
        self.loss = loss
        self.device = torch.device("cuda")
        self.prefix = prefix
        self.classes_names = classes_name
        self.cfg = cfg

    def reset(self):
        self._predictions = []
        self._labels = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        self._predictions.append(outputs)
        self._labels.append(inputs["numerical_label"])

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        predictions = torch.cat(self._predictions)
        labels = torch.cat(self._labels)
        scalar_pred_labels = torch.max(F.softmax(predictions, dim=1), 1)[1].cpu()
        scalar_true_labels = torch.cat(self._labels)

        # confusion matrix
        # from sklearn.metrics import confusion_matrix
        # import pandas as pd
        # import os
        # from MultiModalGraph.evaluation.utils import pp_matrix
        # import matplotlib.pyplot as plt
        # cm = metrics.confusion_matrix(scalar_true_labels, scalar_pred_labels)
        # df_cm = pd.DataFrame(cm, index=self.classes_names, columns=self.classes_names)
        # cmap = 'PuRd'
        # pp_matrix(df_cm, cmap=cmap, annot=True,
        #           fmt=".2f",
        #           fz=20,
        #           lw=0.5,
        #           cbar=False,
        #           figsize=[2*len(self.classes_names), 2*len(self.classes_names)],
        #           show_null_values=0,
        #           pred_val_axis="y",
        #           show_plot=False,
        #           )
        # plt.savefig(os.path.join(self.cfg.OUTPUT_DIR, f"{self.prefix}_confusion_matrix.png"), dpi=100)
        # # plt.savefig("confusion_matrix.png", dpi=300)
        # plt.close()


        if len(predictions) == 0:
            self._logger.warning(
                "[AudioSetEvaluator] Did not receive valid predictions."
            )
            return {}

        self._results = OrderedDict()
        criterion = torch.nn.CrossEntropyLoss()
        classification_loss = {}
        if self.loss == "CrossEntropyLoss":
            classification_loss["cls_loss_" + self.prefix] = criterion(
                predictions, labels
            )
        elif self.loss == "FocalLoss":
            classification_loss["cls_loss_" + self.prefix] = focal_loss(
                predictions, labels, alpha=0.5, gamma=2.0, reduction="mean"
            )  # criterion(predictions, labels)
        target = (
            F.one_hot(labels, self.num_class).type(torch.FloatTensor).to(self.device)
        )
        # classification_loss['cls'] = F.binary_cross_entropy(pred, target)

        # computing average precision
        try:
            average_precision = metrics.average_precision_score(
                target.cpu().float().numpy(),
                # predictions.detach().cpu().float().numpy(),
                F.softmax(predictions, dim=1).detach().cpu().float().numpy(),
                average=None,
            )
        except ValueError:
            average_precision = np.array([np.nan] * self.out_dim)
        try:
            roc = metrics.roc_auc_score(
                labels.numpy(), predictions.softmax(1).numpy(), multi_class="ovr"
            )
        except ValueError:
            roc = np.array([np.nan])

        # computing accuracy
        acc = (
            torch.max(predictions, 1)[1].cpu() == labels
        ).sum().item() / labels.shape[0]

        self._results["mAP_" + self.prefix] = (
            average_precision[~np.isnan(average_precision)].sum()
            / average_precision.shape[0]
        )
        self._results["accuracy_" + self.prefix] = acc
        self._results["loss_" + self.prefix] = classification_loss[
            "cls_loss_" + self.prefix
        ].item()
        self._results["roc_" + self.prefix] = roc
        return copy.deepcopy(self._results)
