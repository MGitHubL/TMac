from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kornia.losses import focal_loss
from sklearn import metrics

from MultiModalGraph.configs.config import configurable
from MultiModalGraph.model.build import META_ARCH_REGISTRY

# from torch_geometric.nn import HeteroConv
from MultiModalGraph.model.utils import HeteroConv
from MultiModalGraph.structures.instances import Instances
from MultiModalGraph.utils.events import get_event_storage

__all__ = ["LSTM1D"]


@META_ARCH_REGISTRY.register()
class LSTM1D(nn.Module):
    @configurable
    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, tagset_size, loss, device="cuda"
    ):
        super(LSTM1D, self).__init__()
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.n_classes = tagset_size
        self.vocab_size = vocab_size

        self.word_embeddings = nn.Linear(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.device = device

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embedding_dim": 128,
            "tagset_size": cfg.MODEL.OUT_DIM,
            "hidden_dim": 256,
            "vocab_size": 128,
            "loss": cfg.TRAINING.LOSS,
            "device": cfg.DEVICE,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        batched_inputs["graph"] = batched_inputs["graph"].to(self.device)

        batched_x, batched_edges, batches = (
            batched_inputs["graph"].x_dict,
            batched_inputs["graph"].edge_index_dict,
            batched_inputs["graph"].batch_dict,
        )
        input = batched_x["audio"]
        input = torch.cat(
            [
                input[batches["audio"] == i].unsqueeze(0)
                for i in torch.unique(batches["audio"])
            ]
        )

        embeds = input.view(input.shape[0], input.shape[1], self.vocab_size)
        lstm_out, _ = self.lstm(embeds.view(len(input), embeds.shape[1], -1))
        avg = torch.mean(lstm_out, dim=1)
        tag_space = self.hidden2tag(avg.view(len(input), -1))
        out = F.log_softmax(tag_space, dim=1)

        criterion = torch.nn.CrossEntropyLoss()
        classification_loss = {}
        if self.loss == "CrossEntropyLoss":
            classification_loss["cls_loss"] = criterion(
                out, batched_inputs["numerical_label"].to(self.device)
            )
        elif self.loss == "FocalLoss":
            classification_loss["cls_loss"] = focal_loss(
                out,
                batched_inputs["numerical_label"].to(self.device),
                alpha=0.5,
                gamma=2.0,
                reduction="mean",
            )
        target = (
            F.one_hot(batched_inputs["numerical_label"], self.n_classes)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        # classification_loss['cls_loss'] = F.binary_cross_entropy(pred, target)

        # computing average precision
        try:
            average_precision = metrics.average_precision_score(
                target.cpu().float().numpy(),
                out.detach().cpu().float().numpy(),
                average=None,
            )
        except ValueError:
            average_precision = np.array([np.nan] * self.n_classes)
        # try:
        #     roc = metrics.roc_auc_score(batched_inputs['numerical_label'].numpy(), pred.softmax(1).numpy(),
        #     multi_class='ovr' )
        # except ValueError:
        #     roc = np.array([np.nan] * 527)

        # computing accuracy
        acc = (
            torch.max(out, 1)[1].cpu() == batched_inputs["numerical_label"]
        ).sum().item() / batched_inputs["numerical_label"].shape[0]

        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs)

        losses = {}
        losses.update(classification_loss)
        # losses.update(proposal_losses)
        metric = {}
        metric["mAP"] = torch.from_numpy(
            np.asarray(
                average_precision[~np.isnan(average_precision)].sum()
                / average_precision.shape[0]
            )
        ).to(self.device)
        metric["accuracy"] = torch.from_numpy(np.asarray(acc)).to(self.device)
        return losses, metric

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        batched_inputs["graph"] = batched_inputs["graph"].to(self.device)

        batched_x, batched_edges, batches = (
            batched_inputs["graph"].x_dict,
            batched_inputs["graph"].edge_index_dict,
            batched_inputs["graph"].batch_dict,
        )
        input = batched_x["audio"]
        input = torch.cat(
            [
                input[batches["audio"] == i].unsqueeze(0)
                for i in torch.unique(batches["audio"])
            ]
        )

        embeds = input.view(input.shape[0], input.shape[1], self.vocab_size)
        lstm_out, _ = self.lstm(embeds.view(len(input), embeds.shape[1], -1))
        avg = torch.mean(lstm_out, dim=1)
        tag_space = self.hidden2tag(avg.view(len(input), -1))
        out = F.log_softmax(tag_space, dim=1)

        return out.detach().cpu()
