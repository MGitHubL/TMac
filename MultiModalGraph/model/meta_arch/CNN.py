from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import focal_loss
from sklearn import metrics
from torch import nn
from torch.nn.init import xavier_uniform_

# from MultiModalGraph.model.GNN_models import GATConv
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GlobalAttention,
    LayerNorm,
    Linear,
    SAGEConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)
# from torch_geometric.nn.glob.gmt import GraphMultisetTransformer

from MultiModalGraph.configs.config import configurable
from MultiModalGraph.model.build import META_ARCH_REGISTRY

# from torch_geometric.nn import HeteroConv
from MultiModalGraph.model.utils import HeteroConv
from MultiModalGraph.structures.instances import Instances
from MultiModalGraph.utils.events import get_event_storage

__all__ = ["ResNet1D"]


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.GELU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.GELU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


@META_ARCH_REGISTRY.register()
class ResNet1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    @configurable
    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        loss=None,
        device="cuda",
        use_bn=False,
        use_do=False,
        verbose=False,
    ):
        super(ResNet1D, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.loss = loss
        self.device = device
        self.n_classes = n_classes

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        self.start = nn.Linear(1024, 128)
        # first block
        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.GELU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(
                    base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
                )
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.GELU()  # (inplace=True)
        self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        self.softmax = nn.Softmax(dim=1)

    @classmethod
    def from_config(cls, cfg):

        return {
            "in_channels": 40,  # cfg.MODEL.HIDDEN_CHANNELS,
            "n_classes": cfg.MODEL.OUT_DIM,
            "base_filters": 256,  # cfg.MODEL.BASE_FILTERS,
            "kernel_size": 16,  # cfg.MODEL.KERNEL_SIZE,
            "stride": 5,  # cfg.MODEL.STRIDE,
            "groups": 1,  # 32 cfg.MODEL.GROUPS,
            "n_block": 8,  # 48 cfg.MODEL.N_BLOCK,
            # "num_layers": cfg.MODEL.NUM_LAYERS,
            "device": cfg.DEVICE,
            "loss": cfg.TRAINING.LOSS,
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
        out = batched_x["video"]
        out = torch.cat(
            [
                out[batches["video"] == i].unsqueeze(0)
                for i in torch.unique(batches["video"])
            ]
        )

        # out = self.start(out.view(-1, out.shape[-1])).view(out.shape[0], out.shape[1], 128)

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print("final pooling", out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print("dense", out.shape)
        # out = torch.softmax(out, dim=1)
        if self.verbose:
            print("softmax", out.shape)

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
        out = batched_x["video"]
        out = torch.cat(
            [
                out[batches["video"] == i].unsqueeze(0)
                for i in torch.unique(batches["video"])
            ]
        )

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print("final pooling", out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print("dense", out.shape)
        # out = torch.softmax(out, dim=1)
        if self.verbose:
            print("softmax", out.shape)

        return out.detach().cpu()
