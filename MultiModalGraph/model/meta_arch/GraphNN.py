from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import focal_loss
from sklearn import metrics
from torch import nn
from torch.nn.init import xavier_uniform_
from torch_geometric.data import HeteroData

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import signal

from MultiModalGraph.model.utils import pairwise_distances, num_of_graphs
from sklearn.metrics.pairwise import cosine_similarity

# from MultiModalGraph.model.GNN_models import GATConv
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GlobalAttention,
    LayerNorm,
    GraphNorm,
    PairNorm,
    InstanceNorm,
    Linear,
    SAGEConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)

# from torch_geometric.nn.glob.gmt import GraphMultisetTransformer

from MultiModalGraph.configs.config import configurable
from MultiModalGraph.model.build import META_ARCH_REGISTRY
from MultiModalGraph.model.meta_arch.GATSelfConv import GATSelfConv

# from torch_geometric.nn import HeteroConv
from MultiModalGraph.model.utils import HeteroConv
from MultiModalGraph.structures.instances import Instances
from MultiModalGraph.utils.events import get_event_storage
from MultiModalGraph.utils.utils import atten_dive_score

__all__ = ["TemGNN"]


@META_ARCH_REGISTRY.register()
class TemGNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        hidden_channels: int = 512,
        out_dim: int = 128,
        num_layers: int = 4,
        device: str = "cpu",
        normalize: bool = False,
        self_loops: bool = False,
        vis_period: int = 0,
        loss: str = "CrossEntropyLoss",
        cfg: Optional[Dict] = None,
    ):
        """
        Args:
            hidden_channels (int): number of hidden channels
            out_dim (int): output dimension (number of classes)
            num_layers (int): number of heterogenous layers
            device (str): device to use
            normalize (bool): whether to normalize the node features
            self_loops (bool): whether to add self-loops to the graph
            vis_period: the period to run visualization. Set to 0 to disable.
            loss (str): loss function to use
        """
        super().__init__()

        self.vis_period = vis_period
        self.loss = loss
        self.device = device
        self.out_dim = out_dim
        self.fusion_layers = np.array(cfg.GRAPH.FUSION_LAYERS) + num_layers
        self.l2_reg = True

        # define conv layer for each edge
        self.convs = torch.nn.ModuleList()
        self.Norm_layers = []
        for i in range(num_layers):
            if i in self.fusion_layers:
                audio_conv = video_conv = GCNConv(
                        -1,
                        hidden_channels,
                        normalize=normalize,
                        add_self_loops=self_loops,
                    )
                # audio_conv = video_conv = TransformerConv(-1, hidden_channels)
            else:
                audio_conv = GCNConv(
                        -1,
                        hidden_channels,
                        normalize=normalize,
                        add_self_loops=self_loops,
                    )
                video_conv = GCNConv(
                        -1,
                        hidden_channels,
                        normalize=normalize,
                        add_self_loops=self_loops,
                    )
            conv = HeteroConv(
                {
                    ("video", "video-video", "video"): video_conv,
                    ("audio", "audio-audio", "audio"): audio_conv,
                    ("video", "video-audio", "audio"): GATSelfConv(
                        (-1, -1),
                        hidden_channels,
                        add_self_loops=self_loops,
                        heads=1,
                        concat=False,
                    ),
                },
                aggr="sum",
            )

            self.convs.append(conv)
            self.Norm_layers.append(
                {
                    "audio": LayerNorm(hidden_channels).cuda(),
                    "video": LayerNorm(hidden_channels).cuda(),
                }
            )

        # define output layer
        feat_dim = (
            1 * hidden_channels
        )
        self.lin = nn.Sequential(
            nn.Linear(feat_dim, out_dim)
        )

        # global attention pooling
        self.att_w_audio = nn.Linear(hidden_channels, 1, bias=False)
        xavier_uniform_(self.att_w_audio.weight)
        self.map_audio = nn.Identity()  # nn.Linear(hidden_channels, hidden_channels)
        self.att_w_video = nn.Linear(hidden_channels, 1, bias=False)
        xavier_uniform_(self.att_w_video.weight)
        self.map_video = nn.Identity()  # nn.Linear(hidden_channels, hidden_channels)

        self.graph_read_out_audio = GlobalAttention(
            gate_nn=self.att_w_audio, nn=self.map_audio
        )
        self.graph_read_out_video = GlobalAttention(
            gate_nn=self.att_w_video, nn=self.map_video
        )
        # attention weights for each sample
        self.att_w_audio_sample = None
        self.att_w_video_sample = None

    def l2_regularization(self):
        """
        Compute the l2 norm of the model parameters.
        """
        l2_reg = torch.tensor(0.0).to(self.device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return l2_reg

    @classmethod
    def from_config(cls, cfg):

        return {
            "hidden_channels": cfg.MODEL.HIDDEN_CHANNELS,
            "out_dim": cfg.MODEL.OUT_DIM,
            "num_layers": cfg.MODEL.NUM_LAYERS,
            "device": cfg.DEVICE,
            "normalize": cfg.GRAPH.NORMALIZE,
            "self_loops": cfg.GRAPH.SELF_LOOPS,
            "vis_period": cfg.VIS_PERIOD,
            "loss": cfg.TRAINING.LOSS,
            "cfg": cfg,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        '''
        batched_inputs: 'graph', 'label', 'numerical_label', 'video_add', 'span_over_time_audio', 'audio_dilation',
                        'span_over_time_video', 'video_dilation'
        batched_inputs["graph"]: video={x=[1280, 1024],batch=[1280],ptr=[33]},
                                 audio={x=[3232, 128],batch=[3232],ptr=[33]},
                                (video, video-video, video)={ edge_index=[2,10240]},
                                (audio, audio-audio, audio)={ edge_index=[2,38784]},
                                (video, video-audio, audio)={ edge_index=[2,3808]}
        batched_x: 'video'[1280,1024], 'audio'[3232,128]
        batched_edges: ('video','video-video','video')[2,10240],
                       ('audio','audio-audio','audio')[2,38784],
                       ('video','video-audio','audio')[2,3808]
        batches: 'video'[1280], 'audio'[3232]
        '''
        batched_inputs["graph"] = batched_inputs["graph"].to(self.device)

        batched_x, batched_edges, batch_weights, batches = (
            batched_inputs["graph"].x_dict,
            batched_inputs["graph"].edge_index_dict,
            batched_inputs["graph"].edge_attr_dict,
            batched_inputs["graph"].batch_dict,
        )

        x_dict = batched_x
        # computing MAD metric
        # MAD_audio_begin = np.mean(1 - cosine_similarity(x_dict["audio"].detach().cpu().numpy(),
        #                                               x_dict["audio"].detach().cpu().numpy()))
        # MAD_video_begin = np.mean(1 - cosine_similarity(x_dict["video"].detach().cpu().numpy(),
        #                                               x_dict["video"].detach().cpu().numpy()))

        edge_index_dict = batched_edges
        edge_weights_dict = batch_weights
        for key in edge_weights_dict:
            edge_weights_dict[key] = edge_weights_dict[key].view(1, -1).to(torch.float32)

        for i, (conv, norm_layer) in enumerate(zip(self.convs, self.Norm_layers)):
            x_dict, _ = conv(x_dict, edge_index_dict, edge_weights_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.1) for key, x in x_dict.items()}
            x_dict = {
                key: norm_layer[key](x, batches[key]) for key, x in x_dict.items()
            }

        graph_embed = self.graph_read_out_audio(x_dict["audio"], batches["audio"])  # [b,d]
        # x_dict["audio"]:[3232,512], batches["audio"]:[3232]

        # pred = self.lin(torch.sigmoid(graph_embed))
        pred = self.lin(graph_embed)

        criterion = torch.nn.CrossEntropyLoss()
        classification_loss = {}
        if self.loss == "CrossEntropyLoss":
            classification_loss["cls_loss"] = criterion(
                pred, batched_inputs["numerical_label"].to(self.device)
            )
        elif self.loss == "FocalLoss":
            classification_loss["cls_loss"] = focal_loss(
                pred,
                batched_inputs["numerical_label"].to(self.device),
                alpha=0.5,
                gamma=2.0,
                reduction="mean",
            )
        target = (
            F.one_hot(batched_inputs["numerical_label"], self.out_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        # classification_loss['cls_loss'] = F.binary_cross_entropy(pred, target)
        if self.l2_reg:
            l2_loss = +0.5 * self.l2_regularization()

        # computing average precision
        try:
            average_precision = metrics.average_precision_score(
                target.cpu().float().numpy(),
                pred.detach().cpu().float().numpy(),
                average=None,
            )
        except ValueError:
            average_precision = np.array([np.nan] * self.out_dim)

        acc = (
            torch.max(pred, 1)[1].cpu() == batched_inputs["numerical_label"]
        ).sum().item() / batched_inputs["numerical_label"].shape[0]

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs)

        losses = {}
        losses.update(classification_loss)

        if self.l2_reg:
            losses.update({"l2_loss": l2_loss})

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

        batched_x, batched_edges, batch_weights, batches = (
            batched_inputs["graph"].x_dict,
            batched_inputs["graph"].edge_index_dict,
            batched_inputs["graph"].edge_attr_dict,
            batched_inputs["graph"].batch_dict,
        )

        x_dict = batched_x
        # computing MAD metric
        # MAD_audio_begin = np.mean(1 - cosine_similarity(x_dict["audio"].detach().cpu().numpy(),
        #                                               x_dict["audio"].detach().cpu().numpy()))
        # MAD_video_begin = np.mean(1 - cosine_similarity(x_dict["video"].detach().cpu().numpy(),
        #                                               x_dict["video"].detach().cpu().numpy()))

        edge_index_dict = batched_edges
        edge_weights_dict = batch_weights

        for key in edge_weights_dict:
            edge_weights_dict[key] = edge_weights_dict[key].view(1, -1).to(torch.float32)

        for conv, norm_layer in zip(self.convs, self.Norm_layers):
            x_dict, _ = conv(x_dict, edge_index_dict, edge_weights_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            # x_dict = {key: F.dropout(x, p=0.1) for key, x in x_dict.items()}
            x_dict = {
                key: norm_layer[key](x, batches[key]) for key, x in x_dict.items()
            }

        from torch_geometric.utils import softmax
        self.att_w_audio_sample = softmax(
            self.graph_read_out_audio.gate_nn(x_dict["audio"]).view(-1, 1), batches["audio"],
            num_nodes=batches["audio"][-1]+1
        ).view(1, -1)
        self.att_w_audio_sample = torch.cat(
            [
                self.att_w_audio_sample[:, batches["audio"] == i]
                for i in torch.unique(batches["audio"])
            ]
        )
        self.att_w_video_sample = softmax(
            self.graph_read_out_video.gate_nn(x_dict["video"]).view(-1, 1), batches["video"],
            num_nodes=batches["video"][-1]+1
        ).view(1, -1)
        self.att_w_video_sample = torch.cat(
            [
                self.att_w_video_sample[:, batches["video"] == i]
                for i in torch.unique(batches["video"])
            ]
        )

        # for i, (attn_a, attn_v) in enumerate(zip(self.att_w_audio_sample, self.att_w_video_sample)):
        #     if '19730' in batched_inputs['video_add'][i]:
        #         if torch.where(attn_v > 0.03)[0].shape[0] > 0 or torch.where(attn_a > 0.02)[0].shape[0] > 0:
        #             # print(i)
        #             plt.figure()
        #             v = self.att_w_video_sample[i].detach().cpu().numpy()
        #             v = (v - np.min(v)) / (np.max(v) - np.min(v))
        #             a = self.att_w_audio_sample[i].detach().cpu().numpy()
        #             a = (a - np.min(a)) / (np.max(a) - np.min(a))
        #             plt.plot(signal.resample(savgol_filter(v, 39, 3), 101))
        #             plt.plot(savgol_filter(a, 21, 3))
        #             plt.legend(["video", "audio"])
        #             plt.xlabel("Graph Nodes")
        #             plt.ylabel("Attention Weight")
        #             plt.show()
        #             plt.close()

        # graph_embed = self.graph_read_out_audio(x_dict["audio"], batches["audio"]) + \
        #               self.graph_read_out_video(x_dict["video"], batches["video"])
        # graph_embed = torch.cat([self.graph_read_out_audio(x_dict["audio"], batches["audio"]),
        #               self.graph_read_out_video(x_dict["video"], batches["video"])], dim=1)
        graph_embed = self.graph_read_out_audio(x_dict["audio"], batches["audio"])

        # pred = self.lin(torch.sigmoid(graph_embed))
        pred = self.lin(graph_embed)

        return pred.detach().cpu()
