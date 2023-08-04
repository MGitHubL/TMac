from __future__ import division, print_function

import os

# Ignore warnings
import warnings
from collections import defaultdict

import h5py
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from torchvision import transforms, utils
from statistics import mean

__all__ = ["AudioSetGraphDataset", "AudioSetGraphDataset_plain"]

warnings.filterwarnings("ignore")


class AudioSetGraphDataset(Dataset):
    """AudioSet graph dataset."""

    def __init__(
        self, path, graph_config, seed=0, transform=None, disired_classes="All"
    ):
        """
        Args:
            path (string): Path to the node attributes files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        f = h5py.File(path, "r")
        self.videos = f["video"][()]
        self.audios = f["audio"][()]
        self.labels = f["label"][()]
        self.vid_add = f["video_add"][()]
        self.aud_add = f["audio_add"][()]
        True_labels = f["unique_labels"][()]
        del f
        self.labels = np.array([label.decode("ascii") for label in self.labels])
        self.vid_add = np.array([add.decode("ascii") for add in self.vid_add])
        self.aud_add = np.array([add.decode("ascii") for add in self.aud_add])
        self.label_dict = defaultdict(int)
        ## filter data based on the classes
        if disired_classes != ["All"]:
            self.disired_idx = [
                idx for idx, label in enumerate(self.labels) if label in disired_classes
            ]
            self.videos = self.videos[self.disired_idx]
            self.audios = self.audios[self.disired_idx]
            self.labels = self.labels[self.disired_idx]
            self.vid_add = self.vid_add[self.disired_idx]
            self.aud_add = self.aud_add[self.disired_idx]
            for idx, label in enumerate(disired_classes):
                self.label_dict[label] = idx
        else:
            for idx, label in enumerate(True_labels):
                self.label_dict[label.decode("ascii")] = idx

        self.num_vid_nodes = self.videos[0].shape[0]
        self.num_aud_nodes = self.audios[0].shape[0]
        assert (
            graph_config.SPAN_OVER_TIME_AUDIO > 0
        ), "span_over_time_audio must be greater than 0"
        assert (
            graph_config.SPAN_OVER_TIME_VIDEO > 0
        ), "span_over_time_video must be greater than 0"
        assert (
            graph_config.SPAN_OVER_TIME_BETWEEN > 0
        ), "span_over_time_between must be greater than 0"
        self.span_over_time_audio = (
            graph_config.SPAN_OVER_TIME_AUDIO
        )  # np.random.randint(low=2, high=graph_config.SPAN_OVER_TIME_AUDIO + 1)
        self.span_over_time_video = (
            graph_config.SPAN_OVER_TIME_VIDEO
        )  # np.random.randint(low=2, high=graph_config.SPAN_OVER_TIME_VIDEO + 1)
        self.span_over_time_between = graph_config.SPAN_OVER_TIME_BETWEEN
        self.audio_dilation = graph_config.AUDIO_DILATION
        self.video_dilation = graph_config.VIDEO_DILATION
        self.dynamic = graph_config.DYNAMIC
        self.normalize = graph_config.NORMALIZE  # make attributes normalized
        self.self_loops = graph_config.SELF_LOOPS  # add self loops
        self.transform = transform

        # np.random.seed(seed)
        self.hop_length = np.random.randint(
            np.ceil(self.num_aud_nodes / self.num_vid_nodes)
        )
        # compute the residual audio nodes
        self.resd = max(
            int(self.num_aud_nodes - (self.num_vid_nodes - 1) * self.hop_length) - 2, 0
        )
        # set a residual list for considering extra hop
        self.hop_prob_list = np.random.randint(
            np.floor(self.resd / self.num_vid_nodes),
            np.ceil(self.resd / self.num_vid_nodes) + 1,
            self.num_vid_nodes,
        )
        # make sure that the first video node connects to the first audio node
        self.hop_prob_list[0] = 0
        # make sure that the last video node will reach the end of the audio
        if self.resd - self.hop_prob_list.sum() > 0:
            self.hop_prob_list_comp = np.array(
                [1] * (self.resd - self.hop_prob_list.sum())
                + [0] * (self.num_vid_nodes - (self.resd - self.hop_prob_list.sum()))
            )
            # shuffle the list
            np.random.shuffle(self.hop_prob_list_comp)

            self.hop_prob_list = self.hop_prob_list + self.hop_prob_list_comp

        # normalize the features np.corrcoef
        if self.normalize:
            # self.videos = torch.nn.functional.normalize(torch.from_numpy(self.videos))
            self.videos = torch.from_numpy(self.videos)

            B, N, C = self.audios.shape
            self.audios = F.normalize(
                torch.from_numpy(self.audios).view(B, -1), p=2, dim=1
            ).view(B, N, C)
            # self.audios = torch.nn.functional.normalize(torch.from_numpy(self.audios))
        else:
            self.videos = torch.from_numpy(self.videos)
            self.audios = torch.from_numpy(self.audios)

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # set the span over time for each modality
        span_over_time_audio = (
            np.random.randint(low=1, high=self.span_over_time_audio + 1)
            if self.dynamic
            else self.span_over_time_audio
        )
        span_over_time_video = (
            np.random.randint(low=1, high=self.span_over_time_video + 1)
            if self.dynamic
            else self.span_over_time_video
        )
        span_over_time_between = (
            np.random.randint(low=1, high=self.span_over_time_between + 1)
            if self.dynamic
            else self.span_over_time_between
        )
        video_dilation = (
            np.random.randint(low=1, high=self.video_dilation + 1)
            if self.dynamic
            else self.video_dilation
        )
        audio_dilation = (
            np.random.randint(low=1, high=self.audio_dilation + 1)
            if self.dynamic
            else self.audio_dilation
        )

        graph = HeteroData()

        # creating edges
        vid_sr_edges = []
        vid_dt_edges = []
        aud_sr_edges = []
        aud_dt_edges = []
        aud_vid_sr_edges = []
        aud_vid_dt_edges = []
        vid_dt_matrix = []
        aud_dt_matrix = []
        aud_vid_dt_matrix = []


        # creating video edges
        for i in range(self.num_vid_nodes):
            start_idx = i - span_over_time_video * video_dilation
            # end_idx = min(
            #     1 + i + span_over_time_video * video_dilation, self.num_vid_nodes
            # )
            end_idx = 1 + i + span_over_time_video * video_dilation
            dt_edges = list(range(start_idx, end_idx, video_dilation))
            dt_matrix = []

            # remove negative index
            while dt_edges[0] < 0:
                dt_edges.pop(0)
                # make sure all nodes have same number of neighbors (same degree)
                dt_edges.append(dt_edges[-1] + video_dilation)
            # remove the last indices if they are out of bound
            while dt_edges[-1] >= self.num_vid_nodes:
                dt_edges.pop(-1)
                # make sure all nodes have same number of neighbors (same degree)
                dt_edges.insert(0, dt_edges[0] - video_dilation)
            # remove self loops if not self_loops
            if not self.self_loops and i in dt_edges:
                dt_edges.remove(i)

            sr_edges = [i] * len(dt_edges)
            vid_dt_edges += dt_edges
            vid_sr_edges += sr_edges

            for x in dt_edges:
                dt_matrix.append(math.exp(-(dt_edges[-1] - x + 1) / (dt_edges[-1] - dt_edges[0] + 1)))
                # t_c = (dt_edges[-1] - dt_edges[0])/2
                # dt_matrix.append(math.exp(((t_c + 1) / (abs(x - t_c) + 1))))
            vid_dt_matrix += dt_matrix
        # vid_edges = np.array([vid_sr_edges, vid_dt_edges])
        vid_edges = np.array([vid_dt_edges, vid_sr_edges])
        vid_weights = np.array([vid_dt_matrix])

        # creating audio edges
        for i in range(self.num_aud_nodes):
            start_idx = i - span_over_time_audio * audio_dilation
            # end_idx = min(
            #     1 + i + span_over_time_audio * audio_dilation, self.num_aud_nodes
            # )
            end_idx = 1 + i + span_over_time_audio * audio_dilation
            dt_edges = list(range(start_idx, end_idx, audio_dilation))
            dt_matrix = []

            # remove negative index
            while dt_edges[0] < 0:
                dt_edges.pop(0)
                dt_edges.append(dt_edges[-1] + audio_dilation)
            while dt_edges[-1] >= self.num_aud_nodes:
                dt_edges.pop(-1)
                # make sure all nodes have same number of neighbors (same degree)
                dt_edges.insert(0, dt_edges[0] - audio_dilation)
            # remove self loops if not self_loops
            if not self.self_loops:
                dt_edges.remove(i)

            sr_edges = [i] * len(dt_edges)
            aud_dt_edges += dt_edges
            aud_sr_edges += sr_edges

            for x in dt_edges:
                dt_matrix.append(math.exp(-(dt_edges[-1] - x + 1) / (dt_edges[-1] - dt_edges[0] + 1)))
                # t_c = (dt_edges[-1] - dt_edges[0])/2
                # dt_matrix.append(math.exp(((t_c + 1) / (abs(x - t_c) + 1))))
            aud_dt_matrix += dt_matrix
        # vid_edges = np.array([vid_sr_edges, vid_dt_edges])
        # aud_edges = np.array([aud_sr_edges, aud_dt_edges])
        aud_edges = np.array([aud_dt_edges, aud_sr_edges])
        aud_weights = np.array([aud_dt_matrix])

        # creating audio-video edges
        # hop_length = np.floor(self.num_aud_nodes / self.num_vid_nodes)
        # # compute the residual audio nodes
        # resd = int(self.num_aud_nodes - (self.num_vid_nodes - 1) * hop_length)
        # # set a residual list for considering extra hop
        # hop_prob_list = [0] * (self.num_vid_nodes - resd) + [1] * resd
        # # shuffle the list
        # np.random.shuffle(hop_prob_list)
        # keep track of hop shift
        hop_shift = 0
        for i in range(self.num_vid_nodes):
            dt_matrix = []
            hop_shift = min(hop_shift + self.hop_prob_list[i], self.resd)
            start_idx = min(
                int(hop_shift + i * self.hop_length), self.num_aud_nodes - 1
            )
            end_idx = min(int(start_idx + span_over_time_between), self.num_aud_nodes)
            dt_edges = list(range(start_idx, end_idx))
            # if not self.self_loops: dt_edges.remove(i)
            sr_edges = [i] * len(dt_edges)
            aud_vid_dt_edges += dt_edges
            aud_vid_sr_edges += sr_edges

            for x in dt_edges:
                dt_matrix.append(math.exp(-(dt_edges[-1] - x + 1) / (dt_edges[-1] - dt_edges[0] + 1)))
                # t_c = (dt_edges[-1] - dt_edges[0])/2
                # dt_matrix.append(math.exp(((t_c + 1) / (abs(x - t_c) + 1))))
            aud_vid_dt_matrix += dt_matrix
        aud_vid_edges = np.array([aud_vid_sr_edges, aud_vid_dt_edges])
        aud_vid_weights = np.array([aud_vid_dt_matrix])
        # for i in range(self.num_aud_nodes):
        #     dt_edges = list(
        #         range(max(0, i - self.span_over_time_between // 2),
        #               min(1 + i + self.span_over_time_between // 2, self.num_aud_nodes)))
        #     # if not self.self_loops: dt_edges.remove(i)
        #     sr_edges = [i] * len(dt_edges)
        #     aud_vid_dt_edges += sr_edges
        #     aud_vid_sr_edges += dt_edges
        # aud_vid_edges = np.array([aud_vid_sr_edges, aud_vid_dt_edges])

        # if self.normalize:
        #     graph['video'].x = torch.nn.functional.normalize(torch.from_numpy(self.videos[idx]))
        #     graph['audio'].x = torch.nn.functional.normalize(torch.from_numpy(self.audios[idx]))
        # else:
        #     graph['video'].x = torch.from_numpy(self.videos[idx])
        #     graph['audio'].x = torch.from_numpy(self.audios[idx])
        graph["video"].x = self.videos[idx]
        graph["audio"].x = self.audios[idx]

        graph["video", "video-video", "video"].edge_index = torch.from_numpy(vid_edges)
        graph["audio", "audio-audio", "audio"].edge_index = torch.from_numpy(aud_edges)
        # the task is acoustics classification so we need to add the edges from video to audio
        graph["video", "video-audio", "audio"].edge_index = torch.from_numpy(
            aud_vid_edges
        )
        # graph['video', 'video-audio', 'audio'].edge_index = torch.from_numpy(aud_vid_edges)

        graph["video", "video-video", "video"].edge_attr = torch.from_numpy(vid_weights)
        graph["audio", "audio-audio", "audio"].edge_attr = torch.from_numpy(aud_weights)
        # the task is acoustics classification so we need to add the edges from video to audio
        graph["video", "video-audio", "audio"].edge_attr = torch.from_numpy(
            aud_vid_weights
        )

        sample = {
            "graph": graph,
            "label": self.labels[idx],
            "numerical_label": self.label_dict[self.labels[idx]],
            "video_add": self.vid_add[idx],
            "audio_add": self.aud_add[idx],
            "span_over_time_between": span_over_time_between,
            "span_over_time_audio": span_over_time_audio,
            "audio_dilation": audio_dilation,
            "span_over_time_video": span_over_time_video,
            "video_dilation": video_dilation,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class AudioSetGraphDataset_plain(Dataset):
    """AudioSet graph dataset."""

    def __init__(
        self, path, graph_config, seed=0, transform=None, disired_classes="All"
    ):
        """
        Args:
            path (string): Path to the node attributes files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        f = h5py.File(path, "r")
        self.videos = f["video"][()]
        self.audios = f["audio"][()]
        self.labels = f["label"][()]
        self.vid_add = f["video_add"][()]
        self.aud_add = f["audio_add"][()]
        True_labels = f["unique_labels"][()]
        del f
        self.labels = np.array([label.decode("ascii") for label in self.labels])
        self.vid_add = np.array([add.decode("ascii") for add in self.vid_add])
        self.aud_add = np.array([add.decode("ascii") for add in self.aud_add])
        self.label_dict = defaultdict(int)
        ## filter data based on the classes
        if disired_classes != ["All"]:
            self.disired_idx = [
                idx for idx, label in enumerate(self.labels) if label in disired_classes
            ]
            self.videos = self.videos[self.disired_idx]
            self.audios = self.audios[self.disired_idx]
            self.labels = self.labels[self.disired_idx]
            self.vid_add = self.vid_add[self.disired_idx]
            self.aud_add = self.aud_add[self.disired_idx]
            for idx, label in enumerate(disired_classes):
                self.label_dict[label] = idx
        else:
            for idx, label in enumerate(True_labels):
                self.label_dict[label.decode("ascii")] = idx

        self.num_vid_nodes = self.videos[0].shape[0]
        self.num_aud_nodes = self.audios[0].shape[0]
        self.normalize = graph_config.NORMALIZE
        self.transform = transform

        # normalize the features np.corrcoef
        if self.normalize:
            # self.videos = torch.nn.functional.normalize(torch.from_numpy(self.videos))
            self.videos = torch.from_numpy(self.videos)

            B, N, C = self.audios.shape
            self.audios = F.normalize(
                torch.from_numpy(self.audios).view(B, -1), p=2, dim=1
            ).view(B, N, C)
            # self.audios = torch.nn.functional.normalize(torch.from_numpy(self.audios))
        else:
            self.videos = torch.from_numpy(self.videos)
            self.audios = torch.from_numpy(self.audios)

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "audio_data": self.audios[idx],
            "video_data": self.videos[idx],
            "label": self.labels[idx],
            "numerical_label": self.label_dict[self.labels[idx]],
            "video_add": self.vid_add[idx],
            "audio_add": self.aud_add[idx],

        }

        if self.transform:
            sample = self.transform(sample)

        return sample
