import h5py
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import HeteroData

# Load data embeddings
path = "../Output_clip_len_1.0_audio_10/AudioSet_embedds_all.h5"
f = h5py.File(path, "r")
videos = f["video"][()]
audios = f["audio"][()]
labels = f["label"][()]

# set graph construction param
span_over_time = 2
normalize = True  # make attributes normalized
self_loops = True

Graphs = []
num_vid_nodes, num_aud_nodes = videos[0].shape[0], audios[0].shape[0]

for video, audio, label in zip(videos, audios, labels):
    graph = HeteroData()

    # creating edges
    vid_sr_edges = []
    vid_dt_edges = []
    aud_sr_edges = []
    aud_dt_edges = []
    for i in range(num_vid_nodes):
        dt_edges = list(
            range(
                max(0, i - span_over_time // 2),
                min(1 + i + span_over_time // 2, num_vid_nodes),
            )
        )
        if not self_loops:
            dt_edges.remove(i)
        sr_edges = [i] * len(dt_edges)
        vid_dt_edges += dt_edges
        vid_sr_edges += sr_edges
    vid_edges = np.array([vid_sr_edges, vid_dt_edges])

    for i in range(num_vid_nodes, num_vid_nodes + num_aud_nodes):
        dt_edges = list(
            range(
                max(num_vid_nodes, i - span_over_time // 2),
                min(1 + i + span_over_time // 2, num_vid_nodes + num_aud_nodes),
            )
        )
        if not self_loops:
            dt_edges.remove(i)
        sr_edges = [i] * len(dt_edges)
        aud_dt_edges += dt_edges
        aud_sr_edges += sr_edges
    aud_edges = np.array([aud_sr_edges, aud_dt_edges])

    # import torch.nn.functional as F
    if normalize:
        graph["video"].x = torch.nn.functional.normalize(torch.from_numpy(video))
        graph["audio"].x = torch.nn.functional.normalize(torch.from_numpy(audio))
    else:
        graph["video"].x = torch.from_numpy(video)
        graph["audio"].x = torch.from_numpy(audio)
    # graph['video'].y = torch.zeros(num_aud_nodes, dtype=torch.long)
    # graph['audio'].y = torch.ones(num_aud_nodes, dtype=torch.long)

    graph["video", "video-video", "video"].edge_index = torch.from_numpy(vid_edges)
    graph["audio", "audio-audio", "audio"].edge_index = torch.from_numpy(aud_edges)
    graph["audio", "audio-video", "video"].edge_index = torch.from_numpy(
        np.array(
            [
                list(range(num_vid_nodes)),
                list(range(num_vid_nodes, num_vid_nodes + num_aud_nodes)),
            ]
        )
    )

    # graph['label'].x = label.decode('ascii')

    # import torch_geometric.transforms as T
    # graph2 = T.NormalizeFeatures(['video'])(graph)
    # graph2 = T.ToUndirected()(graph.__copy__())

    Graphs.append(graph)
    # break

a = 0

# from torch_geometric.datasets import OGB_MAG
#
# dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
# data = dataset[0]
#
# paper_node_data = data['paper']
# cites_edge_data = data['paper', 'cites', 'paper']
