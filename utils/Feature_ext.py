import glob
import os
import time

# Ignore warnings
import warnings

import h5py
import numpy as np
import torch
from joblib import Parallel  # install psutil library to manage memory leak
from joblib import delayed
from torchvision import transforms
from torchvision.models import resnext50_32x4d as resnext
from tqdm import tqdm

import utils.augmentation as A
import utils.transforms as T
from model.s3dg import S3D
from MultiModalGraph.utils.utils import load_model, video_reader

warnings.filterwarnings("ignore")


# load sound embedding network
audio_enc = torch.hub.load("harritaylor/torchvggish", "vggish")
audio_enc.cuda()
audio_enc.eval()

# load image encoder network
# image_enc = resnext(pretrained=True, progress=True)
# image_enc.cuda()
# image_enc.eval()

# load video encoder network
video_enc = S3D(input_channel=3)
load_model(video_enc, "pretrained_models/CoCLR-ucf101-rgb-128-s3d-ep182.tar")
video_enc.cuda()
video_enc.eval()

# Downloaded AudioSet examples
audioset_path = "/media/amir_shirian/Amir/Datasets/Sound Recognition/AudioSet/Eval"

# params
img_dim = 128
num_img = 10  # number of images were sampled from each video
threshold = 100  # number of clips which will process in a single step
num_workers = 5

# essential transforms for image and video
with torch.no_grad():
    transform = transforms.Compose(
        [
            # A.CenterCrop(size=(224, 224)),
            A.Scale(size=(img_dim, img_dim)),
            # A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
            A.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([A.ToTensor()])
    transform_cuda = transforms.Compose(
        [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)]
    )

# a param that set the length of sub-video
sub_clip_len = 0.1  # sec  0.25
# desired length of audio clips
audio_clip_len = 129  # segments  101
# desired length of video clips
video_clip_len = 100  # segments  40
# make sure that the fps of video is above that the desired length of sub-video clips
threshold_len = 10  # in fps

# Output path
output_path = f"./Output_clip_len_{sub_clip_len}_audio_{audio_clip_len}"

# Creat destination folders
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

# name of processed folders
processed_folders = [
    name.split("_")[-1].split(".")[0] for name in os.listdir(output_path)
]
if "all" in processed_folders:
    processed_folders.remove("all")

for folder in os.listdir(audioset_path):

    if folder not in processed_folders:
        # if True:
        VIDEO_embeddings = []
        AUDIO_embeddings = []
        # IMAGE_embeddings = []
        VIDEO_addresses = []
        AUDIO_addresses = []
        LABEL = []

        print(f"Processing {folder} folder...")
        folder_path = os.path.join(audioset_path, folder)
        vid_list = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))
        aud_list = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
        length_vid = [
            int(add.split("_end_")[1].split(".")[0])
            - int(add.split("_end_")[0].split("_start_")[1])
            for add in vid_list
        ]

        if len(vid_list) > threshold:
            vid_sub_list = []
            aud_sub_list = []
            video_embed = []
            audio_embed = []
            image_embed = []
            step_size = 10
            for i in tqdm(
                range(len(vid_list) // threshold), total=len(vid_list) // threshold
            ):
                start_idx, end_idx = i * threshold, (i + 1) * threshold
                vid_sub_list = vid_list[start_idx:end_idx]
                aud_sub_list = aud_list[start_idx:end_idx]
                length_sub_vid = length_vid[start_idx:end_idx]
                T = time.time()

                print("\n Reading videos...")
                vid_sub_array = Parallel(n_jobs=num_workers)(
                    delayed(video_reader)(vp, transform) for vp in vid_sub_list
                )
                print("\n Finished reading...")
                FPS = [
                    vid.shape[2] // len
                    for vid, len in zip(vid_sub_array, length_sub_vid)
                ]

                # remove videos with fps less than threshold_len
                while len(np.where(np.array(FPS) < threshold_len)[0]) > 0:
                    detected_clips_idx = np.where(np.array(FPS) < threshold_len)[0][0]
                    vid_sub_array.pop(detected_clips_idx)
                    aud_sub_list.pop(detected_clips_idx)
                    FPS.pop(detected_clips_idx)
                    length_sub_vid.pop(detected_clips_idx)

                # make sure that the length of videos are not less than ...
                vid_sub_array = [
                    vid.repeat(
                        1,
                        1,
                        int(
                            np.ceil(
                                (int(length / sub_clip_len) * int(fps * sub_clip_len))
                                / vid.shape[2]
                            )
                        )
                        + 1,
                        1,
                        1,
                    )
                    if vid.shape[2]
                    < int(length / sub_clip_len) * int(fps * sub_clip_len)
                    else vid
                    for vid, length, fps in zip(vid_sub_array, length_vid, FPS)
                ]

                Vid_array = [
                    [
                        vid[:, :, x : min(x + int(fps * sub_clip_len), length * fps)]
                        for x in range(0, length * fps, int(fps * sub_clip_len))[
                            : int(length / sub_clip_len)
                        ]
                    ]
                    for vid, length, fps in zip(vid_sub_array, length_vid, FPS)
                ]
                # make sure that each sub clip has the length more that 5 frames otherwise the video encoder will throw error,
                # I just repeat the sub clip if the length is less than 5 frames
                Vid_array = [
                    [
                        subclip
                        if subclip.shape[2] >= 5
                        else subclip.repeat(1, 1, 5 // subclip.shape[2] + 1, 1, 1)
                        for subclip in vid
                    ]
                    for vid in Vid_array
                ]

                idx = [
                    list(range(vid.shape[2]))[:: (vid.shape[2] // num_img)][:num_img]
                    for vid in vid_sub_array
                ]
                img_sub_array = [
                    vid.permute(0, 2, 1, 3, 4).squeeze(0)[idx[i]]
                    for i, vid in enumerate(vid_sub_array)
                ]

                try:
                    with torch.no_grad():

                        video_embed.extend(
                            [
                                video_enc(torch.cat(vid, dim=0).cuda())
                                .detach()
                                .cpu()
                                .numpy()
                                for vid in Vid_array
                            ]
                        )
                        audio_embed.extend(
                            [
                                audio_enc.forward(aud).detach().cpu().numpy()
                                for aud in aud_sub_list
                            ]
                        )

                except:
                    raise ValueError(" ")
            vid_sub_list = vid_list[end_idx:]
            aud_sub_list = aud_list[end_idx:]
            length_sub_vid = length_vid[end_idx:]
            vid_sub_array = Parallel(n_jobs=num_workers)(
                delayed(video_reader)(vp, transform, transform_cuda)
                for vp in vid_sub_list
            )
            FPS = [
                vid.shape[2] // len for vid, len in zip(vid_sub_array, length_sub_vid)
            ]

            # remove videos with fps less than threshold_len
            while len(np.where(np.array(FPS) < threshold_len)[0]) > 0:
                detected_clips_idx = np.where(np.array(FPS) < threshold_len)[0][0]
                vid_sub_array.pop(detected_clips_idx)
                aud_sub_list.pop(detected_clips_idx)
                FPS.pop(detected_clips_idx)
                length_sub_vid.pop(detected_clips_idx)

            # make sure that the length of videos are not less than ...
            vid_sub_array = [
                vid.repeat(
                    1,
                    1,
                    int(
                        np.ceil(
                            (int(length / sub_clip_len) * int(fps * sub_clip_len))
                            / vid.shape[2]
                        )
                    )
                    + 1,
                    1,
                    1,
                )
                if vid.shape[2] < int(length / sub_clip_len) * int(fps * sub_clip_len)
                else vid
                for vid, length, fps in zip(vid_sub_array, length_vid, FPS)
            ]

            Vid_array = [
                [
                    vid[:, :, x : min(x + int(fps * sub_clip_len), length * fps)]
                    for x in range(0, length * fps, int(fps * sub_clip_len))[
                        : int(length / sub_clip_len)
                    ]
                ]
                for vid, length, fps in zip(vid_sub_array, length_vid, FPS)
            ]

            # make sure that each sub clip has the length more that 5 frames otherwise the video encoder will throw error,
            # I just repeat the sub clip if the length is less than 5 frames
            Vid_array = [
                [
                    subclip
                    if subclip.shape[2] >= 5
                    else subclip.repeat(1, 1, 5 // subclip.shape[2] + 1, 1, 1)
                    for subclip in vid
                ]
                for vid in Vid_array
            ]

            idx = [
                list(range(vid.shape[2]))[:: (vid.shape[2] // num_img)][:num_img]
                for vid in vid_sub_array
            ]

            try:
                with torch.no_grad():

                    video_embed.extend(
                        [
                            video_enc(torch.cat(vid, dim=0).cuda())
                            .detach()
                            .cpu()
                            .numpy()
                            for vid in Vid_array
                        ]
                    )
                    audio_embed.extend(
                        [
                            audio_enc.forward(aud).detach().cpu().numpy()
                            for aud in aud_sub_list
                        ]
                    )

            except:
                raise ValueError(" ")
        else:
            print("Reading videos...")
            vid_array = Parallel(n_jobs=num_workers)(
                delayed(video_reader)(vp, transform)
                for vp in tqdm(vid_list, total=len(vid_list))
            )
            print("Finished reading...")
            FPS = [vid.shape[2] // len for vid, len in zip(vid_array, length_vid)]

            # remove videos with fps less than threshold_len
            while len(np.where(np.array(FPS) < threshold_len)[0]) > 0:
                detected_clips_idx = np.where(np.array(FPS) < threshold_len)[0][0]
                vid_array.pop(detected_clips_idx)
                aud_list.pop(detected_clips_idx)
                FPS.pop(detected_clips_idx)
                length_vid.pop(detected_clips_idx)

            # make sure that the length of videos are not less than ...
            vid_array = [
                vid.repeat(
                    1,
                    1,
                    int(
                        np.ceil(
                            (int(length / sub_clip_len) * int(fps * sub_clip_len))
                            / vid.shape[2]
                        )
                    )
                    + 1,
                    1,
                    1,
                )
                if vid.shape[2] < int(length / sub_clip_len) * int(fps * sub_clip_len)
                else vid
                for vid, length, fps in zip(vid_array, length_vid, FPS)
            ]

            Vid_array = [
                [
                    vid[:, :, x : min(x + int(fps * sub_clip_len), length * fps)]
                    for x in range(0, length * fps, int(fps * sub_clip_len))[
                        : int(length / sub_clip_len)
                    ]
                ]
                for vid, length, fps in zip(vid_array, length_vid, FPS)
            ]

            # make sure that each sub clip has the length more that 5 frames otherwise the video encoder will throw error,
            # I just repeat the sub clip if the length is less than 5 frames
            Vid_array = [
                [
                    subclip
                    if subclip.shape[2] >= 5
                    else subclip.repeat(1, 1, 5 // subclip.shape[2] + 1, 1, 1)
                    for subclip in vid
                ]
                for vid in Vid_array
            ]
            idx = [
                list(range(vid.shape[2]))[:: (vid.shape[2] // num_img)][:num_img]
                for vid in vid_array
            ]
            img_array = [
                vid.permute(0, 2, 1, 3, 4).squeeze(0)[idx[i]]
                for i, vid in enumerate(vid_array)
            ]

            try:
                with torch.no_grad():

                    video_embed = [
                        video_enc(torch.cat(vid, dim=0).cuda()).detach().cpu().numpy()
                        for vid in Vid_array
                    ]
                    audio_embed = [
                        audio_enc.forward(aud).detach().cpu().numpy()
                        for aud in aud_list
                    ]

            except:
                print(f"could not process {folder}")
                pass

        c = 0
        for _, (vid_embed, aud_embed, vid_add, aud_add) in enumerate(
            zip(video_embed, audio_embed, vid_list, aud_list)
        ):
            if (
                aud_embed.shape[0] == audio_clip_len
                and vid_embed.shape[0] == video_clip_len
            ):
                VIDEO_embeddings.append(vid_embed)
                AUDIO_embeddings.append(aud_embed)
                VIDEO_addresses.append(vid_add)
                AUDIO_addresses.append(aud_add)
                # IMAGE_embeddings.append(img_embed)
                LABEL.append(folder)
                c += 1
        print(f"Added {c} from {len(vid_list)} items ...")

        hf = h5py.File(f"{output_path}/AudioSet_embedds_{folder}.h5", "w")
        hf.create_dataset("audio", data=AUDIO_embeddings)
        hf.create_dataset("video", data=VIDEO_embeddings)
        # hf.create_dataset('image', data=IMAGE_embeddings)
        hf.create_dataset("video_add", data=np.string_(VIDEO_addresses))
        hf.create_dataset("audio_add", data=np.string_(AUDIO_addresses))
        hf.close()
