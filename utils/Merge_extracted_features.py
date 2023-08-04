import os

import h5py
import numpy as np

dir = "./Output_clip_len_1.0_audio_10"
VIDEO_embeddings = []
AUDIO_embeddings = []
# IMAGE_embeddings = []
VIDEO_addresses = []
AUDIO_addresses = []
LABELS = []
UNIQUE_LABELS = {}
for file in os.listdir(dir):
    # To make sure that the file is not AudioSet_embedds_all.h5
    if file != "AudioSet_embedds_all.h5":
        print(file)
        f = h5py.File(os.path.join(dir, file), "r")
        label = file.split(".")[0].split("_")[-1]
        VIDEO_embeddings.extend(f["video"][()])
        AUDIO_embeddings.extend(f["audio"][()])
        # IMAGE_embeddings.extend(f['image'][()])
        VIDEO_addresses.extend(f["video_add"][()])
        AUDIO_addresses.extend(f["audio_add"][()])
        LABELS.extend([label] * len(f["video"][()]))
        if label not in UNIQUE_LABELS:
            UNIQUE_LABELS[label] = 1

hf = h5py.File(f"{dir}/AudioSet_embedds_all.h5", "w")
hf.create_dataset("video", data=VIDEO_embeddings)
hf.create_dataset("audio", data=AUDIO_embeddings)
# hf.create_dataset('image', data=IMAGE_embeddings)
hf.create_dataset("video_add", data=np.string_(VIDEO_addresses))
hf.create_dataset("audio_add", data=np.string_(AUDIO_addresses))
hf.create_dataset("label", data=np.string_(LABELS))
hf.create_dataset("unique_labels", data=np.string_(list(UNIQUE_LABELS.keys())))
hf.close()
