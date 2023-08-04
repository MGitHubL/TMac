# import relevant libraries
import os

import torch

from MultiModalGraph.configs.config import get_cfg
from MultiModalGraph.model.meta_arch import CNN, LSTM, GraphNN
# from MultiModalGraph.model.meta_arch import CNN, LSTM, GraphNN
from MultiModalGraph.train_utils.train_loop import DefaultTrainer
from MultiModalGraph.utils.logger import setup_logger

setup_logger()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

# set default configs
cfg = get_cfg()
cfg.merge_from_file("configs/AudioSet.yaml")
cfg.DEVICE = device
cfg.MODEL.DEVICE = device
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
# specify the disered classes
# cfg.DATALOADER.DISERED_CLASSES = ['Speech', 'Music']
# cfg.DATALOADER.DISERED_CLASSES = [
#     "Speech",
#     "Music",
#     "Cat",
#     "Dog",
#     "Car",
#     "Train",
#     "Bird",
#     "Gunshot, gunfire",
# ]

cfg.DATALOADER.DISERED_CLASSES = [
    "Aircraft",
    "Ambulance (siren)",
    "Bicycle",
    "Bird",
    "Boom",
    "Bus",
    "Camera",
    "Car",
    "Cash register",
    "Cat",
    "Cattle, bovinae",
    "Church bell",
    "Clock",
    "Dog",
    "Mechanical fan",
    "Fireworks",
    "Goat",
    "Gunshot, gunfire",
    "Hammer",
    "Horse",
    "Motorcycle",
    "Ocean",
    "Pant",
    "Pig",
    "Printer",
    "Rain",
    "Sawing",
    "Sewing machine",
    "Skateboard",
    "Stream",
    "Thunderstorm",
    "Train",
    "Truck",
]
cfg.MODEL.OUT_DIM = len(cfg.DATALOADER.DISERED_CLASSES)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
