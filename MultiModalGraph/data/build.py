import itertools
import logging

import numpy as np
from tabulate import tabulate
from termcolor import colored

from MultiModalGraph.data.samplers import *
from MultiModalGraph.utils.logger import log_first_n
import torch
from torch_geometric.loader import DataLoader

from utils.dataloader import AudioSetGraphDataset, AudioSetGraphDataset_plain
from sklearn.model_selection import train_test_split


def build_AudioSet_data_loaders(cfg, output_mode="train"):


    seed = cfg.SEED  # comm.shared_random_seed()

    dataset = AudioSetGraphDataset(
        path=cfg.DATASETS.TRAIN_PATH,
        graph_config=cfg.GRAPH,
        seed=seed,
        disired_classes=cfg.DATALOADER.DISERED_CLASSES,
    )
    if cfg.DATASETS.EVAL_PATH is "" and cfg.DATASETS.TEST_PATH is "":
        train_size = int(cfg.DATASETS.TRAIN_RATIO * len(dataset))
        eval_size = int(cfg.DATASETS.EVAL_RATIO * len(dataset))
        test_size = len(dataset) - train_size - eval_size
        train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, eval_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
    elif cfg.DATASETS.EVAL_PATH is "":
        test_dataset = AudioSetGraphDataset(
            path=cfg.DATASETS.TEST_PATH,
            graph_config=cfg.GRAPH,
            seed=seed,
            disired_classes=cfg.DATALOADER.DISERED_CLASSES,
        )

        train_size = int(cfg.DATASETS.TRAIN_RATIO * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(seed)
        )

    # stratified split
    if cfg.DATALOADER.STRATIFIED_SPLIT:
        if cfg.DATASETS.EVAL_PATH is "" and cfg.DATASETS.TEST_PATH is "":
            indices = np.arange(len(dataset))
            numerical_labels = [dataset.label_dict[dataset.labels[i]] for i in range(len(dataset))]
            x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(indices, numerical_labels,
                                                                                              test_size=1 - cfg.DATASETS.TRAIN_RATIO,
                                                                                              random_state=seed)
            x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                          test_size=.5, random_state=seed)
            train_dataset.indices = list(x_train)
            eval_dataset.indices = list(x_validation)
            test_dataset.indices = list(x_test)
        elif cfg.DATASETS.EVAL_PATH is "":
            indices = np.arange(len(dataset))
            numerical_labels = [dataset.label_dict[dataset.labels[i]] for i in range(len(dataset))]
            x_train, x_validation, y_train, y_validation = train_test_split(indices, numerical_labels,
                                                                            test_size=1 - cfg.DATASETS.TRAIN_RATIO,
                                                                            random_state=seed)
            train_dataset.indices = list(x_train)
            eval_dataset.indices = list(x_validation)

    if output_mode == "train":
        print_instances_class_histogram(
            [dataset.label_dict[label] for label in dataset.labels],
            list(dataset.label_dict.keys()),
        )
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(train_dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                train_dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))
        return DataLoader(
            train_dataset,
            cfg.DATALOADER.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
    if output_mode == "eval":
        return DataLoader(eval_dataset, cfg.DATALOADER.BATCH_SIZE, shuffle=True)
    if output_mode == "test":
        return DataLoader(test_dataset, cfg.DATALOADER.BATCH_SIZE, shuffle=False)
    # return (
    #     DataLoader(train_dataset, cfg.DATALOADER.BATCH_SIZE, sampler=sampler),
    #     DataLoader(eval_dataset, cfg.DATALOADER.BATCH_SIZE, shuffle=True),
    #     DataLoader(test_dataset, cfg.DATALOADER.BATCH_SIZE, shuffle=False),
    # )


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for Class in dataset_dicts:
        assert Class >= 0, f"Got an invalid category_id={Class}"
        assert (
            Class < num_classes
        ), f"Got an invalid category_id={Class} for a dataset of {num_classes} classes"

        histogram += np.histogram(Class, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(
            *[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)]
        )
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )
