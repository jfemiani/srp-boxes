"""Main script for the training process"""
from .config import C
from .rgblidar_dataset import RgbLidarDataset
from .arch import Solver, Architecture
import torch
import torch.nn
import torch.utils.data


def train():
    """Do the main training loop, as described in the config.
    """

    training_data = RgbLidarDataset('train')
    eval_data = RgbLidarDataset('eval')

    net = Architecture(shape=C.TRAIN.PATCH_SIZE,
                       lidar_channels=C.VOLUME.Z.STEPS,
                       rgb_channels=3,
                       num_features=C.TRAIN.NUM_FEAURES)

    class_sample_counts = training_data.dataset['label']

    training_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=C.TRAIN.BATCH_SIZE,
        shuffle=True,
        sampler=, batch_sampler=None`> <`6:  # :, num_workers=0`><`7:#:, collate_fn=default_collate`><`8:#:, pin_memory=False`><`9:#:, drop_last=False`><`10:#:, timeout=0`><`11:#:, worker_init_fn=None`>)<`12`>
    solver=Solver(trn_loader, val_loader, net, **kwargs)
