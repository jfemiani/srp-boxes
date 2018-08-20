"""Main script for the training process"""
import numpy as np

import torch
import torch.nn
import torch.utils.data
from srp.config import C
from srp.model.arch import Architecture, Solver
from srp.model.rgblidar_dataset import RgbLidarDataset
from srp.util import tqdm


def make_stratified_sampler(dataset, target_ratio, num_samples = None):
    """Generate a sampler that will balance the labels.

    Parameters
    ----------
    dataset: An RgbLidar dataset that yields (x,y) where y=(label, params)
    target_ratio: The desired ration of neative and positive samples; As a list [num_neg, num_pos]
    num_samples: The size of the resampled dataeset; default stays he same size.

    Returns
    -------
    sampler (WeightedRandomSampler):A pytorch sampler that acheives the target rations.

    """
    labels = []
    for i, (x, y) in enumerate(tqdm(dataset, desc='Counting labels')):
        classlabel, params = y
        labels.append(classlabel)
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    if num_samples is None:
        num_samples = len(dataset)

    # The _negative_ weight is proportional to the number of _non-negative_ samples. And vice-versa.
    class_weights = np.array([num_pos, num_neg])/max([num_neg, num_pos])
    class_weights *= C.TRAIN.CLASS_BALANCE
    sample_weights = class_weights[labels]
    return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, num_samples=num_samples)

def train():
    """Do the main training loop, as described in the config.
    """
    training_data = RgbLidarDataset('train')
    eval_data = RgbLidarDataset('eval')
    train_sampler = make_stratified_sampler(training_data, C.TRAIN.CLASS_BALANCE)
    eval_sampler = make_stratified_sampler(eval_data, C.TRAIN.CLASS_BALANCE)
    trn_loader = torch.utils.data.DataLoader(training_data, batch_size=C.TRAIN.BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(eval_data, batch_size=C.TRAIN.BATCH_SIZE, sampler=eval_sampler)

    net = Architecture(shape=(C.TRAIN.PATCH_SIZE, C.TRAIN.PATCH_SIZE),
                       lidar_channels=C.VOLUME.Z.STEPS,
                       rgb_channels=3,
                       num_features=C.TRAIN.NUM_FEAURES)

    solver=Solver(trn_loader, val_loader, net, **kwargs)



if __name__ == '__main__':
    with torch.cuda.device(C.TORCH.DEVICE):
        train()
