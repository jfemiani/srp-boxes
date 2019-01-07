"""Main script for the training process"""
import numpy as np
import os
from srp.data.generate_patches import Patch
import torch
import torch.nn
import torch.utils.data
from srp.config import C
from srp.model.arch import Architecture, Solver
from srp.model.rgblidar_dataset import RgbLidarDataset
from srp.util import tqdm
import warnings
import pandas as pd

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

def train(fold=C.TRAIN.SAMPLES.CURRENT_FOLD):
    """Do the main training loop, as described in the config.
    """
    warnings.filterwarnings('always')
    training_data = RgbLidarDataset('train', fold=fold)
    eval_data = RgbLidarDataset('test', fold=fold)
    
    
    train_sampler = make_stratified_sampler(training_data, C.TRAIN.CLASS_BALANCE)
    eval_sampler = make_stratified_sampler(eval_data, C.TRAIN.CLASS_BALANCE)
    
    trn_loader = torch.utils.data.DataLoader(training_data, batch_size=C.TRAIN.BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(eval_data, batch_size=C.TRAIN.BATCH_SIZE, sampler=eval_sampler)
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    net = Architecture(shape=(C.TRAIN.PATCH_SIZE, C.TRAIN.PATCH_SIZE),
                       lidar_channels=C.VOLUME.Z.STEPS,
                       rgb_channels=3,
                       num_features=C.TRAIN.NUM_FEATURES)
    
    solver = Solver(trn_loader, val_loader, net=net)
    solver.train()

    eval_data = [(h.epoch, h.trn_loss, h.val_loss, h.f_measure, h.accuracy, h.precision, h.recall, h.f1, h.iou, h.l1, h.l2) for h in solver.history]
    columns = ['epoch', 'trn_loss', 'val_loss', 'f_2', 'accuracy', 'precision', 'recall', 'f_1', 'iou', 'L_1', 'L_2']
    pd.DataFrame(np.array(eval_data), columns=columns).to_csv('history_fold{}.csv'.format(fold))



if __name__ == '__main__':
    with torch.cuda.device(C.TORCH.DEVICE):
        # training on 5 folds
        for i in range(1, C.TRAIN.SAMPLES.FOLDS + 1):
            print (os.getcwd())
            print ("This is the {} fold!!!".format(i))
            #f = 'history_fold{}.csv'.format(i)
            # if len(pd.read_csv(f).columns) < 12:
            train(fold=i)
