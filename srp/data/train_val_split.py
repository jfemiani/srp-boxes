import numpy as np
from srp.config import C
import os
import pandas as pd
from sklearn.model_selection import KFold


def train_val_split():
    kf = KFold(n_splits=C.TRAIN.SAMPLES.FOLDS, shuffle=True, random_state=C.TRAIN.SRAND)
    
    posinfo = pd.read_csv(os.path.join(C.TRAIN.SAMPLES.DIR, "positives.csv")).values
    neginfo = pd.read_csv(os.path.join(C.TRAIN.SAMPLES.DIR, "negatives.csv")).values

    for i, (train_index, test_index) in enumerate(kf.split(posinfo)):
        fold_dir = os.path.join(C.TRAIN.SAMPLES.DIR, "fold{}".format(i + 1))
        os.makedirs(fold_dir, mode=777, exist_ok=True)

        with open(os.path.join(fold_dir, "train.txt"), mode='w') as file:
            for idx in train_index:
                file.write(posinfo[idx,0] + '\n')
                # file.write("1, {}\n".format(idx+1))

        with open(os.path.join(fold_dir, "test.txt"), mode='w') as file:
            for idx in test_index:
                file.write(posinfo[idx,0] + '\n')
                # file.write("1, {}\n".format(idx+1))

    for i, (train_index, test_index) in enumerate(kf.split(neginfo)):
        fold_dir = os.path.join(C.TRAIN.SAMPLES.DIR, "fold{}".format(i + 1))

        with open(os.path.join(fold_dir, "train.txt"), mode='a') as file:
            for idx in train_index:
                file.write(neginfo[idx,0] + '\n')
                
                # file.write("0, {}\n".format(idx+1))

        with open(os.path.join(fold_dir, "test.txt"), mode='a') as file:
            for idx in test_index:
                file.write(neginfo[idx,0] + '\n')
                # file.write("0, {}\n".format(idx+1))

if __name__ == '__main__':
    train_val_split()
