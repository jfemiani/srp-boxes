import numpy as np
import srp.config as C
import os
import pandas as pd
from sklearn.model_selection import KFold

def train_val_split():
    kf = KFold(n_splits=C.FOLDS, shuffle=True, random_state=C.FOLD_RANDOM_SEED)
    prefix = os.path.join(C.INT_DATA, "srp")
    
    posinfo = pd.read_csv(os.path.join(C.CSV_DIR, "positives_{}.csv".format(C.VOLUME_DEFAULT_CRS.replace(":","")))).values
    neginfo = pd.read_csv(os.path.join(C.CSV_DIR, "negatives_{}.csv".format(C.VOLUME_DEFAULT_CRS.replace(":","")))).values
    
    for i, (train_index, test_index) in enumerate(kf.split(posinfo)):
        fold_dir = os.path.join(prefix, "fold{}".format(i+1))
        os.makedirs(fold_dir, mode=777, exist_ok=True)

        with open(os.path.join(fold_dir, "train.txt"), mode='w') as file:
            for idx in train_index:
                file.write("1, {}\n".format(idx))

        with open(os.path.join(fold_dir, "test.txt"), mode='w') as file:
            for idx in test_index:
                file.write("1, {}\n".format(idx))

    for i, (train_index, test_index) in enumerate(kf.split(neginfo)):
        fold_dir = os.path.join(prefix, "fold{}".format(i+1))

        with open(os.path.join(fold_dir, "train.txt"), mode='a') as file:
            for idx in train_index:
                file.write("0, {}\n".format(idx))

        with open(os.path.join(fold_dir, "test.txt"), mode='a') as file:
            for idx in test_index:
                file.write("0, {}\n".format(idx))
