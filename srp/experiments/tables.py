import os
from glob import glob
import pandas as pd
import numpy as np
from srp.config import C


def output_mean_variance_table(root):
    index_label = []
    rows = []
    col_names = None
    for subdir in os.listdir(root):
        
        if subdir.startswith(('.', '_')):
            continue
        index_label.append(subdir)
        csvs = glob(os.path.join(root, subdir, '*.csv'))
        
        metrics = []
        for csv in csvs:
            
            if not col_names:
                col_names = pd.read_csv(csv).columns[1:].tolist()
            
            history = pd.read_csv(csv)
            metrics.append(history.iloc[len(history)-5, 1:].values)
            # print (index_label)
        mean = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0)
        rows.append(["{:.3f}±{:.3f}".format(m,s) for (m, s) in zip(mean, std)])
    return pd.DataFrame(data=np.array(rows), columns=col_names, index=index_label)   

def output_folds_table(root):
    
    csvs = glob(os.path.join(root, '*/*.csv'))
    useful_cols = pd.read_csv(csvs[0]).columns[1:].tolist()

    rows = []
    index_label = []
    for exp in csvs:
        history = pd.read_csv(exp)
        assert history.columns[1:].tolist() == useful_cols
        rows.append(history.iloc[len(history)-5, 1:].values)
        index_label.append(os.path.basename(os.path.dirname(exp))+'_'+os.path.basename(exp)[-5:-4])
    assert len(useful_cols) == len(rows[0])

    return pd.DataFrame(data=np.array(rows), columns=useful_cols, index=index_label)

if __name__ == '__main__':
    experiment_root =os.path.join(C.DATA, 'experiments')
    roots = [exp for exp in os.listdir(experiment_root) if not exp.startswith(('.', '_'))]
    
    for r in roots:
        df = output_folds_table(os.path.join(experiment_root, r))
        df.to_csv('folds_{}_c{}r{}.csv'.format(r, C.TRAIN.REGRESSION_WEIGHT, C.TRAIN.CLASSIFICATION_WEIGHT))
        df = output_mean_variance_table(os.path.join(experiment_root, r))
        df.to_csv('summary_{}_c{}r{}.csv'.format(r, C.TRAIN.REGRESSION_WEIGHT, C.TRAIN.CLASSIFICATION_WEIGHT)) 
