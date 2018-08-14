import pickle
import affine
import pandas as pd
from srp.config import C
import sys
import skimage
import numpy as np
import scipy
from srp.data.data_provider import DataProvider
from srp.data.orientedboundingbox import OrientedBoundingBox
from pathlib import Path
from srp.util import tqdm
from collections import namedtuple
from torch.utils.data import Dataset
from skimage.transform import rotate

from srp.data.generate_patches import Patch


class RgbLidarDataset(Dataset):
    """An Rgb+Lidar Combined Dataset
    
    This dataset reads in patches that were saved as pickle files. 
    
    """
    def __init__(self, txt_dir=None):
        """
        This class is design to use in conjunction with the pytorch dataloader class
        txt_dir: a PosixPath object which specify the path of the .txt which stores train (or test) sample indices.
        
        Attributes:
        txt_dir: the directory to dataset file, relative to C.TRAIN.SAMPLES.DIR
        
        
        Example:

        >>> trn_data = RgbLidarDataset(trn_txt)
        >>> dl = DataLoader(trn_data, batch_size=128, shuffle=True, num_workers=3)
        >>> trn_data.preaugment(num_variants=20)      #generates 20 samples for every record in trn_txt
        >>> X, y = iter(dl).next()
        >>> X.shape
        torch.Size([128, 9, 64, 64])

        >>> len(y)
        2

        >>> y[0].shape         # here are all the isbox labels
        torch.Size([128])

        >>> y[1].shape         # here are all the four corners. Negative samples are np.zero((4,2))
        torch.Size([128, 4, 2])

        Note that the length of the dataset is the number of rows in `trn_txt`.
        Please specify epoch length accordingly.
        """
        super().__init__()
        
        self.txt_dir = os.path.join(C.TRAIN.SAMPLES.DIR, txt_dir)
        with open(txt_dir, 'rb') as f:
            self.data = f.splitlines
        self.data = np.loadtxt(txt_dir.as_posix(), dtype=np.uint32, delimiter=',')
        self.dataframe = pd.DataFrame(data, columns=['label', 'idx'])
        # self.pre_augmented = False
        self.cache_dir = Path(C.INT_DATA) / "srp/samples"
        

    def __getitem__(self, index):
        """Load an augmented version of the sample at `index`

        :param index: The index of a sample.
        """
        rec = self.dataframe.iloc[index]
        label = "pos" if rec.label == 1 else "neg"
        sample_name = "s{0:05d}".format(rec.idx)
        X, y = self._load_random_pickle(self.cache_dir / label / sample_name)

        return X, y

    def __len__(self):
        return len(self.dataframe)



def generate_variations():
    """Generate variations of the original patches. 

    This is used to pre-compute data augmentation for deep learning.
    
    The number and types of variation are controlled by the configuration file. 
    """
