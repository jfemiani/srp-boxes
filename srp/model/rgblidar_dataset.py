import pickle
import os
import numpy as np
from srp.config import C
from glob import glob
from srp.data.orientedboundingbox import OrientedBoundingBox
from torch.utils.data import Dataset
from srp.data.generate_patches import Patch


class RgbLidarDataset(Dataset):
    """An Rgb+Lidar Combined Dataset
    
    This dataset reads in patches that were saved as pickle files. 
    
    """
    def __init__(self, train_or_test, cache_dir=None):
        """
        This class is design to use in conjunction with the pytorch dataloader class
        
        Attributes:
        train_or_test: string 'train' or 'test' representing train or test dataset
        cache_dir: dir to the root of samples, default is C.TRAIN.SAMPLES.DIR. This dataset class load samples
                   within this directory.
        
        Example:

        >>> trn_data = RgbLidarDataset('train')
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
        assert isinstance('train', str)
        
        self.cache_dir = cache_dir or C.TRAIN.SAMPLES.DIR
        self.txt_dir = os.path.join(C.TRAIN.SAMPLES.DIR, 
                                    'fold{}'.format(C.TRAIN.SAMPLES.CURRENT_FOLD),
                                    '{}.txt'.format(train_or_test))
        
        with open(self.txt_dir, 'r') as f:
            self.dataset = f.read().splitlines()
        
        
    def _load_random_pickle(self, dirname):
        patches = glob(os.path.join(dirname, '*.pkl'))
        assert len(patches) == C.TRAIN.AUGMENTATION.VARIATIONS
        with open(np.random.choice(patches), 'rb') as handle:
            p = pickle.load(handle)
        
        X = np.concatenate((p.rgb, p.volumetric))
        points = p.obb.points() if p.label else np.zeros((4,2)) 
        return X, (p.label, points)
        
        
    def __getitem__(self, index):
        """Load an augmented version of the sample at `index`

        :param index: The index of a sample.
        """
        rec = self.dataset[index]
        label = os.path.dirname(rec)
        print(rec, label)
        idx = os.path.basename(rec).split('.')[0]
        
        dirname = os.path.dirname(C.TRAIN.AUGMENTATION.NAME_PATTERN).format(label=label, name=idx)
        
        return self._load_random_pickle(os.path.join(self.cache_dir, dirname))


    def __len__(self):
        return len(self.dataset)
    