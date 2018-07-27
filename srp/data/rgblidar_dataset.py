import pickle
import pandas as pd
import srp.config as C
import numpy as np
from srp.data.data_augment import DataAugment
from pathlib import Path
from torch.utils.data import Dataset

class RgbLidarDataset(Dataset):
    def __init__(self, txt_dir):
        """
        
        txt_dir: a PosixPath object specifying the path of the .txt which stores train (or test) sample indices.  
        """
        super().__init__()
        data = np.loadtxt(txt_dir.as_posix(), dtype=np.uint32, delimiter=',')
        self.df = pd.DataFrame(data, columns=['label', 'index'])
        # self.data = data
        # self.transform = transform
        self.pre_augmented = False
        self.cache_dir = Path(C.INT_DATA)/"samples"
        self.num_variants = C.NUM_SAMPLES
    
    def pre_augment(self, cache_dir=None, num_variants=C.NUM_SAMPLES, force_recompute=False):
        """Precompute the data augmentation and cache it. 
        
        :param cach_dir: A mirror of the TRAINVAL folder that will hold variations on each input
        :param num_variants: The number of variants of each image to produce
        :param force_recompute: Whether the cach should be replaced (True) or reused if present (False)
        """
        cache_dir = cache_dir or self.cache_dir
        da = DataAugment(cache_dir)
        da.make_next_batch_variations(cache_dir, 20)
        self.pre_augmented = True
        self.num_variants = num_variants
    
    def _load_random_pickle(path):
        ppath = list(path.glob("*_var*.pickle"))[np.random.randint(C.NUM_PRECOMPUTE_VARIATION)]
        with open(ppath.as_posix(), 'rb') as handle:
            p = pickle.load(handle)
        if p.obb:
            angle =  np.degrees(np.arctan2(p.obb.uy, p.obb.ux))
            y = np.array([p.obb.cx, p.obb.cy, angle, 2*p.obb.ud, 2*p.obb.vd])
        else:
            y = np.zeros((5))
        return (p.rgb.transpose(1,2,0), p.volumetric.transpose(1,2,0)), (p.label, y)
    
    def __getitem__(self, index):
        rec = self.df.iloc[index]
        # if self.pre_augmented:
        label = "pos" if rec.label == 1 else "neg"
        sindex = "s{0:05d}".format(rec.index)
        X, y = self._load_random_pickle(self.cache_dir/label/sindex)
#         else:
#             cache_path = self.cache_dir/Path(rec.id)
#             j = randint(self.num_variants)
#             variant_path = cache_path/f"{j}.npz"
#             npz = np.load(variant_path)
#             x = npz['x']
#             y = npz['y']
        
        return X, y
    
    def __len__(self):
        return len(self.data)