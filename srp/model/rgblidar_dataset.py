import pickle
import affine
import pandas as pd
import srp.config as C
import sys
import numpy as np
from srp.data.orientedboundingbox import OrientedBoundingBox
from pathlib import Path
from tqdm import tqdm
from collections import namedtuple
from torch.utils.data import Dataset
from skimage.transform import rotate

Patch = namedtuple("Patch", ["obb", "volumetric", "rgb", "label", "dr_dc_angle", "ori_xy"])


class RgbLidarDataset(Dataset):
    def __init__(self, txt_dir, prop_synthetic=0):
        """
        This class is design to use in conjunction with the pytorch dataloader class
        txt_dir: a PosixPath object which specify the path of the .txt which stores train (or test) sample indices.

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
        data = np.loadtxt(txt_dir.as_posix(), dtype=np.uint32, delimiter=',')
        self.dataframe = pd.DataFrame(data, columns=['label', 'idx'])
        # self.pre_augmented = False
        self.cache_dir = Path(C.INT_DATA) / "srp/samples"
        self.num_variants = C.NUM_PRECOMPUTE_VARIATION
        self.prop_synthetic = prop_synthetic

    def _get_orig_path(self, top, isbox, idx):
        """Helper function to get the path to the original version of a sample.

        :param top: The root folder for the dataset.
        :param isbox: The label associated with the sample (True=positive, False=negative)
        :param idx: The index of the sample.
        """
        label_dir = "pos" if isbox else "neg"
        idx_dir = "s{0:05d}".format(idx)
        return top / label_dir / idx_dir / "{}_orig.pickle".format(idx_dir)

    def pre_augment(self, cache_dir=None, num_variants=C.NUM_PRECOMPUTE_VARIATION):
        """Precompute the data augmentation and cache it.

        :param cach_dir: A mirror of the TRAINVAL folder that will hold variations on each input
        :param num_variants: The number of variants of each image to produce
        """
        cache_dir = cache_dir or self.cache_dir

        for i in tqdm(range(len(self.dataframe)), desc='augmenting', file=sys.stdout, leave=False):
            rec = self.dataframe.iloc[i]
            orig_dir = self._get_orig_path(cache_dir, rec.label, rec.idx).as_posix()
            with open(orig_dir, 'rb') as handle:
                p = pickle.load(handle)
                for i in range(self.num_variants):
                    var = self._augment(p, radius=C.PATCH_SIZE / 2)
                    with open(orig_dir.replace("orig", "var{0:02d}".format(i + 1)), 'wb') as handle:
                        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _augment(self, p, radius):
        """Generate an augmented version of patch `p`.

        :param p: The original patch.
        :param radius: The radius for the augmented patch (typically smaller
                       to accomodate rotation and cropping)
        """
        radius = int(radius)
        dr = int(np.random.uniform(-1, 1) * C.MAX_OFFSET)
        dc = int(np.random.uniform(-1, 1) * C.MAX_OFFSET)
        rotate_angle = np.random.rand() * 360
        p_center = int(p.volumetric.shape[1] / 2)

        source_patch = np.concatenate((p.rgb, p.volumetric))
        rotated_patch = np.zeros((source_patch.shape))

        if np.random.rand() <= self.prop_synthetic:
            # TODO: Generate synthetic data
            pass

        for i in range(len(source_patch)):
            rotated_patch[i] = rotate(source_patch[i], rotate_angle, preserve_range=True)

            cropped_patch = rotated_patch[:, p_center - radius + dc:p_center + radius + dc, p_center - radius -
                                          dr:p_center + radius - dr]
            rgb = cropped_patch[:3]
            volumetric = cropped_patch[3:]
            obb = p.obb

        if p.label:
            R = affine.Affine.rotation(rotate_angle)
            T = affine.Affine.translation(dr, dc)
            A = T * R

            after = np.vstack(A * p.obb.points().T).T
            obb = OrientedBoundingBox.from_points(after)

        return Patch(
            obb=obb,
            ori_xy=p.ori_xy,
            rgb=cropped_patch[:3],
            label=p.label,
            volumetric=cropped_patch[3:],
            dr_dc_angle=(dr, dc, rotate_angle))

    def _load_random_pickle(self, sample_path):
        """Load an augmented version of the sample `sample_path`

        :param sample_path: The path to a sample.
        """
        i = np.random.randint(C.NUM_PRECOMPUTED_VARIATIONS)
        # print (sample_path.name)
        sample_name = sample_path.name
        variation = '{}_var{:02}.pickle'.format(sample_name, i + 1)
        ppath = sample_path / variation
        with open(ppath.as_posix(), 'rb') as handle:
            p = pickle.load(handle)
        if p.obb:
            y = (1, p.obb.points())
        else:
            y = (0, np.zeros((4, 2)))
        return np.concatenate((p.rgb, p.volumetric)), y

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
