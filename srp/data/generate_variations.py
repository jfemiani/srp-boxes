import os
import pickle
import skimage.filters
import affine
import glob
import scipy
import numpy as np
from skimage.transform import rotate
from logging import debug
from srp.data.generate_patches import Patch
from srp.config import C
from srp.data.orientedboundingbox import OrientedBoundingBox
from srp.util import tqdm


class VariationMaker(object):
    """DataAugment

    Generates variations(including synthetic) from patches generated by module 'generate_patches.py' randomly.

    Attributes:

        variations:
            The number of variations to generate (default C.TRAIN.AUGMENTATION.VARIATIONS)

        max_offset:
            The maximum offset for augmentation, both dx and dy (default C.TRAIN.AUGMENTATION.MAX_OFFSET)

        radius:
            The size of the output patches (default is C.TRAIN.PATCH_SIZE/2)

        synthetic_prop:
            The probability of making synthetic data that lines up perfectly with the expected result
            (default C.TRAIN.AUGMENTATION.SYNTHETIC_PROBABILITY)

        current_fold:
            A integer value specifing fold that is being used now (default C.TRAIN.SAMPLES.CURRENT_FOLD)

        cache_root: the root where all subfolders will reside (default is C.TRAIN.SAMPLES.DIR)


    """

    def __init__(self, **kwargs):
        super().__init__()

        self.variations = kwargs.pop('variations', C.TRAIN.AUGMENTATION.VARIATIONS)
        self.max_offset = kwargs.pop('max_offset', C.TRAIN.AUGMENTATION.MAX_OFFSET)
        self.radius = kwargs.pop('radius', C.TRAIN.PATCH_SIZE / 2)
        self.synthetic_prop = kwargs.pop('synthetic_prop', C.TRAIN.AUGMENTATION.SYNTHETIC_PROBABILITY)
        self.cache_root = kwargs.pop('cache_root', C.TRAIN.SAMPLES.DIR)

    def _cropped_rotate_patch(self, source_patch, rothate_angle, p_center, dr, dc):
        rotated_patch = np.zeros((source_patch.shape))
        for i in range(len(source_patch)):
            rotated_patch[i] = rotate(source_patch[i], rotate_angle, preserve_range=True)

        cropped_patch = rotated_patch[:, p_center - radius + dc:p_center + radius + dc, p_center - radius -
                                      dr:p_center + radius - dr]
        return cropped_patch

    def _fake_positive_layer(self, obb, radius, edge_factor=1, sigma=12, fg_noise=0.1, bg_noise=0.1):
        diameter = int(radius * 2)
        square = np.zeros((diameter, diameter))

        cd = obb.u_length
        rd = obb.v_length
        square[int(radius - rd):int(radius + rd), int(radius - cd):int(radius + cd)] = 1

        outline = scipy.ndimage.morphology.morphological_gradient(square, 3)
        outline[int(radius - rd):, int(radius - cd):int(radius + cd)] = 0
        square = (1 - edge_factor) * square + edge_factor * outline

        gradient = np.zeros_like(square)
        gradient[:64] = 1
        gradient = skimage.filters.gaussian(gradient, sigma=sigma)
        square *= gradient
        square /= np.percentile(square.flat, 99.9)

        background = square == 0
        noisy = square
        noisy += background * np.random.randn(diameter, diameter) * bg_noise
        noisy += ~background * np.random.randn(diameter, diameter) * fg_noise
        noisy = noisy.clip(0, 1)

        return noisy

    def _fake_data(self, obb, radius=C.TRAIN.PATCH_SIZE / 2):
        radius = int(C.TRAIN.SAMPLES.GENERATOR.PADDED_PATCH_SIZE / 2)
        data = np.zeros((6, 2 * radius, 2 * radius))
        data[2] = self._fake_positive_layer(obb, radius, edge_factor=1)
        data[3] = self._fake_positive_layer(obb, radius, edge_factor=0.7)
        data[3] *= 0.3
        data *= 40
        return data

    def _augment(self, p, radius):
        """Generate an augmented version of patch `p`.

        :param p: The original patch.
        :param radius: The radius for the augmented patch (typically smaller
                       to accomodate rotation and cropping)
        """
        radius = int(radius)
        dr = int(np.random.uniform(-1, 1) * C.TRAIN.AUGMENTATION.MAX_OFFSET)
        dc = int(np.random.uniform(-1, 1) * C.TRAIN.AUGMENTATION.MAX_OFFSET)
        rotate_angle = np.random.rand() * 360
        p_center = int(p.volumetric.shape[1] / 2)

        vol = p.volumetric

        if p.label and np.random.random() <= self.synthetic_prop:
            vol = self._fake_data(p.obb, C.TRAIN.SAMPLES.GENERATOR.PADDED_PATCH_SIZE)
        assert vol.shape[1:] == p.rgb.shape[1:]

        source_patch = np.concatenate((p.rgb, vol))
        rotated_patch = np.zeros((source_patch.shape))
        obb = p.obb

        for i in range(len(source_patch)):
            rotated_patch[i] = rotate(source_patch[i], rotate_angle, preserve_range=True)

        cropped_patch = rotated_patch[:, p_center - radius + dc:p_center + radius + dc, p_center - radius -
                                      dr:p_center + radius - dr]

        if p.label:
            R = affine.Affine.rotation(rotate_angle)
            T = affine.Affine.translation(dr, dc)
            A = T * R

            after = np.vstack(A * p.obb.points().T).T
            obb = OrientedBoundingBox.from_points(after)

        return Patch(
            name=p.name,
            obb=obb,
            ori_xy=p.ori_xy,
            rgb=cropped_patch[:3],
            label=p.label,
            volumetric=cropped_patch[3:],
            dr_dc_angle=(dr, dc, rotate_angle))

    def make_variations(self, patch):
        """Precompute the variations using original patches in designated cache_dir.

        This is used to pre-compute data augmentation for deep learning.

        The number and types of variation are controlled by the configuration file.
        :param patch: an original 'Patch' class object
        """
        label, name = patch.name.split('/')[-2:]
        name = name.split('.')[0]
        for i in range(self.variations):
            var = self._augment(patch, radius=C.TRAIN.PATCH_SIZE / 2)

            var_name = os.path.join(self.cache_root,
                                    C.TRAIN.AUGMENTATION.NAME_PATTERN.format(label=label, name=name, var_idx=i + 1))

            os.makedirs(os.path.dirname(var_name), exist_ok=True)
            with open(var_name, 'wb') as handle:
                pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_variations(names=None, synthetic_prop=C.TRAIN.AUGMENTATION.SYNTHETIC_PROBABILITY, cache_root=None):
    """
    :param names: a list of names relative to the C.TRAIN.SAMPLES.DIR (default is all patches pos + neg)
    :param cache_root: the root where all subfolders will reside (default is C.TRAIN.SAMPLES.DIR)
    """
    cache_root = cache_root or C.TRAIN.SAMPLES.DIR
    if names:
        samples = [os.path.join(C.TRAIN.SAMPLES.DIR, n) for n in list(names)]
    else:
        samples = glob.glob(os.path.join(C.TRAIN.SAMPLES.DIR, '*/*.pkl'))

    maker = VariationMaker(synthetic_prop=synthetic_prop, cache_root=cache_root)
    progress = tqdm(samples, desc='Generating variation patches')

    for i, name_dir in enumerate(progress):
        with open(os.path.join(C.TRAIN.SAMPLES.DIR, name_dir), 'rb') as handle:
            p = pickle.load(handle)

        maker.make_variations(p)


if __name__ == '__main__':
    generate_variations()
