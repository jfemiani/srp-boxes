# doctest: +NORMALIZE_WHITESPACE
"""
Process a vector file to produce positive and negative samples.
"""

from __future__ import division, print_function

import os

import fiona
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry as sg
from skimage.morphology import binary_dilation, disk

from srp.config import C
from srp.data.orientedboundingbox import OrientedBoundingBox
from srp.util import tqdm


class SampleGenerator:
    """SampleGenerator

    Generates patches with positive and negative examples for training.

    """

    def __init__(self,
                 rgb_path=C.COLOR.FILE,
                 volume_path=C.VOLUME.FILE,
                 annotation_path=C.ANNOTATIONS.FILE,
                 outdir=C.TRAIN.SAMPLES.DIR,
                 min_seperation=C.TRAIN.SAMPLES.GENERATOR.MIN_SEPARATION,
                 threshold=C.TRAIN.SAMPLES.GENERATOR.MIN_DENSITY,
                 total_num_of_samples=C.TRAIN.SAMPLES.GENERATOR.NUM_SAMPLES,
                 patch_size=C.TRAIN.PATCH_SIZE):
        """
        This class primarily reads its parameters from the 'config.py' file. Please specify your parameters there.
        Given a very high resolution ortho-image, a pre-processed LiDAR density file and human labeled '.geojson'
        (all in the same coordinate system), this class prepares two CSV files. The positive csv specifies the (x,y)
        coordinates, rotation angle (we constrain the angle within the first quadrant), length (x-axis vector length),
        and width (y-axis vector length) IN THAT ORDER. The negative CSV only provides the center coordinates (x,y)
        of each sample.

        rgb_path:
            path to the ortho-image
        volume_path:
            path to the volumetric data
        annotation_path:
            a .geojson file that labels the four corners of the box clock-wise
        outdir:
            where the .csv fies should be stored. This class can generate both positive and negative .csv files
        min_seperation:
            the minimum diameter around a positive center that we DO NOT draw negative samples from
        threshold:
            the minimum density count that the joined first and second tile above "fake ground" have to pass to count as
            an "useful" negative spot. We need to make sure the negative samples are draw from meaningful sites
            (exculding streets and such)
        total_num_of_samples:
            The the number of positive samples + negative samples. (default config.TRAIN.SAMPLES.GENERATOR.NUM_SAMPLES)
        """
        self.rgb_path = rgb_path
        self.volume_path = volume_path
        self.annotation_path = annotation_path
        self.min_seperation = min_seperation
        self.threshold = threshold
        self.bounds = None
        self.patch_size = patch_size

        suffix = C.VOLUME.CRS.replace(':', '')
        self.csv_dir = outdir
        self.positive_csv_file = '{}/positives_{}.csv'.format(self.csv_dir, suffix)
        self.negative_csv_file = '{}/negatives_{}.csv'.format(self.csv_dir, suffix)
        self.num_of_samples = total_num_of_samples
        with fiona.open(self.annotation_path) as vector_file:
            self.hotspots = np.array([f['geometry']['coordinates'] for f in vector_file if f['geometry'] is not None])

    def _get_tight_rectangle_info(self, box):
        """
        This is an internal use function. Given 4 human labeled corner points, this function returns the estimated
        rotation. length and width of the minimum_rotated_rectangle

        :param box:
            4 by 2 array (each row is a point).

        :return:
            returns center-x, center-y, rotation, length, width of the box

        Example:
            >>> sc = SampleGenerator()
            >>> sc._get_tight_rectangle_info(box=np.array([[0,0],
            ...                                         [0,1],
            ...                                         [2,1],
            ...                                         [2,0]])) # doctest: +NORMALIZE_WHITESPACE
            array([1. ,  0.5,  0. ,  2. ,  1. ])
        """
        p = sg.Polygon(box)
        return OrientedBoundingBox.rot_length_width_from_points(np.array(p.minimum_rotated_rectangle.exterior)[:-1])

    def make_pos_csv(self):
        """
        This file generates a ".csv" file for the positive samples. It specifies the center(x,y) coordinates,
        rotated angle, length, width IN THAT ORDER. It currently supports squares and rectangular inputs.
        """
        pos_samples = np.array([self._get_tight_rectangle_info(b) for b in self.hotspots])

        colnames = ['orig-x', 'orig-y', 'box-ori-deg', 'box-ori-length', 'box-ori-width']
        posdf = pd.DataFrame(data=pos_samples, columns=colnames)

        os.makedirs(os.path.dirname(self.positive_csv_file), exist_ok=True)
        posdf.to_csv(path_or_buf=self.positive_csv_file, index=False)

        print("Positive data .csv file saved as {}".format(self.positive_csv_file))

    def _read_densities(self):
        densities = rasterio.open(self.volume_path)
        colors = rasterio.open(self.rgb_path)

        self.bounds = tuple((max(densities.bounds.left, colors.bounds.left),
                             max(densities.bounds.bottom, colors.bounds.bottom),
                             min(densities.bounds.right, colors.bounds.right),
                             min(densities.bounds.top, colors.bounds.top)))

        window = densities.window(*self.bounds)
        stack = densities.read([3, 4], window=window, boundless=True)
        tfm = densities.window_transform(window)
        return stack, tfm

    def make_neg_csv(self):

        pos_xy = pd.read_csv(self.positive_csv_file).iloc[:, :2].values
        num_of_negs = self.num_of_samples - len(pos_xy)

        assert num_of_negs > 0

        progress = tqdm(total=3)

        progress.description = 'Loading the volumetric densities'

        import pdb
        pdb.set_trace()
        stack, tfm = self._read_densities()

        progress.update()

        # Only generate samples where there is point cloud data

        progress.set_description('Masking unusable regions')
        mask = stack.sum(0) > self.threshold

        # Make sure none of the negatives are too close to the positive regions
        positive_regions = np.zeros_like(mask)
        positive_regions[pos_xy] = True
        positive_regions = binary_dilation(positive_regions, disk(self.min_seperation))
        mask[positive_regions] = False

        # Mask out the edges of the image
        mask[:, :self.patch_size] = False
        mask[:, -self.patch_size:] = False
        mask[:self.patch_size, :] = False
        mask[-self.patch_size:, :] = False

        progress.update()

        progress.set_description('Sampling negative locations')
        neg_xy = np.argwhere(mask)
        neg_indices = np.random.choice(len(neg_xy), num_of_negs)
        neg_xy = neg_xy[neg_indices, ...]
        progress.update()

        progress.close()

        colnames = ['orig-x', 'orig-y']
        negdf = pd.DataFrame(data=neg_xy, columns=colnames)
        negdf.to_csv(path_or_buf=self.negative_csv_file, index=False)

        print("Negative data .csv file saved as {}".format(self.negative_csv_file))


def generate_samples():
    import srp.config
    print("Using config settings to generate samples")
    print("-" * 80)
    srp.config.dump()
    print("-" * 80)

    generator = SampleGenerator()
    generator.make_pos_csv()
    generator.make_neg_csv()


if __name__ == '__main__':
    generate_samples()
