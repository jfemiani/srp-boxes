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
import rasterio.errors
import rasterio.features
import rasterio.transform
import shapely.geometry

from srp.config import C
from srp.data.orientedboundingbox import OrientedBoundingBox
from srp.util import tqdm
from logging import info


class SampleGenerator:
    """SampleGenerator

    Generates patches with positive and negative examples for training.

    """

    def __init__(self, **kwargs):
        """
        This class primarily reads its parameters from the 'config.py' file. Please specify your parameters there.
        Given a very high resolution ortho-image, a pre-processed LiDAR density file and human labeled '.geojson'
        (all in the same coordinate system), this class prepares two CSV files. The positive csv specifies the (x,y)
        coordinates, rotation angle (we constrain the angle within the first quadrant), length (x-axis vector length),
        and width (y-axis vector length) IN THAT ORDER. The negative CSV only provides the center coordinates (x,y)
        of each sample.

        Arguments
        ---------
        rgb_path (str):
            path to the ortho-image
        volume_path (str):
            path to the volumetric data
        annotation_path (str):
            a .geojson file that labels the four corners of the box clock-wise
        outdir (str):
            where the .csv fies should be stored. This class can generate both positive and negative .csv files
        min_seperation (float):
            the minimum diameter around a positive center that we DO NOT draw negative samples from
        threshold (float0):
            the minimum density count that the joined first and second tile above "fake ground" have to pass to count as
            an "useful" negative spot. We need to make sure the negative samples are draw from meaningful sites
            (exculding streets and such)
        num_samples (int):
            The number of positive samples + negative samples. (default config.TRAIN.SAMPLES.GENERATOR.NUM_SAMPLES)
        density_layers (list of int):
            The layers of the volume densities that we use to decide if we should use that location as a sample.
        """

        self.rgb_path = kwargs.pop('rgb_path', C.COLOR.FILE)
        self.volume_path = kwargs.pop('volume_path', C.VOLUME.FILE)
        self.annotation_path = kwargs.pop('annotation_path', C.ANNOTATIONS.FILE)
        self.num_of_samples = kwargs.pop('num_samples', C.TRAIN.SAMPLES.GENERATOR.NUM_SAMPLES)
        self.min_seperation = kwargs.pop('min_seperation', C.TRAIN.SAMPLES.GENERATOR.MIN_SEPARATION)
        self.threshold = kwargs.pop('threshold', C.TRAIN.SAMPLES.GENERATOR.MIN_DENSITY)
        self.patch_size = kwargs.pop('patch_size', C.TRAIN.PATCH_SIZE)
        self.csv_dir = kwargs.pop('outdir', C.TRAIN.SAMPLES.DIR)
        self.density_layers = kwargs.pop('density_layers', C.TRAIN.SAMPLES.GENERATOR.DENSITY_LAYERS)

        self.sample_name_pattern = kwargs.pop('sample_pattern', C.TRAIN.SAMPLES.GENERATOR.NAME_PATTERN)

        self.positive_csv_file = os.path.join(self.csv_dir, 'positives.csv')
        self.negative_csv_file = os.path.join(self.csv_dir, 'negatives.csv')

        self.bounds = None

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
            ...                                            [0,1],
            ...                                            [2,1],
            ...                                            [2,0]])) # doctest: +NORMALIZE_WHITESPACE
            array([1. ,  0.5,  0. ,  2. ,  1. ])
        """
        poly = shapely.geometry.Polygon(box)
        points = np.array(poly.minimum_rotated_rectangle.exterior)[:-1]
        return OrientedBoundingBox.rot_length_width_from_points(points)

    def make_pos_csv(self):
        """
        This file generates a ".csv" file for the positive samples. It specifies the center(x,y) coordinates,
        rotated angle, length, width IN THAT ORDER. It currently supports squares and rectangular inputs.
        """
        with tqdm(self.hotspots, desc='Generating positive samples') as progress:
            pos_samples = []
            for i, b in enumerate(progress):
                x, y, deg, length, width = self._get_tight_rectangle_info(b)
                rel_path = self.sample_name_pattern.format(label='pos', index=i + 1)
                pos_samples.append([rel_path, x, y, deg, length, width])

            colnames = ['name', 'orig-x', 'orig-y', 'box-ori-deg', 'box-ori-length', 'box-ori-width']
            posdf = pd.DataFrame(data=pos_samples, columns=colnames)

            os.makedirs(os.path.dirname(self.positive_csv_file), exist_ok=True)
            posdf.to_csv(path_or_buf=self.positive_csv_file, index=False)

        info("Positive data .csv file saved as {}".format(self.positive_csv_file))

    def make_neg_csv(self):
        """Generate a CSV file with the locations of some negative examples.

        We look through the volume, and choose samples that are "interesting"
        in the sense that they include points at the hight-layers where
        we expect to see boxes.
        """

        pos_xy = pd.read_csv(self.positive_csv_file).iloc[:, 1:1 + 2].values
        num_of_negs = self.num_of_samples - len(pos_xy)

        assert num_of_negs > 0

        # Generate region that should not be negative
        positive_region = shapely.geometry.MultiPoint(points=pos_xy)
        positive_region = positive_region.buffer(self.min_seperation)

        densities = rasterio.open(self.volume_path)
        colors = rasterio.open(self.rgb_path)

        self.bounds = tuple((max(densities.bounds.left, colors.bounds.left),
                             max(densities.bounds.bottom, colors.bounds.bottom),
                             min(densities.bounds.right, colors.bounds.right),
                             min(densities.bounds.top, colors.bounds.top)))

        combined_window = densities.window(*self.bounds)

        block_size = C.TRAIN.SAMPLES.GENERATOR.BLOCK_SIZE

        rowcols = np.mgrid[0:densities.shape[0]:block_size, 0:densities.shape[1]:block_size].reshape(2, -1)
        block_windows = [rasterio.windows.Window(row, col, block_size, block_size) for row, col in rowcols.T]

        neg_xy = []

        with tqdm(block_windows, desc="Processing volume data blocks") as progress:
            for window in progress:
                try:
                    overlapping_window = combined_window.intersection(window)
                    stack = densities.read([3, 4], window=overlapping_window, boundless=True)
                    tfm = densities.window_transform(overlapping_window)
                except rasterio.errors.WindowError as e:
                    # Non overlapping window: windows do not intersect"
                    continue

                density_mask = stack.sum(0) > self.threshold
                positive_mask = rasterio.features.geometry_mask(positive_region, stack.shape[1:], tfm)
                sample_mask = density_mask & positive_mask
                ij = np.argwhere(sample_mask.T)
                if len(ij):
                    xy = np.c_[tfm * ij.T]
                    neg_xy.append(xy)
            neg_xy = np.concatenate(neg_xy)

        # Choose which samples to keep
        indices = np.random.choice(len(neg_xy), num_of_negs)
        neg_xy = neg_xy[indices]

        samples = [[self.sample_name_pattern.format(label='neg', index=i), x, y]
                   for i, (x, y) in enumerate(tqdm(neg_xy))]

        colnames = ['name', 'orig-x', 'orig-y']
        negdf = pd.DataFrame(data=samples, columns=colnames)
        negdf.to_csv(path_or_buf=self.negative_csv_file, index=False)

        info("Negative data .csv file saved as {}".format(self.negative_csv_file))


def generate_samples():
    generator = SampleGenerator()
    generator.make_pos_csv()
    generator.make_neg_csv()


if __name__ == '__main__':
    generate_samples()
