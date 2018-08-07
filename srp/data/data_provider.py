""" Extract patches of data to use for training.

Rather than go to the original GIS data each time, we extract the input
ahead of time and save the data and annotations as pkl files.
"""
import os
import pickle
import time
from logging import debug, info

import numpy as np
import pandas as pd
import rasterio
from pylab import plt
from skimage.transform import rotate

from srp.config import C
from srp.data.orientedboundingbox import OrientedBoundingBox
from srp.util import tqdm


class Patch:
    # pylint:disable=too-few-public-methods,too-many-instance-attributes
    """Patch

    A pre-extracted portion of the input data.
    The values to predict are converted into pixel-locations (rather than meters)

    """

    def __init__(self, **kwargs):
        self.obb = kwargs.pop('obb', None)
        self.volumetric = kwargs.pop('volumetric', None)
        self.rgb = kwargs.pop('rgb', None)
        self.label = kwargs.pop('label', 0)
        self.dr_dc_angle = kwargs.pop('dr_dc_angle', (0, 0, 0))
        self.ori_xy = kwargs.pop('ori_xy', None)

        assert self.volumetric is not None
        assert self.rgb is not None
        assert self.ori_xy is not None

    def plot(self):
        height, width = self.rgb.shape[1:]
        extent = -height / 2, height / 2., -width / 2., width / 2.
        plt.subplot(131)
        plt.imshow(self.rgb.transpose(1, 2, 0), extent=extent)
        plt.title('rgb')
        plt.subplot(132)
        plt.imshow(self.rgb.transpose(1, 2, 0), extent=extent)
        plt.imshow(np.arctan(self.volumetric[1:4].transpose(1, 2, 0)), alpha=0.5, extent=extent)
        self.obb.plot(ls='--', lw=4.)
        plt.title('both')
        plt.subplot(133)
        plt.imshow(np.arctan(self.volumetric[1:4].transpose(1, 2, 0)), extent=extent)
        plt.title('volume')


class DataProvider(object):
    """DataProvider

    Generates patches from raster input sources (volume and RGB) based on annotations.

    Attributes:

        density_file:
            The name of a file with the volumetric point density information (default C.VOLUME.FILE)

        densities:
            The volumetric density dataset (an open rasterio datasource)

        color_file:
            The name of a file with the color imagery (default C.COLOR.FILE)

        colors:
            The color dataset (an open rasterio data source)

        radius:
            The patch radius (half the size of the input fed into the net). (default C.TRAIN.PATCH_SIZE)

        positive_csv_file:
            A file to hold the information on the locations of positive samples.
            Information include the location and orientation of the samples, data that is
            derived from the annotations.
            (default C.TAIN.SAMPLES.DIR / 'positives.csv')

        negative_csv_file:
            A file to hold the information on the locations of (valid) negative samples.
            (default C.TRAIN.SAMPLES.DIR / 'neagtives.csv')

        positive_sample_info:
            The actual  data on positive samples

        negative_sample_info:
            The actual data on negative samples.

    """

    def __init__(self, **kwargs):
        """Construct a data provider.

        This constructs a new data provider, which will read samples from large raster images and format them
        so that they can be fed to our network.

        :param density_file:
            The volumetric file (default C.VOLUME.FILE)
        :param color_file:
            The color file. (default C.COLOR.FILE)
        :param radius:
            Half the size of a patch. (default C.TRAIN.PATCH_SIZE / 2)

        """
        super().__init__()

        self.density_file = kwargs.pop('density_file', C.VOLUME.FILE)
        self.color_file = kwargs.pop('color_file', C.COLOR.FILE)
        self.radius = kwargs.pop('radius', C.TRAIN.PATCH_SIZE / 2)

        self.positive_csv_file = kwargs.pop('positive_csv_file', os.path.join(C.TRAIN.SAMPLES.DIR, 'positives.csv'))
        self.negative_csv_file = kwargs.pop('negative_csv_file', os.path.join(C.TRAIN.SAMPLES.DIR, 'negatives.csv'))

        if os.path.isfile(self.positive_csv_file):
            debug(f'Reading positive sample data from {self.positive_csv_file}')
            self.positive_sample_info = pd.read_csv(self.positive_csv_file).values
            debug(f'Read cached information on {len(self.positive_sample_info)} positive samples')
        else:
            debug(f'Not reading positive sample data from {self.positive_csv_file} because it does not exist (yet).')
            self.positive_sample_info = None

        if os.path.isfile(self.negative_csv_file):
            debug(f'Reading negative sample data from {self.negative_csv_file}')
            self.negative_sample_info = pd.read_csv(self.negative_csv_file).values
            debug(f'Read cached information on {len(self.negative_sample_info)} negative samples')
        else:
            debug(f'Not reading negative sample data from {self.negative_csv_file} because it does not exist (yet).')
            self.negative_sample_info = None

        debug('Opening the volumetric density file')
        start = time.process_time()
        self.densities = rasterio.open(self.density_file)
        end = time.process_time()
        debug(f'Opened the density dataset in {end-start} seconds')

        debug('Opening the color file')
        start = time.process_time()
        self.colors = rasterio.open(self.color_file)
        end = time.process_time()
        debug(f'Opened the color dataset in {end-start} seconds')

    def get_patch_xy(self, x, y, radius_in_pixels=None):
        """Get a patch from the input sources at the specified geographic x,y location.

        .. note::
              This function assumes that the inputs have been warped/reprojected to lie in the same
              geographic CRS.
              See https://github.com/mapbox/rasterio/blob/master/docs/topics/reproject.rst


        :param x: The x location, in the native CRS of the input source.
        :param y: The y location.
        :param radius_in_pixels: Half the size of a patch (in pixels). [default: self.radius]
        """

        if radius_in_pixels is None:
            radius_in_pixels = self.radius

        radius = int(radius_in_pixels)

        layers = self.densities.count
        size = 2 * radius

        # Determine the window in each dataset (resolutions may be different)
        c, r = np.asarray(self.densities.index(x, y))
        densities_window = ((r - radius, r + radius), (c - radius, c + radius))
        colors_window = self.colors.window(*self.densities.window_bounds(densities_window))

        # Extract the patches we are interested in
        densities = self.densities.read(
            window=densities_window, boundless=True, out=np.zeros((layers, size, size), dtype=np.uint16))
        colors = self.colors.read(
            (1, 2, 3), window=colors_window, boundless=True, out=np.zeros((3, size, size), dtype=np.uint8))

        # Convert sources to a common data type
        densities = densities.astype(np.float32)
        colors = colors.astype(np.float32) / 255.

        return np.concatenate((colors, densities))

    def get_patch_xyr(self, x, y, dx, dy, angle, radius_in_pixels=None):
        """ Get a rotated & transformed version of a patch.

        In  order to rotate the patch without issues at the corners we
        will extract a slightly larger patch, rotate it, and then crop it.

        x, y:
            coordinate of the patch (before transformation) in meters
        dx, dy:
            x and y offsets in pixel
        angle:
            in degrees, the additional rotation we apply on the image
        radius_in_pixels:
            half the width of the output patch

        return:
            a rotated cropped image
        """
        radius = int(radius_in_pixels)
        enlarged_radius = radius * 2

        dx, dy = int(dx), int(dy)

        source_patch = self.get_patch_xy(x, y, enlarged_radius)

        rotated_patch = source_patch.copy()

        for i, channel in enumerate(source_patch):
            rotated_patch[i] = rotate(channel, angle, preserve_range=True)

        # The center of the transformed patch is at location 2R -- we crop out a 2r by 2R patch
        # at the center.
        cropped_patch = rotated_patch[:, radius + dy:3 * radius + dy, radius - dx:3 * radius - dx]

        return cropped_patch

    def extract_all_patches(self):
        """Extract all patches for training and store them as files.

        """
        num_patches = len(self.positive_sample_info) + len(self.negative_sample_info)

        debug('Extracting patches')
        start = time.process_time()

        self.extract_positive_patches()
        self.extract_negative_patches()

        end = time.process_time()
        debug(f'Extracted {num_patches} patches in {end-start} seconds')

    def extract_positive_patches(self):
        """Extract the positive examples..

        """
        gsd = C.TRAIN.SAMPLES.GENERATOR.GSD
        progress = tqdm(self.positive_sample_info, desc='Generating positive patches')
        for i, row in enumerate(progress):
            center_x, center_y, original_angle, length, width = row

            directory = os.path.join(C.TRAIN.SAMPLES.DIR, 'pos', "s{0:05d}/".format(i))

            suffix = "s{0:05d}_orig.pickle".format(i)
            os.makedirs(directory, exist_ok=True)

            data = self.get_patch_xyr(center_x, center_y, 0, 0, -original_angle, radius_in_pixels=self.radius)
            obb = OrientedBoundingBox.from_rot_length_width((0, 0), 0, length / gsd, width / gsd)

            patch = Patch(
                obb=obb, volumetric=data[3:], rgb=data[:3], label=1, dr_dc_angle=(0, 0, 0), ori_xy=(center_x, center_y))

            with open(os.path.join(directory, suffix), 'wb') as handle:
                pickle.dump(patch, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_negative_patches(self):
        progress = tqdm(self.negative_sample_info, desc='Generating negative patches')
        for i, row in enumerate(progress):
            center_x, center_y = row
            directory = os.path.join(C.TRAIN.SAMPLES.DIR, 'neg', "s{0:05d}/".format(i))
            os.makedirs(directory, exist_ok=True)
            suffix = "s{0:05d}_orig.pickle".format(i)

            data = self.get_patch_xy(row[0], row[1], radius_in_pixels=self.radius)
            patch = Patch(
                obb=None,
                volumetric=data[3:],
                rgb=data[:3],
                label=0,
                ori_xy=(center_x, center_y),
                dr_dc_angle=(0, 0, 0))

            with open(os.path.join(directory, suffix), 'wb') as handle:
                pickle.dump(patch, handle, protocol=pickle.HIGHEST_PROTOCOL)


def extract_all_patches():
    provider = DataProvider()
    debug("This is a debug message")
    info("This is an info message")

    provider.extract_all_patches()


if __name__ == '__main__':
    extract_all_patches()
