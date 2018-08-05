import os
import numpy as np
import rasterio
import affine
import pickle
import glob
from srp.config import C
import pandas as pd
from srp.data.orientedboundingbox import OrientedBoundingBox
from collections import namedtuple
from skimage.transform import rotate
from tqdm import tqdm
from logging import info, debug
import time

Patch = namedtuple("Patch", ["obb", "volumetric", "rgb", "label", "dr_dc_angle", "ori_xy"])


class DataProvider(object):
    """DataProvider

    Generates patches from raster input sources (volume and RGB) based on annotations.

    Attributes:

        :density_file:
            The name of a file with the volumetric point density information (default C.VOLUME.FILE)

        :densities:
            The volumetric density dataset (an open rasterio datasource)

        :color_file:
            The name of a file with the color imagery (default C.COLOR.FILE)

        :colors:
            The color dataset (an open rasterio data source)

        :radius:
            The patch radius (half the size of the input fed into the net). (default C.TRAIN.PATCH_SIZE)

        :positive_csv_file:
            A file to hold the information on the locations of positive samples.
            Information include the location and orientation of the samples, data that is
            derived from the annotations.
            (default C.TAIN.SAMPLES.DIR / 'positive.csv')

        :negative_csv_file:
            A file to hold the information on the locations of (valid) negative samples.
            (default C.TRAIN.SAMPLES.DIR / 'neagtive.csv')

        :positive_sample_info:
            The actual  data on positive samples

        :negative_sample_info:
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

        self.positive_csv_file = kwargs.pop('positive_csv_file', os.path.join(C.TRAIN.SAMPLES.DIR, 'positive.csv'))
        self.negative_csv_file = kwargs.pop('negative_csv_file', os.path.join(C.TRAIN.SAMPLES.DIR, 'negative.csv'))

        if os.path.isfile(self.positive_csv_file):
            debug(f'Reading positive sample data from {self.positive_csv_file}')
            self.positive_sample_info=pd.read_csv(self.positive_csv_file).values
            debug(f'Read cached information on {len(self.positive_sample_info)} positive samples')
        else:
            debug(f'Not reading positive sample data from {self.positive_csv_file} because it does not exist (yet).')
            self.positive_sample_info=None

        if os.path.isfile(self.negative_csv_file):
            debug(f'Reading negative sample data from {self.negative_csv_file}')
            self.negative_sample_info=pd.read_csv(self.negative_csv_file).values
            debug(f'Read cached information on {len(self.negative_sample_info)} negative samples')
        else:
            debug(f'Not reading negative sample data from {self.nagative_csv_file} because it does not exist (yet).')
            self.negative_sample_info=None

        debug('Opening the volumetric density file')
        start=time.process_time()
        self.densities=rasterio.open(self.density_file)
        end=time.process_time()
        debug('Opened the density dataset in f{end-start} seconds')

        debug('Opening the color file')
        start=time.process_time()
        self.colors=rasterio.open(self.color_file)
        end=time.process_time()
        debug('Opened the color dataset in f{end-start} seconds')

    def get_patch_xy(self, x, y, radius_in_pixels=None):
        """Get a patch from the input sources at the specified geographic x,y location.

        NOTE: This function assumes that the inputs have been warped/reprojected to lie in the same
              geographic CRS.

              See https://github.com/mapbox/rasterio/blob/master/docs/topics/reproject.rst

              The x, y locations are in the common CRS.

        :param x: The x location, in the native CRS of the input source.
        :param y: The y location.
        :param radius_in_pixels: Half the size of a patch (in pixels). [default: self.radius]
        """

        radius=int(radius_in_pixels)

        layers=self.densities.meta['count']
        size=2 * radius

        # Determine the window in each dataset (resolutions may be different)
        c, r=np.asarray(~self.densities.transform * (x, y)).astype(int)
        densities_window=((r - radius, r + radius), (c - radius, c + radius))
        colors_window=self.colors.window(*self.densities.window_bounds(window))

        # Extract the patches we are interested in
        densities=self.densities.read(window=densities_window, boundless=True,
                                      out=np.zeros((layers, size, size), dtype=np.uint16))
        colors=self.colors.read((1, 2, 3), window=colors_window, boundless=True,
                                out=np.zeros((3, size, size), dtype=np.uint8))

        # Convert sources to a common data type
        densities=densities.astpe(np.float32)
        colors=colors.astype(np.float32) / 255.

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
        radius=radius_in_pixels
        enlarged_radiuys=radius * 2

        # TODO: Use and return an affine transform

        dx, dy=int(dx), int(dy)

        source_patch=self.get_patch_xy(x, y, enlarged_radius)

        rotated_patch=source_patch.copy()
        for i in range(len(source_patch)):
            rotated_patch[i]=rotate(source_patch[i], angle, preserve_range=True)

        # The center of the transformed patch is at location 2R -- we crop out a 2r by 2R patch
        # at the center.
        cropped_patch=rotated_patch[:, radius + dy:3 * radius + dy, radius - dx:3 * radius - dx]

        return cropped_patch

    def find_all_samples(self):
        progress=tqdm(total=len(self.positive_sample_info) +
                      len(self.negative_sample_info), desc='Generating Patches')
        self.find_positive_sample(radius_in_pixels=C.TRAIN.SAMPLES.GENERATOR.PATCH_SIZE)
        progress.update(len(self.positive_sample_info))
        self.find_nagative_sample(radius_in_pixels=C.TRAIN.SAMPLES.GENERATOR.PATCH_SIZE)
        progress.update(len(self.negative_sample_info))

    def find_positive_sample(self, radius_in_pixels=C.TRAIN.SAMPLES.GENERATOR.PATCH_SIZE):
        import pdb
        pdb.set_trace()
        for i, row in tqdm(enumerate(self.positive_sample_info), desc='Generating positive patches'):
            directory=os.path.join(C.POS_DATA, "s{0:05d}/".format(i))
            surffix="s{0:05d}_orig.pickle".format(i)
            os.makedirs(directory, mode=777, exist_ok=True)

            data=self.get_patch_xyr(row[0], row[1], 0, 0, (-row[2]), radius_in_pixels=radius_in_pixels)
            obb=OrientedBoundingBox.from_rot_length_width((0, 0), 0, row[3] / C.TRAIN.SAMPLES.GENERATOR.GSD,
                                                            row[4] / C.TRAIN.SAMPLES.GENERATOR.GSD)
            p=Patch(
                obb=obb, volumetric=data[3:], rgb=data[:3], label=1, dr_dc_angle=(0, 0, 0), ori_xy=(row[0], row[1]))

            with open(os.path.join(directory, surffix), 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def find_nagative_sample(self, radius_in_pixels=C.TRAIN.SAMPLES.GENERATOR.PATCH_SIZE):
        import pdb
        pdb.set_trace()
        for i, row in tqdm(enumerate(self.negative_sample_info).desc='Generating negative patches'):
            directory=os.path.join(C.NEG_DATA, "s{0:05d}/".format(i))
            surffix="s{0:05d}_orig.pickle".format(i)
            os.makedirs(directory, mode=777, exist_ok=True)

            data=self.get_patch_xy(row[0], row[1], radius_in_pixels=radius_in_pixels)
            p=Patch(
                obb=None, volumetric=data[3:], rgb=data[:3], label=0, ori_xy=(row[0], row[1]), dr_dc_angle=(0, 0, 0))

            with open(os.path.join(directory, surffix), 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
