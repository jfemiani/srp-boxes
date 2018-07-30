# doctest: +NORMALIZE_WHITESPACE
"""
Process a vector file to produce positive and negative samples.
"""

from __future__ import division
from __future__ import print_function
import fiona
from srp.data.orientedboundingbox import OrientedBoundingBox as oo
from shapely.ops import cascaded_union
import numpy as np
import shapely.geometry as sg
import pandas as pd
import rasterio
from os import path
import random
import srp.config as C

class SampleCSV:
    def __init__(self,
                 rgb_path=C.COLOR_PATH,
                 volume_path=C.VOLUMETRIC_PATH,
                 annotation_path=C.ANNOTATION_PATH,
                 outdir=C.CSV_DIR,
                 min_seperation=C.MIN_SEPARATTION,
                 threshold=C.MIN_DENSITY_COUNT,
                 total_num_of_samples=C.NUM_SAMPLES
                 ):
    """
    This class primarily reads its parameters from the 'config.py' file. Please specify your parameters there.
    Given a very high resolution orthograph, a pre-processed lidar density file and human labeled '.geojson' 
    (all in the same coordinate system), this class prepares two csv files. The positive csv specifies the (x,y)
    coordinates, rotation angle (we constrain the angle within the first quadrant), length (x-axis vector length),
    and width(y-axis vector length) IN THAT ORDER. The negative .csv only provides the center coordinates (x,y)
    of each sample.
    
    rgb_path: path to the orthograph
    volume_path: path to the volumetric data
    annotation_path: a .geojson file that labels the four corners of the box clock-wise
    outdir: where the .csv fies should be stored. This class can generate both positive and negative .csv files
    min_seperation: the minimum diameter around a positive center that we DO NOT draw negative samples from
    threshold: the minimum density count that the joined first and second tile above "fake ground" have to pass to count 
               as an "useful" negative spot. We need to make sure the negative samples are draw from meaningful sites 
               (exculding streets and such)
    total_num_of_samples: the the number of positive samples + negative samples. (default config.NUM_SAMPLES)
    """
        self.rgb_path = rgb_path
        self.volume_path = volume_path
        self.annotation_path = annotation_path
        self.min_seperation = min_seperation
        self.threshold = threshold
        
        suffix = C.VOLUME_DEFAULT_CRS.replace(':','')
        self.csv_dir = outdir
        self.posdir = '{}/positives_{}.csv'.format(self.csv_dir, suffix)
        self.negdir = '{}/negatives_{}.csv'.format(self.csv_dir, suffix)
        self.num_of_samples = total_num_of_samples
        with fiona.open(self.annotation_path) as vectorFile:
            self.hotspots = np.array([
                f['geometry']['coordinates'] for f in vectorFile
                if f['geometry'] is not None
            ])
        

    def _get_tight_rectangle_info(self, box):
        # doctest: +NORMALIZE_WHITESPACE
        """
        
        This is an internal use function. Given 4 human labeled corner points, this function returns the estimated rotation.
        length and width of the minimum_rotated_rectangle
        param box: 
        Return: returns center-x, center-y, rotation, length, width of the box
        Example:
            >>> sc = SampleCSV()
            >>> sc._get_tight_rectangle_info(box=np.array([[0,0],
            ...                                         [0,1],
            ...                                         [2,1],
            ...                                         [2,0]])) # doctest: +NORMALIZE_WHITESPACE
            array([ 1. ,  0.5,  0. ,  2. ,  1. ])
        """
        p = sg.Polygon(box)
        return oo.get_rot_length_width_from_points(np.array(p.minimum_rotated_rectangle.exterior)[:-1])

    def make_pos_csv(self):
        """
        This file generates a ".csv" file for the positive samples. It specifies the center(x,y) coordinates,
        rotated angle, length, width IN THAT ORDER. It currently supports squares and rectangular inputs.
        """
        pos_samples = np.array(
            [self._get_tight_rectangle_info(b) for b in self.hotspots])
        
        colnames = [
            'orig-x', 'orig-y', 'box-ori-deg', 'box-ori-length',
            'box-ori-width'
        ]
        posdf = pd.DataFrame(data=pos_samples, columns=colnames)
        posdf.to_csv(path_or_buf=self.posdir, index=False)
        
        print("Positive data .csv file saved as {}".format(self.posdir))
        
    def _read_densities(self):
        
        densities = rasterio.open(self.volume_path)
        colors = rasterio.open(self.rgb_path)
        
        self.bounds = tuple((max(densities.bounds.left, colors.bounds.left),
                             max(densities.bounds.bottom, colors.bounds.bottom),
                             min(densities.bounds.right, colors.bounds.right),
                             min(densities.bounds.top, colors.bounds.top)))
        
        window = densities.window(*self.bounds)
        stack = densities.read([3,4], window=window, boundless=True)
        tfm = densities.window_transform(window)
        return stack, tfm
    
    
    def _get_batch_negs(self, mask, tfm, positive_regions, w=500):
        # vol is an np array
        negs = []
        for r in range(0, mask.shape[0], w):
            for c in range(0, mask.shape[1], w):
                rows, cols = np.where(mask[r:r+w, c:c+w])
                if len(rows) == 0:        # no positive sample
                    continue
                for i in range(10):
                    idx = np.random.randint(rows.shape[0])         # np.random.choice(len(rows))
                    coords = tfm * np.array([c + cols[idx], r + rows[idx]]) # (row, col) flipped
                    if not positive_regions.contains(sg.Point(coords)):
                        negs.append(coords)
                        break
        return np.array(negs)
    
        
    def make_neg_csv(self):
        pos_xy = pd.read_csv(self.posdir).iloc[:,:2].values
        num_of_negs = self.num_of_samples - len(pos_xy)
        
        assert num_of_negs > 0
        
        stack, tfm = self._read_densities()
        positive_regions = cascaded_union([sg.Point(*center.T).buffer(self.min_seperation) for center in pos_xy])        
        
        mask = stack.sum(0) > C.MIN_DENSITY_COUNT
        mask[:, :C.PATCH_SIZE] = False
        mask[:, -C.PATCH_SIZE:] = False
        mask[:C.PATCH_SIZE, :] = False
        mask[-C.PATCH_SIZE:, :] = False
        
        negs = self._get_batch_negs(mask, tfm, positive_regions, w=500)
        while len(negs) < num_of_negs:
            np.vstack((negs, self._get_batch_negs(mask, tfm, positive_regions, w=500)))
        
        negs = negs[np.random.choice(negs.shape[0], size=num_of_negs, replace=False)]
        
        colnames = ['orig-x', 'orig-y']
        negdf = pd.DataFrame(data=negs, columns=colnames)
        negdf.to_csv(path_or_buf=self.negdir, index=False)
        print("Negative data .csv file saved as {}".format(self.negdir))
