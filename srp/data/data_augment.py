import os
import numpy as np
import rasterio
import affine
import pickle
import glob
import srp.config as C
import pandas as pd
from srp.data.orientedboundingbox import OrientedBoundingBox
from collections import namedtuple
from skimage.transform import rotate

Patch = namedtuple("Patch",["obb", "volumetric", "rgb", "label","dr_dc_angle", "ori_xy"])

class DataAugment:
    def __init__(self, radius=C.PATCH_SIZE/2):
        self.densityfile = C.VOLUMETRIC_PATH
        self.colorfile = C.COLOR_PATH
        self.radius = radius
        self.poscsvdir = os.path.join(C.CSV_DIR, 'positives_{}.csv'.format(C.VOLUME_DEFAULT_CRS.replace(':', '')))
        self.negcsvdir = os.path.join(C.CSV_DIR, 'negatives_{}.csv'.format(C.VOLUME_DEFAULT_CRS.replace(':', '')))
        self.posinfo = pd.read_csv(self.poscsvdir).values
        self.neginfo = pd.read_csv(self.negcsvdir).values
        
        self.densities = rasterio.open(self.densityfile)
        self.colors = rasterio.open(self.colorfile)
    
    def get_patch_xy(self, x, y, radius_in_pixels=None):
        R = int(radius_in_pixels)
        c, r = np.asarray(~self.densities.transform*(x, y)).astype(int)
        
        window = ((r-R, r+R), (c-R, c+R))
        bounds = self.densities.window_bounds(window)
        
        densities = self.densities.read(window=window,
                                        boundless=True,
                                        out=np.zeros((self.densities.meta['count'], 2*R, 2*R),
                                                     dtype=np.uint16)).astype(np.float32)
        colors = self.colors.read((1,2,3),
                                  window=self.colors.window(*bounds),
                                  boundless=True,
                                  out=np.zeros((3, 2*R, 2*R), dtype=np.uint8)).astype(np.float32)/255.

        return np.concatenate((colors, densities)) 
    
    def get_patch_xyr(self, x, y, dx, dy, angle, radius_in_pixels=None):
        """
        
        
        x, y: coordinate in the original crs(epsg:26949) in meters
        dx, dy: x and y offsets in pixel
        angle: in degrees, the additional rotation we apply on the image
        radius_in_pixels: half the width of the output patch
        
        return: a rotated cropped image 
        """
        R = radius_in_pixels
        dx, dy = int(dx), int(dy)
        source_patch = self.get_patch_xy(x, y, R * 2)
        rotated_patch = source_patch.copy()
        for i in range(len(source_patch)):
            rotated_patch[i] = rotate(source_patch[i], angle,
                                                        preserve_range=True)

        cropped_patch = rotated_patch[:, R + dy: 3*R + dy, R - dx : 3 * R - dx]
        
        return cropped_patch
    def make_originals(self):
        self.make_positive_originals(radius_in_pixels=C.PATCH_SIZE)
        self.make_negative_originals(radius_in_pixels=C.PATCH_SIZE)
    
    def make_positive_originals(self, radius_in_pixels=C.PATCH_SIZE):
        for i, row in enumerate(self.posinfo):
            directory = os.path.join(C.POS_DATA, "s{0:05d}/".format(i))
            surffix = "s{0:05d}_orig.pickle".format(i)
            os.makedirs(directory,mode=777, exist_ok=True)
            
            data = self.get_patch_xyr(row[0], row[1], 0, 0, (-row[2]), radius_in_pixels=radius_in_pixels)
            obb = OrientedBoundingBox.from_rot_length_width((0,0), 
                                                            0, 
                                                            row[3]/C.METERS_PER_PIXEL, 
                                                            row[4]/C.METERS_PER_PIXEL)
            p = Patch(obb=obb, volumetric=data[3:], rgb=data[:3],label=1, dr_dc_angle=(0,0,0), ori_xy=(row[0],row[1]))
            
            with open(os.path.join(directory, surffix), 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def make_negative_originals(self, radius_in_pixels=C.PATCH_SIZE):
        for i, row in enumerate(self.neginfo):
            directory = os.path.join(C.NEG_DATA, "s{0:05d}/".format(i))
            surffix = "s{0:05d}_orig.pickle".format(i)
            os.makedirs(directory,mode=777, exist_ok=True)
            
            data = self.get_patch_xy(row[0], row[1], radius_in_pixels=radius_in_pixels)
            p = Patch(obb=None, volumetric=data[3:], rgb=data[:3],label=0, ori_xy=(row[0],row[1]), dr_dc_angle=(0,0,0))
            
            with open(os.path.join(directory, surffix), 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def make_next_batch_variations(self, number_of_variations=20):
        top = os.path.join(C.INT_DATA, "srp/samples")
        origs = glob.glob(os.path.join(top, "*/*/*_orig.pickle"), recursive=False)
        for origd in origs:
            with open(origd, 'rb') as handle:
                p = pickle.load(handle)
                for i in range(number_of_variations):
                    var = self._augment(p, radius=C.PATCH_SIZE/2)
                    with open(origd.replace("orig", "var{0:02d}".format(i+1)), 'wb') as handle:
                        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _augment(self, p, radius):
        radius = int(radius)
        dr = int(np.random.uniform(-1,1) * C.MAX_OFFSET)
        dc = int(np.random.uniform(-1,1) * C.MAX_OFFSET)
        rotate_angle = np.random.rand() * 360
        p_center = int(p.volumetric.shape[1]/2)

        source_patch = np.concatenate((p.rgb, p.volumetric))
        rotated_patch = np.zeros((source_patch.shape))

        for i in range(len(source_patch)):
            rotated_patch[i] = rotate(source_patch[i], rotate_angle, preserve_range=True)


            cropped_patch = rotated_patch[:, 
                                          p_center-radius+dc: p_center+radius+dc, 
                                          p_center-radius-dr: p_center+radius-dr]      
            rgb = cropped_patch[:3]
            volumetric = cropped_patch[3:]

        # ori_xy = (p.ori_xy[0], p.ori_xy[0]-dr*C.METERS_PER_PIXEL)        # not sure about this

        if p.label:
            R = affine.Affine.rotation(rotate_angle)
            T = affine.Affine.translation(dr, dc)
            A = T*R

            after = np.vstack(A * p.obb.points().T).T
            obb = OrientedBoundingBox.from_points(after)
            return Patch(obb=obb, ori_xy=p.ori_xy, rgb=rgb, label=p.label, volumetric=volumetric,dr_dc_angle=(dr, dc, rotate_angle))

        return Patch(obb=None, ori_xy=p.ori_xy, rgb=rgb, label=p.label, volumetric=volumetric, dr_dc_angle=(dr, dc, rotate_angle))
    