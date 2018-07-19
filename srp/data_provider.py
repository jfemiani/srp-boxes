
import pyproj
import rasterio
import numpy as np
import scipy.ndimage
from scipy.ndimage import zoom
from math import cos, sin, radians

EPSG2223 = pyproj.Proj(init="epsg:2223", preserve_units=True)
EPSG26949 = pyproj.Proj(init="epsg:26949", preserve_units=True)
        
class DataProvider(object):
    def __init__(self):        
        self.densities_path = '/home/shared/srp/try2/stack.vrt'
        self.colors_path = '/home/shared/srp/rgb/rgb.vrt'
        self._open_datasets()
      
    def _open_datasets(self):
        self.densities = rasterio.open(self.densities_path)
        self.colors = rasterio.open(self.colors_path)
    
    def get_patch_xyr(self, x, y, angle, radius_in_pixels=32):
        source_patch = self.get_patch_xy(x, y, radius_in_pixels*2)
        width = height = 2*radius_in_pixels
                
        radians = np.radians(angle)
        c, s = cos(radians), sin(radians)
        R = np.matrix([[c, -s], 
                       [s, c]])
        X = np.asarray([width, height])
        X = np.asarray(X-R.dot(X)).flatten()
        
        rotated_patch = np.empty_like(source_patch)
        for i in range(len(rotated_patch)):
            scipy.ndimage.affine_transform(source_patch[i],
                                           matrix=R, offset=X, 
                                           output_shape = rotated_patch[i].shape,
                                           output=rotated_patch[i])
        
        x, y = int((source_patch.shape[2]-width)/2), int( (source_patch.shape[1]-height)/2)
        cropped_patch = rotated_patch[:, y:y+height, x:x+width].copy()
        return cropped_patch    
        
    
    def get_patch_xy(self, x, y, radius_in_pixels=32):
        R = radius_in_pixels
        x_2223, y_2223 = pyproj.transform(EPSG26949, EPSG2223, x, y)
        c_2223, r_2223 = np.asarray(~self.colors.affine * (x_2223, y_2223)).astype(int)
        c, r = np.asarray(~self.densities.affine*(x, y)).astype(int)
                
        colors = self.colors.read(window=((r_2223-R, r_2223+R), (c_2223-R, c_2223+R)), 
                                  boundless=True).astype(np.float32)/255.
        
        densities = self.densities.read(window=((r-R, r+R), (c-R, c+R)),
                                        boundless=True).astype(np.float32)
        
        combined = np.concatenate([colors, densities])
        
        return combined
    