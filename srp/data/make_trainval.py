"""Generate train + val splits of the data.

Locations of data are specified in config.py
"""

import srp.config 
import rtree


# Note -- I may make a class out of this. Logic will remain the same 
# and I will merge any changes you make in
def make_trainval(rgb_image_path=None, 
                  volume_raster_path=None, 
                  annotations_path=None, 
                  output_dir=None,
                  min_separation=None,
                  C=srp.config
                  ):
    """

    No 'train' rectangle may overlap a 'val' rectangle. 

    :param rgb_image_path:
    :param volume_raster_path:
    :param annotations_path:
    :param output_dir:
    :param num_folds: The number of folds to generate

    Examples
    --------
    >>> make_trainval(output_dir='data/test', nfolds=5)
    >>> os.path.isdir('data/test/folds')
    True
    >>> os.path.isdir('data/test/folds/1')
    True
    >>> os.path.isdir('data/test/folds/5')
    True
    >>> os.path.isfile('data/test/folds/1/train.csv')
    True

    >>> import pandas as pd
    >>> ds = pd.load_csv('data/test/folds/1/val.csv')
    >>> ds.columns
    ['label', 'lon', 'lat', 'box-x', 'box-y', 'box-angle', 'box-length', 'box-width']

    The idea is 
    - `box-cx` is an offset from the sample center
    - `box-angle` is relative to the _rotated_ sample

    """
    rgb_image = rgb_image or C.IMAGE
    volume_raster = volume_raster or C.VOLUME
    annotations_path = annotations_path or C.ANNOTATIONS
    output_dir = output_dir or C.OUTPUT_DOR
    min_separation = min_separation or C.MIN_SEPARATION

    
    # TODO:
    # Read in each shape from the annotations and add a positive sample
    # Make a union of all positive shapes (using shapely)
    # Dilate the union by min_separation
    # randomly select negative points and discard any that fall in the dilated region



