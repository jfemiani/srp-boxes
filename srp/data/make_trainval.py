"""Generate train + val splits of the data.

Locations of data are specified in config.py
"""

import srp.config
from shapely.strtree import STRtree
import shapely.geometry as sg
import fiona


# Note -- I may make a class out of this. Logic will remain the same
# and I will merge any changes you make in
def make_sample_meta(rgb_image_path=None,
                     volume_raster_path=None,
                     annotations_path=None,
                     output_dir=None,
                     min_separation=None,
                     C=srp.config):
    """

    No 'train' rectangle may overlap a 'val' rectangle.

    :param rgb_image_path:
    :param volume_raster_path:
    :param annotations_path:
    :param output_dir:
    :param num_folds: The number of folds to generate

    Example:

    >>> pos, neg = make_sample_meta(output_dir='data/test')

    >>> pos.columns
    ['lon', 'lat', 'box-x', 'box-y', 'box-angle', 'box-length', 'box-width']

    The idea is
    - `box-x`. `box-y`' is an offset from the sample center. At this point they will always be 0
    - `box-angle` is the angle (modulo 90 deg) of the box.
    - `box-length` is the length of the box
    - `box-width` is the width of the box

    This function does NOT do data augmentation or split the data.
    It generates a master set of sample metadata.

    """
    rgb_image = rgb_image or C.IMAGE
    volume_raster = volume_raster or C.VOLUME
    annotations_path = annotations_path or C.ANNOTATIONS
    output_dir = output_dir or C.OUTPUT_DOR
    min_separation = min_separation or C.MIN_SEPARATION
    num_samples = C.NUM_SAMPLES

    # Read in the positive samples
    pos_samples = [
        sg.shape(f['geometry']) for f in fiona.open(annotations_path)
    ]
    num_pos = len(pos_samples)

    # Build a spatial index to test if a new sample is within `min_separation` of
    # an existing positive
    pos_centers = [p.center for p in pos_samples]
    index = STRtree(pos_center)

    # Generate the (centers of) negative samples
    num_neg = num_samples - num_pos

    neg_centers = []
    valid_region = sg.box(*np.r_[np.min(pos_centers, axis=0),
                                 np.max(pos_centers, axis=0)])

    # Read in each shape from the annotations and add a positive sample
    # Make a union of all positive shapes (using shapely)
    # Dilate the union by min_separation
    # randomly select negative points and discard any that fall in the dilated region
