"""
Configuration for the project.
------------------------------

This module builds the _default_ configuration for the project.

You can copy it to an experiment folder and modify it to match the experiment

Change this to match an experiment; this way you can keep track of which
results go with which settings
"""

import os

import srp
from srp.units import FT, M

# Paths {{{

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(srp.__file__)))
INT_DATA = os.path.join(ROOT, 'data', 'interim')
RAW_DATA = os.path.join(ROOT, 'data', 'raw')
POS_DATA = os.path.join(INT_DATA, 'srp/samples/pos')
NEG_DATA = os.path.join(INT_DATA, 'srp/samples/neg')

# }}}

# Volume File Settings {{{

#: The raster source for volume data
VOLUME_FILE = '{INT_DATA}/volume.tif'

#: The CRS to use if the data is missing in the file
VOLUME_DEFAULT_CRS = 'epsg:26949'  # AZ

#: The number of channels (slabs) of volumetric data.
VOLUME_STEPS = 5

#: The minimum height above ground(Z)
VOLUME_Z_MIN = -1 * FT
#: The maximum height above ground(Z)
VOLUME_Z_MAX = 4 * FT

# }}}

# Train / Val / Test split {{{

#: The path to the volumetric 6 channel data
#  WRONG: VOLUMETRIC_PATH = os.path.join(ROOT, 'data/interim/srp', 'lidar_volume.vrt')

#: The path to the rgb image
COLOR_PATH = os.path.join(ROOT, 'data/srp/sec11-26949.tif')

#: The path to the volumetric 6 channel data
VOLUMETRIC_PATH = os.path.join(ROOT, 'data/interim/srp/stack_a4b2/lidar_volume.vrt')

#: The path to the rgb image
COLOR_PATH = os.path.join(ROOT, 'data/raw/srp/sec11-26949.tif')

#: The path to annotated box coordinates
ANNOTATION_PATH = os.path.join(ROOT, 'data/raw/srp/box-annotations.geojson')

#: The path to all sample coordinates (including positives and negs)
SAMPLE_PATH = os.path.join(ROOT, 'data/interim/srp/sample_locations_epsg26949.npz')

#: The path to which the script outputs .csv file
CSV_DIR = os.path.join(ROOT, 'srp/data')
#: The number of folds (cross validation)
# Volume File Settings {{{

#: The raster source for volume data
VOLUME_FILE = '{INT_DATA}/volume.tif'

#: The CRS to use if the data is missing in the file
VOLUME_DEFAULT_CRS = 'epsg:26949'  # AZ

#: Total number of folds
FOLDS = 5
#: Which fold is current.
TRAINING_CURRENT_FOLD = 1
VOLUME_STEPS = 5
#: The RNG for folds (for repeatability)
FOLD_RANDOM_SEED = 127

#: The minimum threshold density within the first two layers when picking negative samples
MIN_DENSITY_COUNT = 15

#: Patch size
PATCH_SIZE = 64

#: The maximum offset when jittering each sample
MAX_OFFSET = 15

#: A constant needed for scaling 
METERS_PER_PIXEL=0.0499
# }}}

# Class balancing & Sampling {{{

#: The total number of samples.
#: Every positive (object) sample will be used, the remaining samples will
#: be background samples
NUM_SAMPLES = 2000

#: Whether to draw samples so that each batch has a balanced number of labels.
STRATIFY = False

#: The class weights STRATIFY
SRATIFY_BALANCE = 1, 1

MIN_SEPARATTION = 2 * M

#: Whether to use hard sampling.
#:
#: If this is set, then we weight each sample by its loss in the previous epoch
#: and we draw samples according to the weights.
HARD_SAMPLING = True

#: The number of precomputed variations for each positive and negative sample
NUM_PRECOMPUTE_VARIATION = 20

# }}}
