"""
Configuration for the project.
------------------------------

This module builds the _default_ configuration for the project.

You can copy it to an experiment folder and modify it to match the experiment

Change this to match an experiment; this way you can keep track of which
results go with which settings
"""

from srp.units import FT, M
import os

EXPERIMENT_NAME = 'default'

#: Project root
ROOT = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
# print ROOT
# Early / Layer Fusion Settings {{{


#: Whether use the data before feeding the network
EARLY_FUSION = True

#: Whether to concatenate the feature representations.
#:
#: One of:
#:   :'cat': Concatenate the flattened output vectors of
#:           the "bodies" of the net, then add a single
#:           hidden layer that reduces the number of
#:           outputs by 2.
#:   :'add': Add the flattened output vectors element-wise.
#:
LATE_FUSION_OPERATION = 'cat'
# LATE_FUSION_OPERATION='add'

# }}}

# Data Preparation {{{

#: The number of channels (slabs) of volumetric data.
VOLUME_SLABS = 5

#: The minimum height above ground(Z)
VOLUME_MIN_Z = -1 * FT
#: The maximum height above ground(Z)
VOLUME_MAX_Z = 4 * FT

# }}}

# Train / Val / Test split {{{

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
FOLDS = 5
#: Which fold is current.
FOLD = 1
#: The RNG for folds (for repeatability)
FOLD_RANDOM_SEED = 127
#: A checksum on the actual sampling used in the folds
#: Should be set when folds are made; to ensure no contamination of folds.
FOLD_CHECKSUM = None

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

#: The minimum distance between a background and a foreground sample
MIN_SEPARATION = 2*M

#: Whether to use hard sampling.
#:
#: If this is set, then we weight each sample by its loss in the previous epoch
#: and we draw samples according to the weights.
HARD_SAMPLING = True

# }}}
