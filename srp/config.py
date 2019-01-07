"""
Configuration for the project.
------------------------------

This module loads settings from a variety of config files that may exist on a system.
The configuration settings are stored in a class-like-object called `C`.

You can modify C and use `save_settings` to write-out the settings.

All config files are TOML  formatted files.

At the time a config file is parsed, the ROOT and DATA environment variable will be set based on this project's location
on the system.

Any key named 'PATH', 'FILE', 'FILENAME', 'FOLDER' or 'DIR' will be interpreted as a filename and will have environment
variables expanded.

In addition C.ROOT, C.DATA, C.INT_DATA, and C.RAW_DATA are all folders.

Example:

    >>> os.path.isdir(C.DATA)
    True

    >>> print(type(C.VOLUME.CRS).__name__)
    str

    (It could be anything -- the point of the config module is that the values change on your system)


The default config is kept with the sourcecode as 'srp-boxes.toml'

.. literalinclude:: /../srp-boxes.toml

"""

import logging.config
import os
import sys

import toml
from easydict import EasyDict

import srp

C = EasyDict()
C.ROOT = os.path.dirname(os.path.dirname(os.path.abspath(srp.__file__)))
C.DATA = os.path.join(C.ROOT, 'data')
C.INT_DATA = os.path.join(C.ROOT, 'data', 'interim')
C.RAW_DATA = os.path.join(C.ROOT, 'data', 'raw')

C.POS_DATA = os.path.join(C.INT_DATA, 'srp/samples/pos')
C.NEG_DATA = os.path.join(C.INT_DATA, 'srp/samples/neg')

os.environ.update(C)

_CONFIG_PATHS = [
    f'{C.ROOT}/srp-boxes.toml',
    '/etc/srp-boxes.toml',
    '/etc/.srp-boxes.toml'
    '~/srp-boxes.toml',
    '~/.srp-boxes.toml',
    'srp-boxes.toml',
    '.srp-boxes.toml',
    'config.toml',
]

PATH_KEYS = {'FILE', 'FILENAME', 'DIR', 'PATH', 'FOLDER', 'ROOT', 'DATA', 'INT_DIR', 'RAW_DIR'}

def merge_settings(a, b):
    for k in b:
        if k in a and isinstance(a[k], dict):
            merge_settings(a[k], b[k])
        else:
            a[k] = b[k]


def load_settings():
    """Load settings from config files.

    Settings are loaded (in order) from the files in _CONFIG_PATHS.
    The latter config files take precedence over (they overwrite) the latter config files.

    Config files are TOML formatted files.

    """
    # pylint:disable=global-statement
    global C
    configs = _CONFIG_PATHS
    for c in configs:
        if os.path.isfile(c):
            settings = toml.load(c, EasyDict)
            merge_settings(C, settings)
    
    def expandvars(settings):
        """Recursively replace environment variables for fields that are paths"""
        for key in settings:
            if isinstance(settings[key], dict):
                expandvars(settings[key])
            elif key in PATH_KEYS:
                settings[key] = os.path.expanduser(settings[key])
                settings[key] = os.path.expandvars(settings[key])

    expandvars(C)

    logging.config.dictConfig(C.LOGGING)


def save_settings(f=sys.stdout):
    """save_settings

    Args:
      f (str): File to save settings to. Default is standard output.
    """
    toml.dump(C, f)


load_settings()

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)

# ###############################################
# :arg **/*.py
# :argdo %s/VOLUME_DEFAULT_CRS/C.VOLUME.CRS/gc
# :argdo %s/VOLUME_STEPS/VOLUME.Z.STEPS/gc
# :argdo %s/VOLUME_Z_MIN/VOLUME.Z.MIN/gc
# :argdo %s/VOLUME_Z_MAX/VOLUME.Z.MAX/gc
# :argdo %s/COLOR_PATH/COLOR.FILE/gc
# :argdo %s/VOLUMETRIC_PATH/VOLUME.FILE/gc
# :argdo %s/ANNOTATION_PATH/ANNOTATIONS.FILE/gc
# :argdo %s/CSV_DIR/TRAIN.SAMPLES.DIR/gc
# :argdo %s/FOLDS/TRAIN.SAMPLES.FOLDS/gc
# :argdo %s/TRAINING_CURRENT_FOLD/TRAIN.SAMPLES.CURRENT_FOLD/gc
# :argdo %s/FOLD_RANDOM_SEED/TRAIN.SRAND/gc
# :argdo %s/MIN_DENSITY_COUNT/TRAIN.SAMPLES.GENERATOR.MIN_DENSITY/gc
# :argdo %s/PATCH_SIZE/TRAIN.SAMPLES.GENERATOR.PATCH_SIZE/gc
# :argdo %s/MAX_OFFSET/TRAIN.AUGMENTATION.TRAIN.AUGMENTATION.MAX_OFFSET/gc
# :argdo %s/METERS_PER_PIXEL/TRAIN.SAMPLES.GENERATOR.GSD/gc
# :argdo %s/NUM_SAMPLES/TRAIN.SAMPLES.GENERATOR.NUM_SAMPLES/gc
# :argdo %s/STRATIFY/TRAIN.STRATIFY/gc
# :argdo %s/SRATIFY_BALANCE/TRAIN.CLASS_BALANCE/gc
# :argdo %s/MIN_SEPARATTION/TRAIN.SAMPLES.GENERATOR.MIN_SEPARATTION/gc
# :argdo %s/HARD_SAMPLING/TRAIN.HARD_SAMPLING/gc
# :argdo %s/NUM_PRECOMPUTE_VARIATION/TRAIN.AUGMENTATION.VARIATIONS/gc

# :argdo %s/ANNOTATION\.FILE/ANNOTATIONS.FILE/gc
