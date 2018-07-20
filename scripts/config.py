import os

class Config(object):
    def __init__(self):
        self.ROOT = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
        self.DENSITIES_PATH = os.path.join(self.ROOT, 'data/stack_a4b2/stack.vrt')
        self.COLORS_PATH = '/workspace/transformer_box/sec11-26949.tif'
        self.ANNOTATION_PATH = os.path.join(self.ROOT, 'data/boxes_section11-epsg26949.geojson')
        self.SAMPLE_PATH = os.path.join(self.ROOT, 'data/sample_locations_epsg26949.npz')

        self.RADIUS_IN_PIXELS = 32
        self.JITTER = 0.3
        self.SPLIT_RATIO = .1
        self.nocolor = 0
        self.colorAug = False
        self.seed = 127