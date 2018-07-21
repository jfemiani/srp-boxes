import fiona
import numpy as np
from rasterio._io import RasterReader
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.features import geometry_mask
from os import path


def find_center(v1, v2):
    """calculates correct center given the feature vertices are labeled clockwise"""
    diff = v2 - v1
    magnitude = np.linalg.norm(diff)
    mid = (v1 + v2) / 2.
    perp = np.array([diff[1], -diff[0]])
    unit_vec = perp / np.linalg.norm(perp)
    return mid + .5 * magnitude * unit_vec


def get_angle(feature):
    diff = feature[2] - feature[1]
    return np.arctan2(diff[1], diff[0]) * 180 / np.pi

def check_vector(vectorPath):
    """
    :param vector_path: in the format of .geojson. This file includes features that we labeled manually
    This function checks the manually labeled vector file, so that it can safely proceed to the next step
    1) checks duplicates in the vector file and removes duplicates
    2) checks features of the wrong size: eg for boxes their should be 4 coordinates each
    3) Checks if our method of finding the centroid for all samples works
    :return: 
    """
    with fiona.open(vectorPath) as vectorFile:
        hotspots = [np.array(f['geometry']['coordinates']) for f in vectorFile if f['geometry'] is not None]

    duplicate = [idx for idx in range(len(hotspots) - 1) if np.array_equal(hotspots[idx], hotspots[idx + 1])]
    hotspots = np.delete(hotspots, duplicate, axis=0)
    if duplicate:
        print "Deleted Indices {} which are duplicates".format(duplicate)
    else:
        print "No duplicated features found."

    wrong_size = [idx for idx in range(len(hotspots)) if hotspots[idx].size != 8]
    if wrong_size:
        print "Indices {} are of the wrong size, please fix them.".format(wrong_size)
    else:
        print "All features consists of 4 coordinates."

    assert len(wrong_size) == 0
    pos_xy = [find_center(hotspots[idx][1], hotspots[idx][2]) for idx in range(len(hotspots))]

    weird = []
    for idx in range(len(pos_xy)):
        pt = Point(pos_xy[idx][0], pos_xy[idx][1])
        b = Polygon(hotspots[idx])
        if not pt.within(b):
            weird.append(idx)
            hotspots[idx] = np.flipud(hotspots[idx])  # reverse the order of vertices
            pos_xy[idx] = find_center(hotspots[idx][1], hotspots[idx][2])
    if not weird:
        print "All features are labeled clockwise, ready to proceed. Returning box and centroid coordinates."
        return hotspots, pos_xy
    else:
        print "Indices {} are not labeled clockwise, flipped those, please check them.".format(weird)


def make_segmentation_ground_truth(vector_path, density_path):
    """
    Makes a mask for segmentation training.The output has same width and height as the LiDAR file.
    :param vector_path: manually labeled ground truth, boxes 
    :param density_path: LiDAR stack
    :return: 
    """
    vectorFile = fiona.open(vector_path)
    polygons = [f['geometry'] for f in vectorFile]
    for i in range(len(polygons)):
        polygons[i]['type'] = 'Polygon'
        polygons[i]['coordinates'] = [polygons[i]['coordinates'] + polygons[i]['coordinates'][:1]]

    densities = rasterio.open(density_path)
    out = rasterio.features.geometry_mask(polygons,
                                          out_shape=(densities.height, densities.width),
                                          transform=densities.affine,
                                          all_touched=True,
                                          invert=True)
    return out


class ProcessVector(object):
    def __init__(self, vector_path, densities_path, pos_path, outdir, roi=None):
        """
        :param vector_path: in the format of .geojson. This file includes features that we labeled manually
        :param densities_path: the LiDAR stack
        :param the tif file where 1 indicates isbox and 0 indicates not a box
        """
        super(ProcessVector, self).__init__()
        self.vectorPath = vector_path
        self.outdir = outdir
        self.vectorFile = fiona.open(self.vectorPath)
        self.hotspots, self.pos_xy = check_vector(self.vectorPath)

        assert self.pos_xy

        self.densitiesPath = densities_path
        self.densities = rasterio.open(self.densitiesPath)
        self.posPath = pos_path
        self.pos_regions = rasterio.open(self.posPath)

        self.bounds = tuple((max(self.pos_regions.bounds.left, self.densities.bounds.left),
                             max(self.pos_regions.bounds.bottom, self.densities.bounds.bottom),
                             min(self.pos_regions.bounds.right, self.densities.bounds.right),
                             min(self.pos_regions.bounds.top, self.densities.bounds.top))) if roi is None else roi
        self.neg_xy = []
        self.pos_angles = []

    def _open_datasets(self):

        assert isinstance(self.densities, RasterReader)
        window1 = self.densities.window(*self.bounds, boundless=True)
        self.tfm = self.densities.window_transform(window1)
        self.full_stack = self.densities.read((1, 2, 3), window=window1, boundless=True)

        window2 = self.pos_regions.window(*self.bounds, boundless=True)
        self.full_posregions = self.pos_regions.read(1,
                                                     window=window2,
                                                     out_shape=(self.full_stack.shape[1], self.full_stack.shape[2]),
                                                     boundless=True).astype(bool)

    def make_samples(self):
        self.pos_angles = [get_angle(self.hotspots[idx]) for idx in range(len(self.hotspots))]
        self.make_negs()
        np.savez(path.join(self.outdir, "sample_locations"),
                 pos_xy=self.pos_xy,
                 pos_angles=self.pos_angles,
                 neg_xy = self.neg_xy[::16])
        #return {'pos_xy': self.pos_xy, 'pos_angle': self.pos_angles, 'neg_xy': self.neg_xy}

    def make_negs(self):
        self._open_datasets()

        mask = self.full_stack[:2].sum(0) > 9
        mask[:, :64] = False
        mask[:, -64:] = False
        mask[:64, :] = False
        mask[-64:, :] = False

        neg_xy = np.where(mask & ~self.full_posregions)
        neg_xy = np.column_stack(neg_xy)
        neg_xy = np.roll(neg_xy, 1, 1)
        self.neg_xy = np.array(self.tfm * neg_xy.T).T


def fixme_pick_samples(vol, w):
    # vol is an np array
    samples = []
    for r in range(0. vol.size[0], w):
        for c in range(0, vol.size[1], w):
            sub = vol[r:r+w, c:c+w, :].sum(2).flatten()
            positions = np.where(sub > threshold)[]
            sample_indices, ignored = random.choice(len(positions))
            sub_samples = positions[sample_indices]
            samples.append(sub_samples)
    samples = np.concatenate(samples)