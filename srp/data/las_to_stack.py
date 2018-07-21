import argparse
import multiprocessing
import os
import traceback
from configargparse import ArgumentParser
from glob import glob
from multiprocessing import Pool

import fiona
import laspy
import rasterio.crs
import rasterio
import numpy as np
import gc

from math import floor

from tqdm import tqdm


METERS = 1.
CENTIMETERS = CM = 0.01
INCHES = 2.54*CM
FEET = 12*INCHES


def save_stack(stack, las_fn, min_point, max_point, las_width, las_height):
    """Give a las file name, and a raster 'stack' computed based on it, save 
    output to an appropriately named location. 

    When this is called, the LIDAR data has already been read and an ndarray called 'stack' has been
    created based on it. This function just determines the files name and location and saves it to disk. 

    :param[in] stack: The raster file. 
    :param[in] las_fn: Tha name of the LAS file that was used to generate the stack/raster file. 
    :param[in] min_point: The minimum LIDAR point in the stack. 
    :param[in] max_point: The maximum LIDAR point in the stack
    :param[in] las_width: The width of the stack.
    :param[in] las_height: The height of the stack.

    :return: The path to the file that we saves. 
    """
    output_transform = rasterio.transform.from_bounds(min_point[0], min_point[1],
                                                      max_point[0], max_point[1],
                                                      las_width, las_height)
    output_crs = rasterio.crs.CRS.from_epsg(26949)

    output_fn = os.path.basename(las_fn)[:-3] + 'tif'

    # Replace output file if it already existed
    if os.path.isfile(output_fn):
        os.remove(output_fn)

    # Create a new raster file with the right shape & geo-location
    output_file = rasterio.open(output_fn, 'w',
                                driver=u'GTiff',
                                crs = output_crs,
                                transform=output_transform,
                                dtype=rasterio.uint16,
                                count=len(stack),
                                width=las_width,
                                height=las_height)

    # Write out the stack-raster-data and close the file
    indexes = range(1, len(stack)+1)
    output_file.write(indexes=indexes, src=stack.astype(np.uint16))
    output_file.close()


    return output_fn


class Worker(object):

    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def __call__(self, las_fn):
        return self.doit(las_fn, *self.args, **self.kwargs)

    def doit(self, las_fn, minx, maxx, miny, maxy, OUTPUT_METERS_PER_PIXEL, GROUND_FN, SLICE_THICKNESS, ABOVE, BELOW):
        """Processes a single LAS/LAZ file
        This function either returns the name of a file, or an exception object. 
        It is intended to be used in multiproccessing.Pool.imap
        """
        output_fn = ''
        gc.collect()
        try:
            las_file = laspy.file.File(las_fn)
            X = las_file.x
            Y = las_file.y
            Z = las_file.z
            min_point = las_file.header.min
            max_point = las_file.header.max
            las_file.close()

            if max_point[0] < minx or min_point[0] > maxx or max_point[1] < miny or min_point[1] > maxy:
                return "Skipping {}, out-of-bounds file".format(os.path.split(las_fn)[1])

            las_min_x = int(floor((min_point[0] - minx) / OUTPUT_METERS_PER_PIXEL))
            las_min_y = int(floor((min_point[1] - miny) / OUTPUT_METERS_PER_PIXEL))
            las_max_x = int(floor((max_point[0] - minx) / OUTPUT_METERS_PER_PIXEL))
            las_max_y = int(floor((max_point[1] - miny) / OUTPUT_METERS_PER_PIXEL))
            las_width = las_max_x - las_min_x + 1
            las_height = las_max_y - las_min_y + 1

            ground_file = rasterio.open(GROUND_FN)
            ground_min_x, ground_min_y = ~(ground_file.affine ) *(min_point[0], min_point[1])
            ground_max_x, ground_max_y = ~(ground_file.affine ) *(max_point[0], max_point[1])
            ground_min_x = int(round(ground_min_x))
            ground_min_y = int(round(ground_min_y))
            ground_max_x = int(round(ground_max_x))
            ground_max_y = int(round(ground_max_y))

            flipped = False
            if ground_max_y <  ground_min_y:
                ground_min_y, ground_max_y = ground_max_y, ground_min_y
                flipped = True

            ground_width = ground_max_x - ground_min_x + 1
            ground_height = ground_max_y - ground_min_y + 1

            ground = ground_file.read(1,
                                      window=((ground_min_y, ground_max_y + 1), (ground_min_x, ground_max_x + 1)),
                                      boundless=True)
            if flipped:
                ground = ground[::-1 ,:]
            ground_file.close()

            # Find the rows and columns of each LIDAR point in the _output_ file.
            C = np.floor(((X - minx ) /OUTPUT_METERS_PER_PIXEL)).astype(int) - las_min_x
            R = np.floor(((Y - miny ) /OUTPUT_METERS_PER_PIXEL)).astype(int) - las_min_y

            # Subtract the corresponding value from the _ground_ file to get height-above-ground.
            height_above_ground = Z - ground[( R *ground_height ) /las_height, ( C *ground_width ) /las_width]

            # Determine how many slices above the ground each point is at.
            level = np.floor((height_above_ground ) /SLICE_THICKNESS).astype(int)

            def _generate_level(lvl):
                bc = np.bincount(R[ level ==lvl ] *las_width + C[ level ==lvl], minlength= las_width *las_height)
                output = bc.reshape((las_height, las_width)).astype(float)
                return output

            # Count the number of LIDAR points in each slice (for slices 1, 2, 3).
            # Note this does not include 0
            lvls = range(1 - BELOW, ABOVE+1)
            stack = np.stack([_generate_level(lvl) for lvl in lvls])

            # I seem to be working upside-down...
            output_fn = save_stack(stack[: ,::-1 ,:], las_fn,  min_point, max_point, las_width, las_height)
        except Exception as e:
            return (e, traceback.format_exc())

        return output_fn




def make_stack(las_files,
               stack_filename,
               vector_filename,
               ground_filename,
               processes=20,
               meters_per_pixel=0.5,
               padding=5,
               slice_thickness=0.313,
               above=3,
               below=0):

    assert os.path.isfile(vector_filename)
    assert os.path.isfile(ground_filename)

    print len(las_files)

    # These are the points we manually marked, used to determine the region of interest
    vector_file = fiona.open(vector_filename)
    minx, miny, maxx, maxy = vector_file.bounds

    # Expand the region so that a local window centered at points on the edges
    # does not go outside the grid
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding

    print "Region of interest (based on vector data)"
    print minx, miny, "--", maxx, maxy


    if not os.path.isfile(stack_filename):
        workers = Pool(processes=processes)
        pbar = tqdm(total=len(las_files))
        try:
            completed = []

            worker_function = Worker(float(minx), float(maxx), float(miny), float(maxy), float(meters_per_pixel),
                                     str(ground_filename), float(slice_thickness), int(above), int(below))
            for output_fn in workers.imap(worker_function, las_files):
                # print output_fn
                completed.append(output_fn)
                pbar.update(1)
        finally:
            workers.terminate()
            workers.close()
            pbar.close()

def main():
    p = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('-c', '--config', is_config_file=True )
    p.add('-l', '--lasfiles',
         nargs='+',
         help="A list of .las files for a region")
    p.add('--output', '-o',
         help="The output filename (.vrt)",
         default='stack.vrt')
    p.add('--ground',
         help="A digital elevation model / ground elevations for the region of interest")
    p.add('--roi',
         help="A vector file whose bounds are the region of interest.")
    p.add('--jobs', '-j',
         help="The number of processes to run in parallel",
         default=multiprocessing.cpu_count())
    p.add('--mpp',
         help="Output meters per pixel",
         default=0.05)
    p.add('--slice-thickness',
         help="The thickness of each volumetric slice",
         default=1*FEET)
    p.add('--padding',
         help="An amount to pad the ROI by",
         default=5)
    p.add('--above', '-a',
         help="The number of slices applied above the ground for the region of interest",
         default=3)
    p.add('--below', '-b',
          help='The number of slices below ground applied to the region of interest',
          default=0)

    args = p.parse_args()

    lasfiles = sum([glob(fn) for fn in args.lasfiles], [])

    # import ipdb; ipdb.set_trace()

    make_stack(lasfiles,
               args.output,
               args.roi,
               args.ground,
               args.jobs,
               args.mpp,
               args.padding,
               args.slice_thickness,
               args.above,
               args.below)


if __name__ == '__main__':
    main()