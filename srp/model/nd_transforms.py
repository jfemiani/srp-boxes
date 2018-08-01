"""
ND array transformations.

These transforms are applied to the last two dimensions of a numpy array
"""

# flake8:noqa
# pylint:disable=unused-import
from functools import partial

import numpy as np
from numpy.random import randint

from scipy import ndimage

import affine


def nd_normalize(x, mu, sigma):
    """Re-center and scale `x` given the mean (mu) and
    standard deviation (sigma) of the distribution of `x`.

    :param x:
    :param mu:
    :param sigma:
    """
    return (x - mu) / sigma


def nd_denormalize(x, mu, sigma):
    """Restore `x` to its oirignal mean (mu) and deviation (sigma)

    Thus undoes nd_normalize.

    :param x:
    :param mu: The mean of the (original) distribution of x vectors.
    :param sigma: The standard deviation of the (original) distribution of x vectors.
    """
    return (x * sigma) + mu


def nd_affine(x, transform):
    """Apply an affine transformation to x

    :param x:
    :param A: An 'affin.Affine' object
    """
    return ndimage.affine_transform(x, [[1, 0, 0, 0],
                                        [0, transform.a, transform.b, transform.c],
                                        [0, transform.d, transform.e, transform.f]])


def nd_rotation(x, angle=90):
    """Rotate x by the given angle (in degrees)

    :param x:
    :param angle:
    """
    transform = affine.Affine.rotation(angle, pivot=np.array(x.shape[1:]) / 2.)
    return nd_affine(x, transform)


def nd_crop(x, offset, size):
    """Crop out a subimage at the given offset and size.

    :param x:
    :param offset: A tuple (row, col) with the offset.
    :param size: The new size (rows, cols) of the last two dimensions.
    """
    hi = offset + np.array(size)
    return x[:, offset[0]:hi[0], offset[1]:hi[1]]


def nd_center_crop(x, size):
    """Crop around the center of the image.

    :param x:
    :param size: the size of the cropped image
    """
    size = np.array(size)
    offset = (np.array(x.shape[1:]) - size) // 2
    return nd_crop(x, offset, size)


def nd_random_rotation(x, max_angle):
    """Rotate by a random angle within +/- max_angle degrees.

    :param x:
    :param max_angle:
    """
    angle = randint(-max_angle, max_angle)
    return nd_rotation(x, angle)


def nd_random_crop(x, size):
    """Generate a random crop of size `size`

    :param x:
    :param size:
    """
    size = np.array(size)
    maxoff = np.array(x.shape[1:]) - size
    offset = np.array([randint(0, maxoff[0]), randint(0, maxoff[1])])
    return nd_crop(x, offset, size)
