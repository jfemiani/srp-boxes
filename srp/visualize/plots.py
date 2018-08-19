"""Various functions to plot data.

 Plotting data from the torch dataloader during training:

 * plot_rgb: To plot the RGB portion of conctneated color+volumetric data
 * plot_lidar: To plot the LiDAR portion of concatenated color+volumetric data
 * plot_box: To plot the oriented bounding box, alligned with the plots for plot_rgb or plot_lidar.
 * plot_inputs: Plots the concatenated stack (RGB + LiDAR). Does _not_ plot the box.
 * plot_batch: Plots a batch (or a pert of a batch) of data from a dataloader, including the label and box.

"""
import numpy as np
from matplotlib import pyplot as plt
from math import ceil, sqrt
from srp.data.orientedboundingbox import OrientedBoundingBox


def plot_rgb(x, ax=None):
    """Plot the RGB portion of the sample.

    Parameters
    ----------
    x: A stack of RGB and lidar data
    ax: A matplotlib axes
    """
    ax = ax or plt.gca()
    rgb = x[:3].transpose(1, 2, 0)
    cx, cy = np.array(rgb.shape[:2]) / 2
    extent = [-cx, cx, -cy, cy]
    ax.imshow(rgb, extent=extent)


def plot_lidar(x, ax=None, channels=(2, 3, 4), alpha=0.5):
    """Plot the volumetric portion of the sample.

    Parameters
    ----------
    x: A stack of RGB and lidar data
    ax: A matplotlib axes
    channels: Which slices of the volumetric data you wish to plot.
    alpha: Opacity for the plot; useful when you plot lidar on top of color imagery
    """
    ax = ax or plt.gca()
    vol = x[np.array(channels), :, :].transpose(1, 2, 0)
    display_vol = 2 * np.arctan(vol) / np.pi
    cx, cy = np.array(vol.shape[:2]) / 2
    extent = [-cx, cx, -cy, cy]
    ax.imshow(display_vol, alpha=alpha, extent=extent)


def plot_box(params, ax=None):
    """Plot a bounding box based on parameters

    Parameters
    ----------
    params: The parameters (4 points) of an oriented bounding box.
    ax: A matplotlib axes object

    """
    ax = ax or plt.gca()
    obb = OrientedBoundingBox.from_points(params)
    obb.plot(ax, lw=4, color='yellow')
    obb.plot(ax, lw=3, ls='--', color='red')


def plot_inputs(x, ax=None, channels=(3, 4, 5), alpha=0.5):
    ax = ax or plt.gca()
    plot_rgb(x, ax)
    plot_lidar(x, ax, channels, alpha)


def plot_batch(batch, num_samples=None, alpha=0.5):
    """Plot a batch of images.


    Plots a visualization of the _training_ data for each batch.

    Parameters
    ----------
    batch: A batch of data (exactly what is returned from RgbDataLoader)
    num_samples: The number of samples to plot. By default, the entire batch.

    """
    x, (y, params) = batch
    x = x.numpy()
    y = y.numpy()
    params = params.numpy()

    if num_samples is None:
        num_samples = len(x)

    nrows = int(ceil(sqrt(num_samples)))
    ncols = int(ceil(num_samples / nrows))

    axes = [plt.subplot(nrows, ncols, i + 1) for i in range(num_samples)]

    for i, ax in enumerate(axes):
        plt.sca(ax)
        plot_inputs(x[i], ax, alpha=alpha)
        if y[i] == 1:
            plot_box(params[i], ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{}:{}'.format(i + 1, y[i]), fontsize='small', labelpad=1)
