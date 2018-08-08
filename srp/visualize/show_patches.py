from pylab import plt
import os
import pandas as pd
import pickle
from matplotlib.widgets import Slider, RadioButtons
from easydict import EasyDict
from srp.config import C
import numpy as np
from srp.data.generate_patches import Patch


def show_patches():
    dirname = C.TRAIN.SAMPLES.DIR

    labels = EasyDict(pos=EasyDict(), neg=EasyDict())
    labels.pos.samples = pd.read_csv(os.path.join(dirname, 'positives.csv'))
    labels.pos.index = 0
    labels.neg.samples = pd.read_csv(os.path.join(dirname, 'negatives.csv'))
    labels.neg.index = 0

    labels.label = 'pos'

    fig = plt.figure()
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    plt.subplots_adjust(bottom=0.15)

    # Slider to choose the sample
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax_slider,
        'Index',
        0,
        len(labels[labels.label].samples),
        valinit=labels[labels.label].index,
        valstep=1,
        closedmax=False)

    # Radio buttons to choose the label
    ax_label = plt.axes([0.025, 0.05, 0.1, 0.1])
    radio_buttons = RadioButtons(ax_label, ['pos', 'neg'], active=0)

    ax_help = plt.axes([0.25, 0.10, 0.65, 0.05])
    ax_help.axis('off')
    ax_help.text(0, 0.5, 'Press [j,k] to change the index, [p,n] to set the label.')

    def update(dummy):

        if str(radio_buttons.value_selected) != labels.label:
            labels.label = radio_buttons.value_selected
            slider.valmax = len(labels[labels.label].samples)
            slider.val = labels[labels.label].index
            ax_slider.set_xlim(0, slider.valmax)

        labels[labels.label].index = int(slider.val)

        current = labels[labels.label].samples.iloc[labels[labels.label].index]
        with open(os.path.join(dirname, current['name']), 'rb') as f:
            patch = pickle.load(f)
            print(f.name)

        plt.suptitle(patch.name)

        vol = patch.volumetric[2:2 + 3].transpose(1, 2, 0)
        display_vol = 2 * np.arctan(vol) / np.pi

        radius = patch.rgb.shape[1] / 2
        extent = (-radius, radius, -radius, radius)

        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.imshow(patch.rgb.transpose(1, 2, 0), extent=extent)
        ax1.set_title('rgb')

        ax2.imshow(patch.rgb.transpose(1, 2, 0), extent=extent)
        ax2.imshow(display_vol, extent=extent, alpha=0.5)
        if patch.obb is not None:
            patch.obb.plot(ax2, lw=4, color='yellow')
            patch.obb.plot(ax2, lw=3, ls='--', color='red')
        ax2.set_title('both')

        ax3.imshow(display_vol, extent=extent)
        ax3.set_title('vol:max={:.1f}'.format(vol.max()))

        fig.canvas.draw_idle()

    # First plot
    update(0)

    radio_buttons.on_clicked(update)
    slider.on_changed(update)

    def keypress(event):
        if event.key == 'j':
            slider.set_val(slider.val - 1)
        elif event.key == 'k':
            slider.set_val(slider.val + 1)
        elif event.key == 'p':
            radio_buttons.set_active(0)
        elif event.key == 'n':
            radio_buttons.set_active(1)

    fig.canvas.mpl_connect('key_press_event', keypress)
    plt.show()


if __name__ == '__main__':
    show_patches()
