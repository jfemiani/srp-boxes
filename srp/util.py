""" General utility functions that are not big enough to warnat their own module.

For dealing with progress reporting on slow processes I have the functions ``clear_tqdm``, ``tdqm`` and ``trange``
which mainly replace the default values of the ``tqdm`` module basedon our configuration file.


"""
import sys
import tqdm as tq
from srp.config import C


def clear_tqdm():
    """Clear any unfinished progress bars that may remain (in case an exception or similar has occured).

    Get rid of any lingering progress bar that may remain in tqdm. This is mainly useful when
    we are inside of an interactive environment such as ipython or jupyter.
    """
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst:
        return
    try:
        for _ in range(len(inst)):
            inst.pop().close()
    except Exception:  # pylint:disable=broad-except
        pass


def tqdm(*args, **kwargs):
    """tqdm that prints to stdout instead of stderr

    Keyword Arguments:
        file: The output file; defaults to sys.stdout.
        disable: Whether to disable the progress bar. Defaults to C.DISPLAY.PROGRESS.DISABLE.

    All other keyword arguments are forwarded to tqdm.
    """
    kwargs_ = dict(file=sys.stdout, disable=C.DISPLAY.PROGRESS.DISABLE, leave=False)
    kwargs_.update(kwargs)
    clear_tqdm()
    return tq.tqdm(*args, **kwargs_)


def trange(*args, **kwargs):
    """trange that prints to stdout instead of stderr

    :param *args:
    :param **kwargs:
    """
    kwargs_ = dict(file=sys.stdout, disable=C.DISPLAY.PROGRESS.DISABLEi, leave=False)
    kwargs_.update(kwargs)
    clear_tqdm()
    return tq.trange(*args, **kwargs_)
