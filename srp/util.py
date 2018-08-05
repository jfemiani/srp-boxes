import sys
import tqdm as tq
from srp.config import C


def clear_tqdm():
    """clear_tqdm
    Get rid of any lingering progress bar that may remain in tqdm.
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

    :param *args:
    :param **kwargs:
    """
    clear_tqdm()
    return tq.tqdm(*args, file=sys.stdout, disable=C.DISPLAY.PROGRESS.DISABLE, **kwargs)


def trange(*args, **kwargs):
    """trange that prints to stdout instead of stderr

    :param *args:
    :param **kwargs:
    """
    clear_tqdm()
    return tq.trange(*args, file=sys.stdout, disable=C.DISPLAY.PROGRESS.DISABLE, **kwargs)
