from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes, docstring

__all__ = ['axtext', 'mulegend', 'errorbar2', 'get_aspect']


@docstring.copy(axes.Axes.text)
def axtext(x, y, s, *args, **kwargs):
    ax = plt.gca()
    kwargs.setdefault('transform', ax.transAxes)
    return ax.text(x, y, s, *args, **kwargs)


@docstring.copy(axes.Axes.legend)
def mulegend(*args, **kwargs):
    """Multiple legend
    """
    ax = plt.gca()
    ret = plt.legend(*args, **kwargs)
    ax.add_artist(ret)
    return ret


def errorbar2(x, y, yerr=None, xerr=None, **kwds):
    if yerr is not None:
        assert len(yerr) == 2
        ymin, ymax = np.atleast_1d(*yerr)
        yerr = y - ymin, ymax - y
    if xerr is not None:
        assert len(xerr) == 2
        xmin, xmax = np.atleast_1d(*xerr)
        xerr = x - xmin, xmax - x
    return plt.errorbar(x, y, yerr=yerr, xerr=xerr, **kwds)


def get_aspect(ax=None):
    """get aspect of given axes
    """
    if ax is None:
        ax = plt.gca()

    A, B = ax.get_figure().get_size_inches()
    w, h = ax.get_position().bounds[2:]
    disp_ratio = (B * h) / (A * w)

    sub = lambda x, y: x - y
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio
