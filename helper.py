from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt

__all__ = ["axtext", 'errorbar2']


def axtext(x, y, s, *args, **kwargs):
    ax = plt.gca()
    kwargs.setdefault('transform', ax.transAxes)
    return ax.text(x, y, s, *args, **kwargs)


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
