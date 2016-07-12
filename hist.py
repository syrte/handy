from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt
from .stats import binstats, quantile

__all__ = ['hist_stats', 'hist2d_stats', 'steps', 'cdfsteps', 'pdfsteps']


def hist_stats(x, y, bins=10, func=np.mean, nmin=None, **kwds):
    """
    Example
    -------
    x = np.random.rand(1000)
    hist_stats(x, x, func=lambda x:np.percentile(x, [50, 15, 85]),
            ls=['-', '--', '--'], lw=[2, 1, 1], color=['k', 'b', 'b'])
    """
    stats, edges, count = binstats(x, y, bins=bins, func=func, nmin=nmin)
    stats = np.atleast_2d(stats.T)
    assert len(edges) == 1
    assert stats.ndim == 2

    X = (edges[0][:-1] + edges[0][1:]) / 2.
    lines = []
    for i, Y in enumerate(stats):
        args = {k: (v if np.isscalar(v) else v[i]) for k, v in kwds.items()}
        lines += plt.plot(X, Y, **args)
    return lines


def hist2d_stats(x, y, z, bins=10, func=np.mean, nmin=None, **kwds):
    stats, edges, count = binstats([x, y], z, bins=bins, func=func, nmin=nmin)
    (X, Y), Z = edges, stats.T
    mask = ~np.isfinite(Z)
    Z = np.ma.array(Z, mask=mask)
    kwds.setdefault('vmin', Z.min())
    kwds.setdefault('vmax', Z.max())
    return plt.pcolormesh(X, Y, Z, **kwds)


def steps(x, y, *args, **kwargs):
    ''' Make a step plot.
    The interval from x[i] to x[i+1] has level y[i]
    This function is useful for show the results of np.histogram.

    Additional keyword args to :func:`steps` are the same as those
    for :func:`~matplotlib.pyplot.plot`.

    Keyword arguments:
    fill: bool
        If True, the step line will be filled.
    vline: bool
        If True, two vertical lines will be plotted at borders.
    bottom: float
        The bottom of the vlines at borders.
    '''

    m, n = len(x), len(y)
    if m == n:
        return plt.step(x, y, *args, **kwargs)
    elif m != n + 1:
        raise ValueError

    fill = kwargs.pop('fill', False)
    vline = kwargs.pop('vline', False)
    bottom = kwargs.pop('bottom', 0)
    x, y = np.c_[x, x].flatten(), np.c_[y, y].flatten()
    if vline or fill:
        y = np.r_[bottom, y, bottom]
    else:
        x = x[1:-1]
    if fill:
        return plt.fill(x, y, *args, **kwargs)
    else:
        return plt.plot(x, y, *args, **kwargs)


def cdfsteps(x, *args, **kwds):
    side = kwds.pop('side', 'left')
    normed = kwds.pop('normed', True)
    n = x.size
    x = np.sort(x)
    x = np.r_[x[0], x, x[-1]]
    if side == 'left':
        h = np.arange(0, n + 1, dtype='f')
    elif side == 'right':
        h = np.arange(n, -1, -1, dtype='f')
    if normed:
        h = h / n
    steps(x, h, *args, **kwds)


def pdfsteps(x, *args, **kwds):
    x = np.sort(x)
    h = 1. / x.size / np.diff(x)
    steps(x, h, *args, vline=True, **kwds)


def compare(x, y, xbins=10, ybins=None, nan_as=None, nmin=3,
            plot=True, scatter=True, fill=False, sig1=True, sig2=True,
            **kwds):

    x, y = np.asarray(x), np.asarray(y)
    if ybins is not None:
        w, z, bins = y, x, ybins
    else:
        w, z, bins = x, y, xbins

    idx = np.isnan(z)
    if nan_as is None:
        z, w = z[~idx], w[~idx]
    else:
        z = np.array(z, 'f')
        z[idx] = nan_as

    func = lambda x: quantile(x, nsig=[0, -1, 1, -2, 2])
    stats, edges, count = binstats(x, y, bins=bins, func=func, nmin=nmin)
    zs = np.atleast_2d(stats.T)
    ws = (edges[0][:-1] + edges[0][1:]) / 2.

    ax = plt.gca()
    if xbins is not None:
        xs, ys = [ws] * 5, zs
        fill_between = ax.fill_between
    else:
        xs, ys = zs, [ws] * 5
        fill_between = ax.fill_betweenx

    format = kwds.pop("ls", ['k-', 'b--', 'g-.'])
    color = [ls[0] for ls in format]
    linestyle = [ls[1:] for ls in format]
    kwds.setdefault("color", color)
    kwds.setdefault("linestyle", linestyle)
    kwds.setdefault("label", ['median', '1 sigma', '2 sigma'])

    if plot:
        for i in range(3):
            args = {k: (v if np.isscalar(v) else v[i]) for k, v in kwds.items()}
            j = 2 * i
            ax.plot(xs[j], ys[j], **args)
            if j != 1:
                j = j - 1
                args.pop('label')
                ax.plot(xs[j], ys[j], ls[i], **args)
    if scatter:
        ax.scatter(xs[0], ys[0], s=2)
    if fill:
        if sig1:
            fill_between(ws, zs[1], zs[2], color=ls[1][0], edgecolor='none', alpha=0.3)
        if sig2:
            fill_between(ws, zs[3], zs[4], color=ls[2][0], edgecolor='none', alpha=0.2)

    return ws, zs
