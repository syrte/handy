from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt
from .stats import binstats, quantile

__all__ = ['pcolorshow', 'hist_stats', 'hist2d_stats', 'steps',
           'cdfsteps', 'pdfsteps', 'compare']


def _pcolorshow_args(x, m):
    """helper function for pcolorshow, check the args and return 
    the range of data.
    """
    assert x.ndim == 1, "unexpected array dimentions"
    assert len(x) in [m, m + 1], "unexpected array shape"
    dx = x[1] - x[0] if len(x) > 1 else 1
    assert np.allclose(np.diff(x), dx), "the bin size must be equal"
    if len(x) == m:
        return np.min(x) - 0.5 * dx, np.max(x) + 0.5 * dx
    else:
        return np.min(x), np.max(x)


def pcolorshow(*args, **kwargs):
    """pcolorshow([x, y], z, interpolation='nearest', **kwargs)
    similar to pcolormesh but using `imshow` as backend.
    It renders faster than `pcolor(mesh)` and supports more interpolation
    schemes, but only works with equal bins.

    Parameters
    ----------
    x, y : array like, optional
        Coordinates of bins.
    z : 
        The color array. z should be in shape (ny, nx) or (ny + 1, nx + 1)
        when x, y are given.
    interpolation : string, optional
        Acceptable values are 'nearest', 'bilinear', 'bicubic',
        'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
        'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
        'lanczos'
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with norm to normalize
        luminance data.  Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.

    Example
    -------
    a = np.arange(10)
    pcolorshow(a, 0.5, a)
    """
    assert len(args) in [1, 3], "should input `x, y, z` or `z`"
    z = np.atleast_2d(args[-1])
    n, m = z.shape

    if len(args) == 1:
        xmin, xmax = 0, m
        ymin, ymax = 0, n
    else:
        x, y = np.atleast_1d(*args[:2])
        xmin, xmax = _pcolorshow_args(x, m)
        ymin, ymax = _pcolorshow_args(y, n)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("aspect", plt.gca().get_aspect())
    kwargs.setdefault("extent", (xmin, xmax, ymin, ymax))
    kwargs.setdefault('interpolation', 'nearest')

    return plt.imshow(z, **kwargs)


def hist_stats(x, y, bins=10, func=np.mean, nmin=1, **kwds):
    """
    style:
        'plot', 'scatter', 'step'

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

    style_dict = {'plot': plt.plot,
                  'scatter': plt.scatter,
                  'step': steps, }
    style = kwds.pop('style', 'plot')
    assert style in style_dict
    plot = style_dict[style]

    if style == 'step':
        X = edges[0]
    else:
        X = (edges[0][:-1] + edges[0][1:]) / 2.

    lines = []
    for i, Y in enumerate(stats):
        args = {k: (v if np.isscalar(v) else v[i])
                for k, v in kwds.items()}
        lines += plot(X, Y, **args)
    return lines


def hist2d_stats(x, y, z, bins=10, func=np.mean, nmin=1, **kwds):
    stats, edges, count = binstats([x, y], z, bins=bins, func=func, nmin=nmin)
    assert len(edges) == 2
    assert stats.ndim == 2

    (X, Y), Z = edges, stats.T
    mask = ~np.isfinite(Z)
    Z = np.ma.array(Z, mask=mask)
    kwds.setdefault('vmin', Z.min())
    kwds.setdefault('vmax', Z.max())
    return plt.pcolormesh(X, Y, Z, **kwds)


def steps(x, y, *args, **kwargs):
    '''steps(x, y, *args, fill=False, border=False, bottom=0, **kwargs)
    Make a step plot.
    The interval from x[i] to x[i+1] has level y[i]
    This function is useful for show the results of np.histogram.

    Parameters
    ----------
    args, kwargs:
        same as those for `matplotlib.pyplot.plot` or
        `matplotlib.pyplot.plot.fill` if `fill=True`.
    fill: bool
        If True, the step line will be filled.
    border: bool
        If True, two vertical lines will be plotted at borders.
    bottom: float
        The bottom of the vlines at borders.
    '''
    m, n = len(x), len(y)
    if m == n:
        kwargs.setdefault("where", "mid")
        return plt.step(x, y, *args, **kwargs)
    elif m != n + 1:
        raise ValueError

    fill = kwargs.pop('fill', False)
    border = kwargs.pop('border', False)
    bottom = kwargs.pop('bottom', 0)
    kwargs.pop('drawstyle', None)
    kwargs.pop('where', None)

    x, y = np.repeat(x, 2), np.repeat(y, 2)
    if border or fill:
        y = np.hstack([bottom, y, bottom])
    else:
        x = x[1:-1]
    if fill:
        return plt.fill(x, y, *args, **kwargs)
    else:
        return plt.plot(x, y, *args, **kwargs)


def cdfsteps(x, *args, **kwds):
    """
    Parameters
    ----------
    x:
        data
    side: str
        'left' or 'right', assending or decending.
    normed: bool
    sorted: bool
    """
    side = kwds.pop('side', 'left')
    normed = kwds.pop('normed', True)
    sorted = kwds.pop('sorted', False)
    assert side in ['right', 'left']

    if not sorted:
        x = np.sort(x)
    n = float(x.size)
    h = np.arange(0, n + 1, dtype=float)
    if side == 'right':
        h = h[::-1]
    if normed:
        h = h / n
    x = np.hstack([x[0], x, x[-1]])
    return steps(x, h, *args, **kwds)


def pdfsteps(x, *args, **kwds):
    sorted = kwds.pop('sorted', False)
    if not sorted:
        x = np.sort(x)
    h = 1. / x.size / np.diff(x)
    return steps(x, h, *args, border=True, **kwds)


def compare(x, y, xbins=None, ybins=None, weights=None, nanas=None, nmin=3,
            scatter=True, plot=(0, 1, 2), fill=(),
            scatter_kwds={}, fill_kwds={}, **kwds):
    """
    Example
    -------
    compare(x, y, 10, 
        scatter=False, 
        plot=[0, 1],
        fill=[1])
    """
    plot = [plot] if np.isscalar(plot) else plot
    fill = [fill] if np.isscalar(fill) else fill

    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
    if ybins is not None:
        assert xbins is None
        w, z, bins = y, x, ybins
    else:
        xbins = 10 if xbins is None else xbins
        w, z, bins = x, y, xbins

    idx = np.isnan(z)
    if idx.any():
        if nanas is None:
            idx = ~idx
            z, w = z[idx], w[idx]
            if weights is not None:
                weights = weights[idx]
        else:
            z = np.array(z, dtype=float)
            z[idx] = float(nanas)

    if weights is None:
        func = lambda x: quantile(x, nsig=[0, -1, -2, 1, 2], nmin=nmin)
        stats, edges, count = binstats(w, z, bins=bins, func=func)
    else:
        func = lambda x, weights: quantile(x, weights, nsig=[0, -1, -2, 1, 2], nmin=nmin)
        stats, edges, count = binstats(w, [z, weights], bins=bins, func=func)
    zs = np.atleast_2d(stats.T)
    ws = (edges[0][:-1] + edges[0][1:]) / 2.
    #ws = binstats(w, w, bins=bins, func=np.meidan, nmin=nmin).stats

    ax = plt.gca()
    if xbins is not None:
        xs, ys = [ws] * 5, zs
        fill_between = ax.fill_between
    else:
        xs, ys = zs, [ws] * 5
        fill_between = ax.fill_betweenx

    fmt = kwds.pop("fmt", ['k-', 'b--', 'g-.'])
    kwds.setdefault("label", ['median', '1 sigma', '2 sigma'])
    kwds.setdefault("color", [ls[:1] for ls in fmt])
    kwds.setdefault("linestyle", [ls[1:] for ls in fmt])
    for i in plot:
        args = {k: (v if np.isscalar(v) else v[i])
                for k, v in kwds.items()}
        ax.plot(xs[i], ys[i], **args)
        if i > 0:
            args.pop('label', None)
            ax.plot(xs[i + 2], ys[i + 2], **args)

    scatter_kwds.setdefault('s', 20)
    scatter_kwds.setdefault('c', 'k')
    if scatter:
        ax.scatter(xs[0], ys[0], **scatter_kwds)

    fill_kwds.setdefault('color', ['b', 'g'])
    fill_kwds.setdefault('alpha', [0.4, 0.2])
    fill_kwds.setdefault('edgecolor', 'none')
    for i in fill:
        args = {k: (v if np.isscalar(v) else v[i - 1])
                for k, v in fill_kwds.items()}
        fill_between(ws, zs[i], zs[i + 2], **args)

    return
