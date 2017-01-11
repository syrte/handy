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
    if x.ndim != 1:
        raise ValueError("unexpected array dimentions")
    elif x.size > 1:
        dx = x[1] - x[0]
    else:
        dx = 1

    if not np.allclose(np.diff(x), dx):
        raise ValueError("the bin size must be equal.")

    if x.size == m:
        return np.min(x) - 0.5 * dx, np.max(x) + 0.5 * dx
    elif x.size == m + 1:
        return np.min(x), np.max(x)
    else:
        raise ValueError("unexpected array shape")


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
    z = np.atleast_2d(args[-1])
    n, m = z.shape

    if len(args) == 1:
        xmin, xmax = 0, m
        ymin, ymax = 0, n
    elif len(args) == 3:
        x, y = np.atleast_1d(*args[:2])
        xmin, xmax = _pcolorshow_args(x, m)
        ymin, ymax = _pcolorshow_args(y, n)
    else:
        raise ValueError("should input `x, y, z` or `z`")

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("aspect", plt.gca().get_aspect())
    kwargs.setdefault("extent", (xmin, xmax, ymin, ymax))
    kwargs.setdefault('interpolation', 'nearest')

    return plt.imshow(z, **kwargs)


def hist_stats(x, y, bins=10, func=np.mean, nmin=1, style="plot", **kwargs):
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
                  'step': steps}
    plot = style_dict[style]

    if style == 'step':
        X = edges[0]
    else:
        X = (edges[0][:-1] + edges[0][1:]) / 2.

    lines = []
    for i, Y in enumerate(stats):
        args = {k: (v if np.isscalar(v) else v[i])
                for k, v in kwargs.items()}
        lines += plot(X, Y, **args)
    return lines


def hist2d_stats(x, y, z, bins=10, func=np.mean, nmin=1, **kwargs):
    stats, edges, count = binstats([x, y], z, bins=bins, func=func, nmin=nmin)
    assert len(edges) == 2
    assert stats.ndim == 2

    (X, Y), Z = edges, stats.T
    mask = ~np.isfinite(Z)
    Z = np.ma.array(Z, mask=mask)
    kwargs.setdefault('vmin', Z.min())
    kwargs.setdefault('vmax', Z.max())
    return plt.pcolormesh(X, Y, Z, **kwargs)


def steps(x, y, *args, **kwargs):
    '''steps(x, y, *args, style='line', bottom=0, guess=True, 
             orientation='vertical', **kwargs)
    Make a step plot.
    The interval from x[i] to x[i+1] has level y[i]
    This function is useful for show the results of np.histogram.

    Parameters
    ----------
    x, y : 1-D sequences
        Data to plot.
        - If len(x) == len(y) + 1
            y keeps y[i] at interval from x[i] to x[i+1].
        - If len(x) == len(y)
            y jumps from y[i] to y[i+1] at (x[i] + x[i+1])/2.
    style : ['default' | 'step' | 'filled' | 'bar' | 'line'], optional
        The type of steps to draw.
        - 'default': step line plot
        - 'step': step line with vertical line at borders.
        - 'filled': filled step line plot
        - 'bar': traditional bar-type histogram
        - 'line': polygonal line
        See the example below for a visual explanation.
    bottom : float
        The bottom baseline of the plot.
    guess : bool
        Option works only for case len(x) == len(y).
        If True, the marginal bin edges of x will be guessed 
        with assuming equal bin. Otherwize x[0], x[-1] are used.
    orientation : ['horizontal', 'vertical'], optional
        Orientation.
    args, kwargs :
        same as those for
        `matplotlib.pyplot.plot` if `style` in ['line', 'step'], or
        `matplotlib.pyplot.plot.fill` if `style` in ['filled', 'bar'].

    Example
    -------
    np.random.seed(1)
    a = np.random.rand(50)
    b = np.linspace(0.1, 0.9, 6)
    h, bins = np.histogram(a, b)
    for i, style in enumerate(['default', 'step', 'filled', 'bar', 'line']):
        steps(bins + i, h, style=style, lw=2, bottom=1)
        plt.text(i + 0.5, 14, style)
    plt.xlim(0, 5)
    plt.ylim(-1, 16)
    '''
    style = kwargs.pop('style', 'default')
    bottom = kwargs.pop('bottom', 0)
    guess = kwargs.pop('guess', True)
    orientation = kwargs.pop('orientation', 'vertical')

    # a workaround for case 'line'
    if style == 'line':
        guess = True

    m, n = len(x), len(y)
    if m == n:
        if guess and m >= 2:
            xmin, xmax = x[0] * 1.5 - x[1] * 0.5, x[-1] * 1.5 - x[-2] * 0.5
        else:
            xmin, xmax = x[0], x[-1]
        x = np.hstack([xmin, (x[1:] + x[:-1]) * 0.5, xmax])
    elif m == n + 1:
        pass
    else:
        raise ValueError("x, y shape not matched.")

    if style == 'default':
        x, y = np.repeat(x, 2), np.repeat(y, 2)
        x = x[1:-1]
    elif style in ['step', 'filled']:
        x, y = np.repeat(x, 2), np.repeat(y, 2)
        y = np.hstack([bottom, y, bottom])
    elif style == 'bar':
        x, y = np.repeat(x, 3), np.repeat(y, 3)
        x, y = x[1:-1], np.hstack([y, bottom])
        y[::3] = bottom
    elif style == 'line':
        x = (x[1:] + x[:-1]) / 2
    else:
        raise ValueError("invalid style: %s" % style)

    if orientation == 'vertical':
        pass
    elif orientation == 'horizontal':
        x, y = y, x
    else:
        raise ValueError("orientation must be `vertical` or `horizontal`")

    if style in ['default', 'step', 'line']:
        return plt.plot(x, y, *args, **kwargs)
    else:
        return plt.fill(x, y, *args, **kwargs)


def cdfsteps(x, *args, **kwargs):
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
    side = kwargs.pop('side', 'left')
    normed = kwargs.pop('normed', True)
    sorted = kwargs.pop('sorted', False)
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
    return steps(x, h, *args, **kwargs)


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

    if weights is None:
        func = lambda x: quantile(x, nsig=[0, -1, -2, 1, 2],
                                  nmin=nmin, nanas=nanas)
        stats, edges, count = binstats(w, z, bins=bins, func=func)
    else:
        func = lambda x, weights: quantile(x, weights, nsig=[0, -1, -2, 1, 2],
                                           nmin=nmin, nanas=nanas)
        stats, edges, count = binstats(w, [z, weights], bins=bins, func=func)
    zs = np.atleast_2d(stats.T)
    ws = (edges[0][:-1] + edges[0][1:]) / 2.
    # ws = binstats(w, w, bins=bins, func=np.meidan, nmin=nmin).stats

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
