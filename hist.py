from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt
from .stats import binstats, binquantile, generate_bins

__all__ = ['pcolorshow', 'hist_stats', 'hist2d_stats', 'steps',
           'cdfsteps', 'pdfsteps', 'compare', 'compare_violin']


def _pcolorshow_args(x, m):
    """Helper function for `pcolorshow`.
    Check the shape of input and return its range.
    """
    if x.ndim != 1:
        raise ValueError("unexpected array dimensions")
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
    Similar to `plt.hist` but show the binned statistics instead of
    simple number count.

    Parameters
    ----------
    x, y, bins, func, nmin :
        See doc of `binstats`.
    style : {'plot' | 'scatter' | 'step'}
        Style of line.
    kwargs :
        Parameters for style above.

    Example
    -------
    import numpy as np
    n = 10000
    x, s = np.random.randn(2, n)
    y = x * 2 + s / 2
    hist_stats(x, y, func=lambda x:np.percentile(x, [50, 15, 85]),
            ls=['-', '--', '--'], lw=[2, 1, 1], color=['k', 'b', 'b'])
    """
    stats, edges, count = binstats(x, y, bins=bins, func=func, nmin=nmin)
    stats = np.atleast_2d(stats)
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
    """
    Similar to `plt.hist2d` but show the binned statistics instead of
    simple number count.

    Parameters
    ----------
    x, y :
        Coordinates of points.
    z :
        Data for statistics. 
    bins, func, nmin :
        See doc of `binstats`.
    kwargs :
        `pcolormesh` parameters

    """
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
    """steps(x, y, *args, style='line', bottom=0, guess=True, 
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
        with assuming equal bin. Otherwise x[0], x[-1] are used.
    orientation : ['horizontal' | 'vertical'], optional
        Orientation.
    args, kwargs :
        Same as `plt.plot` if `style` in ['default', 'step', 'line'], or
        same as `plt.fill` if `style` in ['filled', 'bar'].

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
    """
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
    """cdfsteps(x, *args, weights=None, side='left', 
                normed=True, sorted=Fasle, **kwargs)

    Parameters
    ----------
    x:
        Input.
    weights : array, Optional
        Weighting.
    side: ['left' | 'right']
        'left': ascending steps,
        'right' : descending steps.
    normed: bool
        If normalize to 1.
    sorted: bool
        Set True, if x follows increasing order.
        Otherwise, sorting will be performed to x.
    """
    weights = kwargs.pop('weights', None)
    side = kwargs.pop('side', 'left')
    normed = kwargs.pop('normed', True)
    sorted = kwargs.pop('sorted', False)
    assert side in ['right', 'left']

    x = np.asarray(x).ravel()
    if not sorted:
        x = np.sort(x)

    if weights is None:
        h = np.arange(0., x.size + 1.)
    else:
        h = np.hstack([0., np.cumsum(weights)])
    if normed:
        h = h / h[-1]
    if side == 'right':
        h = h[::-1]
    x = np.hstack([x[0], x, x[-1]])
    return steps(x, h, *args, **kwargs)


def pdfsteps(x, *args, **kwds):
    sorted = kwds.pop('sorted', False)

    x = np.asarray(x).ravel()
    if not sorted:
        x = np.sort(x)
    h = 1. / x.size / np.diff(x)
    return steps(x, h, *args, border=True, **kwds)


def _expand_args(args, i_idx, j_key):
    """Helper function for `compare`.
    Expand the args for given index.
    """
    res = {}
    for k, v in args.items():
        if np.isscalar(v):
            res[k] = v
        elif isinstance(v, dict):
            res[k] = v[j_key]
        else:
            res[k] = v[i_idx]
    return res


def compare(x, y, xbins=None, ybins=None, weights=None, nmin=3, nanas=None,
            dots=[0], ebar=[], line=[0, 1, 2], fill=[], zorder=2,
            dots_args={}, ebar_args={}, fill_args={}, **line_args):
    """Show the correlation between two data sets.
    Plot the median and 1, 2 sigma regions of the conditional distribution
    p(y|x) for given x bins or p(x|y) for given y bins.

    Parameters
    ----------
    x, y : 1-D sequences
        Data sets to compare.
    xbins, ybins : int or 1-D sequences
        Binning edges. Only one of them can be given.
    weights :
        Weights of data.
    nmin, nanas :
        See doc of `binquantile`.

    Example
    -------
    import numpy as np
    n = 10000
    x, s = np.random.randn(2, n)
    y = x * 2 + s / 2
    compare(x, y, 10, dots=0, line=[0, 1], fill=1, ebar=1)
    """
    # format inputs
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()

    if ybins is None:
        if xbins is None:
            xbins = 10
        w, z, bins = x, y, xbins
    else:
        if xbins is not None:
            raise ValueError("Only one of 'xbins' or 'ybins' can be given.")
        w, z, bins = y, x, ybins

    dots = [dots] if np.isscalar(dots) else dots
    ebar = [ebar] if np.isscalar(ebar) else ebar
    line = [line] if np.isscalar(line) else line
    fill = [fill] if np.isscalar(fill) else fill
    if 0 in ebar or 0 in fill:
        raise ValueError("`ebar` and `fill` can only set to 1 or 2")

    # prepare data
    zs = binquantile(w, z, bins=bins, nsig=[0, -1, -2, 1, 2], shape='stats',
                     weights=weights, nmin=nmin, nanas=nanas).stats
    ws = binquantile(w, w, bins=bins, q=0.5,
                     weights=weights, nmin=nmin, nanas=nanas).stats

    # default style
    dots_args.setdefault('s', 20)
    dots_args.setdefault('c', 'k')
    dots_args.setdefault('edgecolor', 'none')
    dots_args.setdefault('zorder', zorder + 0.3)

    ebar_args.setdefault('ecolor', {1: 'k', 2: 'c'})
    ebar_args.setdefault('fmt', 'none')
    ebar_args.setdefault('zorder', zorder + 0.2)

    line_args.setdefault('fmt', {0: 'k-', 1: 'b--', 2: 'g-.'})
    line_args.setdefault('zorder', zorder)

    fill_args.setdefault('color', {1: 'b', 2: 'g'})
    fill_args.setdefault('alpha', {1: 0.5, 2: 0.3})
    fill_args.setdefault('edgecolor', 'none')
    fill_args.setdefault('zorder', zorder - 1)

    # prepare plots
    ax = plt.gca()
    if xbins is not None:
        xs, ys = [ws] * 5, zs
        fill_between = ax.fill_between
        err = 'yerr'
    else:
        xs, ys = zs, [ws] * 5
        fill_between = ax.fill_betweenx
        err = 'xerr'

    # dots
    for i, k in enumerate(dots):
        args = _expand_args(dots_args, i, k)
        ax.scatter(xs[k], ys[k], **args)

    # ebar
    for i, k in enumerate(ebar):
        args = _expand_args(ebar_args, i, k)
        args[err] = zs[0] - zs[k], zs[k + 2] - zs[0]
        args['zorder'] = args['zorder'] + 0.1 * (1.5 - k)
        ax.errorbar(xs[0], ys[0], **args)

    # line
    for i, k in enumerate(line):
        args = _expand_args(line_args, i, k)
        fmt = args.pop('fmt', '')
        ax.plot(xs[k], ys[k], fmt, **args)
        if k != 0:
            args.pop('label', None)
            ax.plot(xs[k + 2], ys[k + 2], fmt, **args)

    # fill
    for i, k in enumerate(fill):
        args = _expand_args(fill_args, i, k)
        fill_between(ws, zs[k], zs[k + 2], **args)

    return


def compare_violin(x, y, xbins=None, ybins=None, nmin=1, nmax=10000, **kwargs):
    """Show the conditional violin plot for two data sets.
    """
    if ybins is None:
        if xbins is None:
            xbins = 10
    else:
        if xbins is not None:
            raise ValueError("Only one of 'xbins' or 'ybins' can be given.")
        vert = not kwargs.pop('vert', True)
        return compare_violin(y, x, xbins=ybins, nmin=nmin, nmax=nmax,
                              vert=vert, **kwargs)

    nmin, nmax = int(nmin), int(nmax)
    kwargs.setdefault('showmedians', True)

    ix = (np.isfinite(x) & np.isfinite(y)).nonzero()
    x, y = x[ix], y[ix]

    bins = generate_bins(x, xbins)
    bins_mid = (bins[:-1] + bins[1:]) * 0.5

    idx = bins.searchsorted(x, side='right') - 1
    dat, pos = [], []
    for i in range(len(bins) - 1):
        a, b = y[idx == i], bins_mid[i]
        if a.size > nmax:
            a = np.random.choice(a, nmax, replace=False)
        if a.size >= nmin:
            dat.append(a)
            pos.append(b)
    plt.violinplot(dataset=dat, positions=pos, **kwargs)
