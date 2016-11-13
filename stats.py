from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
from scipy.stats import norm
from collections import namedtuple

__all__ = ['mid', 'binstats', 'quantile', 'nanquantile', 'conflevel']

BinStats = namedtuple('BinStats',
                      ('stats', 'edges', 'count'))


def mid(x, base=None):
    '''Return mean value of adjacent member of an array.
    Useful for plotting bin counts.
    '''
    x = np.asarray(x)
    if base is None:
        return (x[1:] + x[:-1]) / 2.
    elif base == 'log':
        return np.exp(mid(np.log(x)))
    elif base == 'exp':
        return np.log(mid(np.exp(x)))
    else:
        if base <= 0:
            raise ValueError("`base` must be positive")
        return np.log(mid(base**x)) / np.log(base)


def binstats(xs, ys, bins=10, func=np.mean, nmin=1):
    """Make binned statistics for multidimensional data.
    xs: array_like or list of array_like
        Data to histogram passed as a sequence of D arrays of length N, or
        as an (D, N) array.
    ys: array_like or list of array_like
        The data on which the `func` will be computed.  This must be
        the same shape as `x`, or a list of sequences - each with the same
        shape as `x`.  If `values` is a list, the `func` will treat them as
        multiple arguments.
    bins : sequence or int, optional
        The bin specification must be in one of the following forms:
          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (n1, n2, ... = bins).
          * The number of bins for all dimensions (n1 = n2 = ... = bins).
    func: callable
        User-defined function which takes a sequece of arrays as input,
        and outputs a scalar or an array with *fixing shape*. This function
        will be called on the values in each bin func(y1, y2, ...).
        Empty bins will be represented by func([], [], ...) or NaNs if this
        returns an error.
    nmin: int
        The bin with points counts smaller than nmin will be treat as empty bin.

    Returns
    -------
    stats: ndarray
        The values of the selected statistic in each bin.
    edges: list of ndarray
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension.
    count: ndarray
        Number count in each bin.

    See Also
    --------
    numpy.histogramdd, scipy.stats.binned_statistic_dd

    Example
    -------
    import numpy as np
    from numpy.random import rand
    x, y = rand(2, 1000)
    b = np.linspace(0, 1, 11)
    binstats(x, y, 10, np.mean)
    binstats(x, y, b, np.mean)
    binstats(x, [x, y], 10, lambda x, y: np.mean(y - x))
    binstats(x, [x, y], 10, lambda x, y: [np.median(x), np.std(y)])
    """
    # check the inputs
    if not callable(func):
        raise TypeError('`func` must be callable.')

    if len(xs) == 0:
        raise ValueError("`xs` must be non empty")
    if len(ys) == 0:
        raise ValueError("`ys` must be non empty")
    if np.isscalar(xs[0]):
        xs = [xs]
        bins = [bins]
    if np.isscalar(ys[0]):
        ys = [ys]
    if np.isscalar(bins):
        bins = [bins] * len(xs)

    xs = [np.asarray(x) for x in xs]
    ys = [np.asarray(y) for y in ys]

    D, N = len(xs), len(xs[0])
    # `D`: number of dimensions
    # `N`: number of elements along each dimension
    for x in xs:
        if len(x) != N:
            raise ValueError("x should have the same length")
        if x.ndim != 1:
            raise ValueError("x should be 1D array")
    for y in ys:
        if len(y) != N:
            raise ValueError("y should have the same length as x")
    if len(bins) != D:
        raise ValueError("bins should have the same number as xs")

    # prepare the edges
    edges = [None] * D
    for i, bin in enumerate(bins):
        if np.isscalar(bin):
            x = xs[i][np.isfinite(xs[i])]  # drop nan, inf
            if len(x) > 0:
                xmin, xmax = np.min(x), np.max(x)
            else:
                # failed to determine range, so use 0-1.
                xmin, xmax = 0, 1
            if xmin == xmax:
                xmin = xmin - 0.5
                xmax = xmax + 0.5
            edges[i] = np.linspace(xmin, xmax, bin + 1)
        else:
            edges[i] = np.asarray(bin)
    dims = tuple(len(edge) - 1 for edge in edges)
    nbin = np.prod(dims)

    # statistical value for empty bin
    with warnings.catch_warnings():
        # Numpy generates a warnings for mean/std/... with empty list
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            yselect = [y[:0] for y in ys]
            null = np.asarray(func(*yselect))
        except:
            yselect = [y[:1] for y in ys]
            temp = np.asarray(func(*yselect))
            null = np.full_like(temp, np.nan, dtype='float')

    # get the index
    indexes = np.empty((D, N), dtype='int')
    for i in range(D):
        ix = np.searchsorted(edges[i], xs[i], side='right') - 1
        ix[(xs[i] >= edges[i][-1])] = -1  # give outlier index < 0
        ix[(xs[i] == edges[i][-1])] = -1 + dims[i]  # include points on edge
        indexes[i] = ix

    # convert nd-index to flattend index
    index = indexes[0]
    ix_out = (indexes < 0).any(axis=0)  # outlier
    for i in range(1, D):
        index *= dims[i]
        index += indexes[i]
    index[ix_out] = nbin  # put outlier in an extra bin

    # make statistics on each bin
    stats = np.empty((nbin,) + null.shape, dtype=null.dtype)
    count = np.bincount(index, minlength=nbin + 1)[:nbin]
    for i in range(nbin):
        if count[i] >= nmin:
            ix = (index == i).nonzero()
            yselect = [y[ix] for y in ys]
            stats[i] = func(*yselect)
        else:
            stats[i] = null

    # change to proper shape
    stats = stats.reshape(dims + null.shape)
    count = count.reshape(dims)
    return BinStats(stats, edges, count)


def quantile(a, weights=None, q=None, nsig=None, origin='middle',
             axis=None, keepdims=False, sorted=False, nmin=0,
             nanas=None):
    '''Compute the quantile of the data.
    Be careful when q is very small or many numbers repeat in a.

    Parameters
    ----------
    a : array_like
        Input array.
    weights : array_like, optional
        Weighting of a.
    q : float or float array in range of [0,1], optional
        Quantile to compute. One of `q` and `nsig` must be specified.
    nsig : float, optional
        Quantile in unit of standard diviation.
        Igored when `q` is given.
    origin : ['middle'| 'high'| 'low'], optional
        Control how to interpret `nsig` to `q`.
    axis : int, optional
        Axis along which the quantiles are computed. The default is to
        compute the quantiles of the flattened array.
    sorted : bool
        If True, the input array is assumed to be in increasing order.
    nmin : int or None
        Return `nan` when the tail probability is less than `nmin/a.size`.
        Set `nmin` if you want to make result more reliable.
        - nmin = None will turn off the check.
        - nmin = 0 will return NaN for q not in [0, 1].
        - nmin >= 3 is recommended for statistical use.
        It is *not* well defined when `weights` is given.
    nanas : None, float, 'ignore'
        - None : do nothing. Note default sorting puts `nan` after `inf`.
        - float : `nan`s will be replaced by given value.
        - 'ignore' : `nan`s will be excluded before any calculation.

    See Also
    --------
    numpy.percentile
    conflevel

    Examples
    --------
    >>> np.random.seed(0)
    >>> x = np.random.randn(3, 100)

    >>> quantile(x, q=0.5)
    0.024654858649703838
    >>> quantile(x, nsig=0)
    0.024654858649703838
    >>> quantile(x, nsig=1)
    1.0161711040272021
    >>> quantile(x, nsig=[0, 1])
    array([ 0.02465486,  1.0161711 ])

    >>> quantile(np.abs(x), nsig=1, origin='low')
    1.024490097937702
    >>> quantile(-np.abs(x), nsig=1, origin='high')
    -1.0244900979377023

    >>> quantile(x, q=0.5, axis=1)
    array([ 0.09409612,  0.02465486, -0.07535884])
    >>> quantile(x, q=0.5, axis=1, keepdims=True).shape
    (3, 1)
    '''
    # check input
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        if origin == 'middle':
            q = norm.cdf(nsig)
        elif origin == 'high':
            q = 2 - 2 * norm.cdf(nsig)
        elif origin == 'low':
            q = 2 * norm.cdf(nsig) - 1
        else:
            raise ValueError("`origin` should be 'center', 'high' or 'low'.")
    else:
        raise ValueError("One of `q` and `nsig` must be specified.")

    a = np.asarray(a)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError("`weights` should have same shape as `a`.")

    # result shape
    if axis is None:
        shape = q.shape
    else:
        shape = list(a.shape)
        if keepdims:
            shape[axis] = 1
        else:
            shape.pop(axis)
        shape = q.shape + tuple(shape)

    # quick return for empty input array
    if a.size == 0 or q.size == 0:
        return np.full(shape, np.nan, dtype='float')

    # handle the nans
    if nanas is None:
        pass
    elif nanas != 'ignore':
        ix = np.isnan(a)
        if ix.any():
            a = np.array(a, dtype="float")  # make copy of `a`
            a[ix] = float(nanas)
        nanas = None  # mark as done the nan conversion.
    # if nanas == 'ignore':
    elif axis is None:
        ix = np.isnan(a)
        if ix.any():
            ix = (~ix).nonzero()
            a = a[ix]
            if weights is not None:
                weights = weights[ix]
        nanas = None
    else:
        # leave the nans to later recursion on axis.
        pass

    # handle the axis
    if axis is not None:
        a = np.moveaxis(a, axis, -1).reshape(-1, a.shape[axis])
        if weights is None:
            func = lambda x: quantile(x, weights=None, q=q, axis=None,
                                      sorted=sorted, nmin=nmin, nanas=nanas)
            res = map(func, a)
        else:
            weights = np.moveaxis(weights, axis, -1).reshape(a.shape)
            func = lambda x, w: quantile(x, weights=w, q=q, axis=None,
                                         sorted=sorted, nmin=nmin, nanas=nanas)
            res = map(func, a, weights)
        # put the shape of quantile first
        res = np.moveaxis(res, 0, -1).reshape(shape)
        return res

    # sort and interpolate
    a = a.ravel()
    if weights is None:
        if not sorted:
            a = np.sort(a)
        pcum = np.arange(0.5, a.size) / a.size
    else:
        weights = weights.ravel()
        if not sorted:
            ix = np.argsort(a)
            a, weights = a[ix], weights[ix]
        pcum = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    res = np.interp(q, pcum, a)

    # check nmin
    # nmin = 0 will assert return nan for q not in [0, 1]
    if nmin is not None:
        tol = 1e-5
        ix = np.fmin(q, 1 - q) < float(nmin) / a.size - tol
        if np.any(ix):
            if hasattr(res, 'ndim'):
                res[ix] = np.nan
            else:
                res = np.nan
    return res


def nanquantile(a, weights=None, q=None, nsig=None, origin='middle',
                axis=None, keepdims=False, sorted=False, nmin=0,
                nanas='ignore'):
    """
    nanas: None or scalar
    """
    return quantile(a, weights=weights, q=q, nsig=nsig, origin=origin,
                    axis=axis, keepdims=keepdims, sorted=sorted, nmin=nmin,
                    nanas=nanas)


def conflevel(p, weights=None, q=None, nsig=None, sorted=False):
    '''Calculate the confidence levels for 2d contour.
    conflevel is equivent to
        quantile(p, weights=p*weights, q=1-q)
    or
        quantile(p, weights=p*weights, nsig=nsig, origin='high')

    Parameters
    ----------
    p : array_like
        Input array. Usually `p` is the probability in grids.
    weights:
        Should be bin size/area of corresponding p.
        Can be ignored for equal binning.
    q : float or float array in range of [0,1], optional
        Quantile to compute. One of `q` and `nsig` must be specified.
    nsig : float, optional
        Quantile in unit of standard diviation.
        If `q` is not specified, then `scipy.stats.norm.cdf(nsig)` is used.
    sorted : bool
        If True, then the input array is assumed to be in increasing order.

    See Also
    --------
    quantile

    Notes
    -----
    Be careful when q is very small or many numbers repeat in p.

    `conflevel` will always normalize as sum(p * weights) = 1, thus 
    the result is biased if the data points do not cover full probability.
    This may be compensated as in following example:
        bin_area = xbin_width * ybin_width
        p = np.histogram2d(x, y, bins)[0]
        p = p / len(x) / bin_area
        w = p * bin_area
        # add an "psudo" point to cover the probability out of box.
        p = np.append(p, 0)
        w = np.append(w, 1 - np.sum(w))
        levels = quantile(p, w, nsig=[1, 2, 3], origin='high')
        plt.contour(..., p, levels, ...)
    '''
    if q is not None:
        q = 1 - np.asarray(q)

    if weights is None:
        weights = p
    else:
        weights = weights * p

    return quantile(p, weights=weights, q=q, nsig=nsig, origin='high',
                    sorted=sorted, nmin=None)


def hdregion(x, p, weights=None, q=None, nsig=None):
    """Highest Density Region (HDR)
    find x s.t.
        p(x) = sig_level
    weights:
        Should be bin size of corresponding p.
        Can be ignored for equal binning.
    """
    from .optimize import findroot
    from .misc import amap

    assert (np.diff(x) >= 0).all()

    levels = conflevel(p, weights=weights, q=q, nsig=nsig)
    x = np.hstack([x[0], x, x[-1]])
    p = np.hstack([0, p, 0])

    intervals = amap(lambda lv: findroot(lv, x, p), levels)
    return intervals


WStats = namedtuple('WStats', 'avg, std, med, sig1, sig2, sig3,'
                    'x, w, var, mean, median,'
                    'sig1a, sig1b, sig2a, sig2b, sig3a, sig3b')


def wstats(x, weights=None, axis=None, keepdims=False):
    """
    a = wstats(randn(100))
    a[:6]
    """
    x = np.asarray(x)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != x.shape:
            raise ValueError("weights must have same shape with x")
    else:
        weights = np.ones_like(x)
    w = weights / np.sum(weights, axis=axis, keepdims=True)

    avg = np.sum(x * w, axis=axis, keepdims=keepdims)
    var = np.sum(x**2 * w, axis=axis, keepdims=keepdims) - avg**2
    std = var**0.5
    sig = quantile(x, w, nsig=[0, -1, 1, -2, 2, -3, 3], axis=axis, keepdims=keepdims)

    med, sig1a, sig1b, sig2a, sig2b, sig3a, sig3b = sig
    sig1, sig2, sig3 = sig[1:3], sig[3:5], sig[5:7]
    mean, median = avg, med

    return WStats(avg, std, med, sig1, sig2, sig3,
                  x, weights, var, mean, median,
                  sig1a, sig1b, sig2a, sig2b, sig3a, sig3b)


if __name__ == '__main__':
    import numpy as np
    from numpy.random import randn
    x, y, z = randn(3, 1000)
    b = np.linspace(-2, 2, 11)
    binstats(x, y, 10, np.mean)
    binstats(x, y, b, np.mean)
    binstats(x, y, b, np.mean, nmin=100)
    binstats(x, [y, z], 10, lambda x, y: np.mean(x + y))
    binstats(x, [y, z], 10, lambda x, y: [np.mean(x), np.std(y)])
    binstats([x, y], z, (10, 10), np.mean)
    binstats([x, y], z, [b, b], np.mean)
    binstats([x, y], [z, z], 10, lambda x, y: [np.mean(x), np.std(y)])

    b1 = np.linspace(-2, 2, 11)
    b2 = np.linspace(-2, 2, 21)
    binstats([x, y], [z, z], [b1, b2], lambda x, y: [np.mean(x), np.std(y)])

    from scipy.stats import binned_statistic_dd
    s1 = binned_statistic_dd(x, x, 'std', bins=[b])[0]
    s2 = binstats(x, x, bins=b, func=np.std)[0]
    # print(s1, s2)
    assert np.allclose(s1, s2)

    s1 = binned_statistic_dd([x, y], z, 'sum', bins=[b, b])[0]
    s2 = binstats([x, y], z, bins=[b, b], func=np.sum)[0]
    # print(s1, s2)
    assert np.allclose(s1, s2)

    a = quantile(np.arange(10), q=[0.1, 0.5, 0.85])
    assert np.allclose(a, [0.5, 4.5, 8.])
    a = np.arange(12).reshape(3, 4)
    b = quantile(a, q=0.5, axis=0)
    c = quantile(a, q=0.5, axis=1)
    assert np.allclose(b, [4., 5., 6., 7.])
    assert np.allclose(c, [1.5, 5.5, 9.5])
