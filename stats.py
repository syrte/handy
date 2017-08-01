from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
from scipy.stats import norm
from collections import namedtuple
from itertools import product

__all__ = ['mid', 'binstats', 'quantile', 'nanquantile', 'conflevel',
           'binquantile', 'alterbinstats']

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
    elif base <= 0:
        raise ValueError("`base` must be positive")
    elif base == 1:
        return mid(x)
    else:
        return np.log(mid(base**x)) / np.log(base)


def generate_bins(x, bins):
    """Generate bins automatically.
    Helper function for binstats.
    """
    if bins is None:
        bins = 10
    if np.isscalar(bins):
        ix = np.isfinite(x)
        if not ix.all():
            x = x[ix]  # drop nan, inf
        if len(x) > 0:
            xmin, xmax = np.min(x), np.max(x)
        else:
            # failed to determine range, so use 0-1.
            xmin, xmax = 0, 1
        if xmin == xmax:
            xmin = xmin - 0.5
            xmax = xmax + 0.5
        return np.linspace(xmin, xmax, bins + 1)
    else:
        return np.asarray(bins)


def binstats(xs, ys, bins=10, func=np.mean, nmin=1, shape='stats'):
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
        User-defined function which takes a sequence of arrays as input,
        and outputs a scalar or an array with *fixed shape*. This function
        will be called on the values in each bin func(y1, y2, ...).
        Empty bins will be represented by func([], [], ...) or NaNs if this
        returns an error.
    nmin: int
        The bin with data point counts smaller than nmin will be 
        treated as empty bin.
    shape : 'bins' | 'stats'
        Put which axes first in the result:
            'bins' - the shape of bins
            'stats' - the shape of func output

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
    if shape != 'bins' and shape != 'stats':
        raise ValueError("`shape` must be 'bins' or 'stats'")

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
    edges = [generate_bins(x, bin) for x, bin in zip(xs, bins)]
    dims = tuple(len(edge) - 1 for edge in edges)
    nbin = np.prod(dims)

    # statistical value for empty bin
    with warnings.catch_warnings():
        # Numpy generates a warnings for mean/std/... with empty list
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            yselect = [y[:0] for y in ys]
            null = np.asarray(func(*yselect))
        except Exception:
            yselect = [y[:1] for y in ys]
            temp = np.asarray(func(*yselect))
            null = np.full_like(temp, np.nan, dtype='float')

    # get the index
    indexes = np.empty((D, N), dtype='int')
    for i in range(D):
        ix = np.searchsorted(edges[i], xs[i], side='right') - 1
        ix[(xs[i] >= edges[i][-1])] = -1  # give outliers index < 0
        ix[(xs[i] == edges[i][-1])] = -1 + dims[i]  # include points on edge
        indexes[i] = ix

    # convert nd-index to flattened index
    index = indexes[0]
    ix_out = (indexes < 0).any(axis=0)  # outliers
    for i in range(1, D):
        index *= dims[i]
        index += indexes[i]
    index[ix_out] = nbin  # put outliers in an extra bin

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
    if shape == 'bins':
        stats = stats.reshape(dims + null.shape)
    elif shape == 'stats':
        stats = np.moveaxis(stats, 0, -1).reshape(null.shape + dims)
    count = count.reshape(dims)
    return BinStats(stats, edges, count)


def quantile(a, weights=None, q=None, nsig=None, origin='middle',
             axis=None, keepdims=False, sorted=False, nmin=0,
             nanas=None, shape='stats'):
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
        Quantile in unit of standard deviation.
        Ignored when `q` is given.
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
    shape : 'data' | 'stats'
        Put which axes first in the result:
            'data' - the shape of data
            'stats' - the shape of `q` or `nsig`
        Only works for case where axis is not None.

    Returns
    -------
    quantile : scalar or ndarray
        The first axes of the result corresponds to the quantiles,
        the rest are the axes that remain after the reduction of `a`.

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
    >>> quantile(x, q=0.5, axis=1).shape
    (3,)
    >>> quantile(x, q=0.5, axis=1, keepdims=True).shape
    (3, 1)
    >>> quantile(x, q=[0.2, 0.8], axis=1).shape
    (2, 3)
    >>> quantile(x, q=[0.2, 0.8], axis=1, shape='stats').shape
    (3, 2)
    '''
    # check input
    a = np.asarray(a)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError("`weights` should have same shape as `a`.")

    # convert nsig to q
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
        q = np.asarray(q)
    else:
        raise ValueError("One of `q` and `nsig` must be specified.")

    # check q and nmin
    # nmin = 0 will assert return nan for q not in [0, 1]
    if nmin is not None and a.size:
        tol = 1 - 1e-5
        if axis is None:
            threshold = nmin * tol / a.size
        else:
            threshold = nmin * tol / a.shape[axis]
        ix = np.fmin(q, 1 - q) < threshold
        if np.any(ix):
            q = np.array(q, dtype="float")  # make copy of `q`
            q[ix] = np.nan

    # result shape
    if axis is None:
        res_shape = q.shape
    else:
        extra_dims = list(a.shape)
        if keepdims:
            extra_dims[axis] = 1
        else:
            extra_dims.pop(axis)

        if shape == 'data':
            res_shape = tuple(extra_dims) + q.shape
        elif shape == 'stats':
            res_shape = q.shape + tuple(extra_dims)
        else:
            raise ValueError("`shape` must be 'data' or 'stats'")

    # quick return for empty input array
    if a.size == 0 or q.size == 0:
        return np.full(res_shape, np.nan, dtype='float')

    # handle the nans
    # nothing to do when nanas is None.
    if nanas is None:
        pass
    elif nanas != 'ignore':
        ix = np.isnan(a)
        if ix.any():
            a = np.array(a, dtype="float")  # make copy of `a`
            a[ix] = float(nanas)
        nanas = None
    elif nanas == 'ignore' and axis is None:
        ix = np.isnan(a)
        if ix.any():
            ix = (~ix).nonzero()
            a = a[ix]
            if weights is not None:
                weights = weights[ix]
        nanas = None
    # if nanas == 'ignore' and axis is not None:
        # leave the nans to later recursion on axis.

    if axis is None:
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
        return res

    else:
        # handle the axis
        # move the target axis to the last and flatten the rest axes for map
        a_ = np.moveaxis(a, axis, -1).reshape(-1, a.shape[axis])
        if weights is None:
            func = lambda x: quantile(x, q=q, sorted=sorted,
                                      nmin=None, nanas=nanas)
            res = map(func, a_)
        else:
            w_ = np.moveaxis(weights, axis, -1).reshape(a_.shape)
            func = lambda x, w: quantile(x, weights=w, q=q, sorted=sorted,
                                         nmin=None, nanas=nanas)
            res = map(func, a_, w_)

        if shape == 'data':
            res = np.array(res).reshape(res_shape)
        elif shape == 'stats':
            # put the shape of quantile first
            res = np.moveaxis(res, 0, -1).reshape(res_shape)
        return res


def nanquantile(a, weights=None, q=None, nsig=None, origin='middle',
                axis=None, keepdims=False, sorted=False, nmin=0,
                nanas='ignore', shape='stats'):
    """Compute the quantile of the data, ignoring NaNs by default.

    Refer to `quantile` for full documentation.

    See Also
    --------
    quantile : Not ignoring NaNs by default.
    """
    return quantile(a, weights=weights, q=q, nsig=nsig, origin=origin,
                    axis=axis, keepdims=keepdims, sorted=sorted, nmin=nmin,
                    nanas=nanas, shape=shape)


def conflevel(p, weights=None, q=None, nsig=None, sorted=False, norm=1):
    '''Calculate the lower confidence bounds with given levels for 2d contour.
    Be careful when q is very small or many numbers repeat in p.

    conflevel is equivalent to
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
        Quantile in unit of standard deviation.
        If `q` is not specified, then `scipy.stats.norm.cdf(nsig)` is used.
    sorted : bool
        If True, then the input array is assumed to be in increasing order.
    norm : float in (0, 1]
        The weights will be normalized as sum(p * weights) = norm.
        This is useful when the data points do not cover full probability.
        See `Examples` for more detail.

    See Also
    --------
    quantile

    Examples
    --------
    >>> n = 10000
    >>> x, y = np.random.randn(2, n)
    >>> xbin, ybin = np.linspace(-2, 2, 10), np.linspace(-2, 2, 15)
    >>> area = np.diff(xbin)[:, np.newaxis] * np.diff(ybin)
    >>> h = np.histogram2d(x, y, [xbin, ybin])[0]
    >>> p = h / n / area
    >>> levels = conflevel(p, area, q=[0.2, 0.5, 0.8], norm=h.sum()/n)
    >>> plt.pcolormesh(xbin, ybin, p.T)
    >>> plt.contour(mid(xbin), mid(ybin), p.T, levels,
        colors='k', linewidths=2, linestyles=['-', '--', '-.'])
    Note that h.sum() is not necessary equal to n.
    '''
    if q is not None:
        q = 1 - np.asarray(q)

    if weights is None:
        weights = p
    else:
        weights = weights * p

    if norm == 1:
        pass
    elif 0 < norm < 1:
        # add an extra "pseudo" point to cover the probability out of box.
        p = np.append(0, p)
        weights = np.append((1 - norm) / norm * np.sum(weights), weights)
    else:
        raise ValueError("`norm` must be in (0, 1].")

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


def binquantile(x, y, bins=10, weights=None, q=None, nsig=None,
                origin='middle', nmin=0, nanas=None, shape='stats'):
    """
    x, y : array_like
        Input data.
    bins : array_like or int
        Bins to compute quantile.
    weights : array_like, optional
        Weighting of data.
    q : float or float array in range of [0,1], optional
        Quantile to compute. One of `q` and `nsig` must be specified.
    nsig : float, optional
        Quantile in unit of standard deviation. Ignored when `q` is given.
    origin, nmin, nanas:
        Refer to `quantile` for full documentation.
    shape : {'bins', 'stats'}
        Put which axes first in the result:
            'bins' - the shape of bins
            'stats' - the shape of quantiles
    """
    if weights is None:
        func = lambda a: quantile(a, q=q, nsig=nsig, origin=origin,
                                  nmin=nmin, nanas=nanas)
        stats = binstats(x, [y], bins=bins,
                         func=func, shape=shape)
    else:
        func = lambda a, w: quantile(a, w, q=q, nsig=nsig, origin=origin,
                                     nmin=nmin, nanas=nanas)
        stats = binstats(x, [y, weights], bins=bins,
                         func=func, shape=shape)

    return stats


def alterbinstats(xs, ys, bins=10, func=np.mean, nmin=1, shape='stats'):
    """Make binned statistics for multidimensional data.
    It allows discontinuous or overlap binning like [[1,5], [3,7], [5,9]]
    at the cost of speed.

    Refer to `binstats` for full documentation.
    """
    # check the inputs
    if not callable(func):
        raise TypeError('`func` must be callable.')
    if shape != 'bins' and shape != 'stats':
        raise ValueError("`shape` must be 'bins' or 'stats'")

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
            edge = np.linspace(xmin, xmax, bin + 1)
        else:
            edge = np.asarray(bin)
        if edge.ndim == 1:
            edge = np.stack([edge[:-1], edge[1:]], -1)
        edges[i] = edge
    dims = tuple(len(edge) for edge in edges)

    # statistical value for empty bin
    with warnings.catch_warnings():
        # Numpy generates a warnings for mean/std/... with empty list
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            yselect = [y[:0] for y in ys]
            null = np.asarray(func(*yselect))
        except Exception:
            yselect = [y[:1] for y in ys]
            temp = np.asarray(func(*yselect))
            null = np.full_like(temp, np.nan, dtype='float')

    # prepare the results
    count = np.empty(dims, dtype='int')
    stats = np.empty(dims + null.shape, dtype=null.dtype)

    # prepare the bin index
    idx = [None] * D
    strides = np.array(count.strides) / count.itemsize
    iter_ij = product(*[range(n) for n in dims])

    # cache indexes of last dimension
    last_index = [None] * dims[-1]
    for j in range(dims[-1]):
        ix0 = (xs[-1] >= edges[-1][j, 0])
        ix1 = (xs[-1] <= edges[-1][j, 1])
        last_index[j] = ix0 & ix1

    # make statistics on each bin
    for n, ij in enumerate(iter_ij):
        idx[-1] = last_index[ij[-1]]
        if D > 1:
            for i, j in enumerate(ij[:-1]):
                if n % strides[i] == 0:
                    ix0 = (xs[i] >= edges[i][j, 0])
                    ix1 = (xs[i] <= edges[i][j, 1])
                    idx[i] = ix0 & ix1
                    if i > 0:
                        idx[i] = idx[i] & idx[i - 1]
            idx[-1] = idx[-1] & idx[-2]

        ix = idx[-1].nonzero()[0]
        count[ij] = ix.size

        if count[ij] >= nmin:
            yselect = [y[ix] for y in ys]
            stats[ij] = func(*yselect)
        else:
            stats[ij] = null

    # change to proper shape
    if shape == 'stats':
        stats = stats.reshape(-1, null.shape)
        stats = np.moveaxis(stats, 0, -1).reshape(null.shape + dims)
    return BinStats(stats, edges, count)


WStats = namedtuple('WStats',
                    'avg, std, med, sigs, sig1, sig2, sig3,'
                    'x, w, var, mean, median,'
                    'sig1a, sig1b, sig2a, sig2b, sig3a, sig3b')


def wstats(x, weights=None, axis=None, keepdims=False):
    """
    a = wstats(randn(100))
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

    return WStats(avg, std, med, sig, sig1, sig2, sig3,
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
