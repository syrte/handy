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
        assert base > 0
        return np.log(mid(base**x)) / np.log(base)


def binstats(xs, ys, bins=10, func=np.mean, nmin=None):
    """
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
    assert callable(func)

    assert hasattr(xs, '__len__') and len(xs) > 0
    if np.isscalar(xs[0]):
        xs = [np.asarray(xs)]
        bins = [bins]
    else:
        xs = [np.asarray(x) for x in xs]
        assert len(xs[0]) > 0
    # `D`: number of dimensions
    # `N`: lenth of element along each dimension
    D, N = len(xs), len(xs[0])
    for x in xs:
        assert len(x) == N and x.ndim == 1

    assert hasattr(ys, '__len__') and len(ys) > 0
    if np.isscalar(ys[0]):
        ys = [np.asarray(ys)]
    else:
        ys = [np.asarray(y) for y in ys]
    for y in ys:
        assert len(y) == N

    if np.isscalar(bins):
        bins = [bins] * D
    else:
        assert len(bins) == D

    # prepare the edges
    edges = [None] * D
    for i, bin in enumerate(bins):
        if np.isscalar(bin):
            x = xs[i][np.isfinite(xs[i])]  # drop nan, inf
            assert len(x) > 0
            xmin, xmax = np.min(x), np.max(x)
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
            yselect = np.array([[] for y in ys])
            null = np.asarray(func(*yselect))
        except:
            yselect = [y[:1] for y in ys]
            test = np.asarray(func(*yselect))
            null = np.full_like(test, np.nan, dtype='float')

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
    bin_count = np.bincount(index, minlength=nbin + 1)

    # make statistics on each bin
    stats = np.empty((nbin,) + null.shape, dtype=null.dtype)
    count = np.empty((nbin,), dtype='int')
    for i in range(nbin):
        if bin_count[i]:
            ix = (index == i)
            yselect = [y[ix] for y in ys]
            stats[i] = func(*yselect)
            count[i] = len(yselect[0])
        else:
            stats[i] = null
            count[i] = 0

    if nmin is not None:
        stats[count < nmin] = null

    # change to proper shape
    stats = stats.reshape(dims + null.shape)
    count = count.reshape(dims)
    return BinStats(stats, edges, count)


def nanquantile(a, weights=None, q=None, nsig=None, sorted=False, nmin=0, nanas=None):
    """
    nanas: None or scalar
    """
    a = np.asarray(a).ravel()
    ix = np.isnan(a)
    if ix.any():
        if nanas is None:
            ix = ~ix
            if weights is not None:
                weights = np.asarray(weights).ravel()
                assert a.shape == weights.shape
                weights = weights[ix]
            a = a[ix]
        else:
            a = np.array(a, dtype="float")
            a[ix] = float(nanas)
    return quantile(a, weights=weights, q=q, nsig=nsig, sorted=sorted, nmin=nmin)


def quantile(a, weights=None, q=None, nsig=None, sorted=False, nmin=0):
    '''
    Compute the quantile of the data.

    Parameters
    ----------
    a : array_like
        Input array.
    q : float in range of [0,1] (or sequence of floats), optional
        Quantile to compute. One of `q` and `nsig` must be specified. 
    nsig : float, optional
        Quantile in unit of standard diviation.
        If `q` is not specified, then `scipy.stats.norm.cdf(nsig)` is used.
    sorted : bool
        If True, then the input array is assumed to be in increasing order.
    nmin: int
        Set `nmin` if you want a more reliable result.
        Return `nan` when the tail probability is less than `nmin/a.size`.
        nmin = 0 will return NaN for q not in [0, 1].
        nmin >= 3 is recommended for statistical use.

    See Also
    --------
    numpy.percentile

    Todo
    ----
    Add keywords `axis`.
    '''
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = norm.cdf(nsig)
    else:
        raise ValueError('One of `q` and `nsig` must be specified.')

    a = np.asarray(a).ravel()
    if a.size == 0 or q.size == 0:
        return np.full_like(q, np.nan, dtype='float')

    if weights is None:
        if not sorted:
            a = np.sort(a)
        pcum = np.arange(0.5, a.size) / a.size
    else:
        w = np.asarray(weights).ravel()
        assert a.shape == w.shape
        if not sorted:
            ix = np.argsort(a)
            a, w = a[ix], w[ix]
        pcum = (np.cumsum(w) - 0.5 * w) / np.sum(w)

    res = np.interp(q, pcum, a)
    if nmin is not None:
        # nmin = 0 will assert return nan for q not in [0, 1]
        ix = np.fmin(q, 1 - q) * a.size < nmin
        if not np.isscalar(ix):
            res[ix] = np.nan
        elif ix:
            res = np.nan
    return res


def conflevel(p, weights=None, q=None, nsig=None, sorted=False):
    '''
    used for 2d contour.
    weights:
        Should be bin size/area of corresponding p. 
        Can be ignored for equal binning.
    '''

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    assert (q >= 0).all()

    p = np.asarray(p).ravel()
    if p.size == 0 or q.size == 0:
        return np.full_like(q, np.nan, dtype='float')

    if weights is None:
        if not sorted:
            p = np.sort(p)[::-1]
        pw = p
    else:
        w = np.asarray(weights).ravel()
        assert p.shape == w.shape
        if not sorted:
            ix = np.argsort(p)[::-1]
            p, w = p[ix], w[ix]
        pw = p * w
    pcum = (np.cumsum(pw) - 0.5 * pw) / np.sum(pw)

    res = np.interp(q, pcum, p)
    return res


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
    #print(s1, s2)
    assert np.allclose(s1, s2)

    s1 = binned_statistic_dd([x, y], z, 'sum', bins=[b, b])[0]
    s2 = binstats([x, y], z, bins=[b, b], func=np.sum)[0]
    #print(s1, s2)
    assert np.allclose(s1, s2)
