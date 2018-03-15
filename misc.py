from __future__ import division, print_function, absolute_import
import numpy as np
from math import log10, floor


__all__ = ['slicer', 'argclip', 'amap', 'atleast_nd', 'dyadic',
           'altcumsum', 'altcumprod', 'siground',
           'DictToClass', 'DefaultDictToClass']


class Slicer(object):
    """Quick making slice object.

    Examples
    --------
    slicer = Slicer()

    slicer[0:5:1]
    # equivalent to slice(0, 5, 1)

    slicer[::1, ::2]
    # equivalent to (slice(None, None, 1), slice(None, None, 2))
    """

    def __getitem__(self, slice):
        return slice


slicer = Slicer()


def indices(x, y, missing='raise'):
    """Find indices such that x[indices] == y
    If multiple indices satisfy this condition, the first index found is returned.

    Parameters
    ----------
    x : indexable object
        items to search in
    y : indexable object
        items to search for
    missing : {'raise', 'ignore', 'mask' or int}
        if `missing` is 'raise', a KeyError is raised if not all elements of `y` are present in `x`
        if `missing` is 'mask', a masked array is returned, where items of `y` not present in `x` are masked out
        if `missing` is 'ignore', all elements of `y` are assumed to be present in `x`, and output is undefined otherwise
        if missing is an integer, x is used as a fill-value

    Returns
    -------
    indices : ndarray, [y.size], int
        indices such y x[indices] == y

    Notes
    -----
    May be regarded as a vectorized numpy equivalent of list.index

    c.f.
    https://stackoverflow.com/a/8251757/2144720 by HYRY
    https://github.com/EelcoHoogendoorn/Numpy_arraysetops_EP by Eelco Hoogendoorn
    """
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y, side='left')
    yindex = np.take(index, sorted_index, mode="clip")

    if missing != 'ignore':
        invalid = x[yindex] != y
        if missing == 'raise':
            if np.any(invalid):
                raise KeyError('Not all keys in `y` are present in `x`')
        elif missing == 'mask':
            yindex = np.ma.array(yindex, mask=invalid)
        else:
            yindex[invalid] = missing
    return yindex


def argclip(a, amin=None, amax=None):
    """argclip(a, amin, amax) == (a >= amin) & (a <= amax)
    """
    a = np.asarray(a)
    if amin is None:
        if amax is None:
            return np.ones_like(a, dtype='bool')
        else:
            return (a <= amax)
    else:
        if amax is None:
            return (a >= amin)
        else:
            return (a >= amin) & (a <= amax)


def amap(func, *args):
    '''Array version of build-in map
    amap(function, sequence[, sequence, ...]) -> array
    Examples
    --------
    >>> amap(lambda x: x**2, 1)
    array(1)
    >>> amap(lambda x: x**2, [1, 2])
    array([1, 4])
    >>> amap(lambda x,y: y**2 + x**2, 1, [1, 2])
    array([2, 5])
    >>> amap(lambda x: (x, x), 1)
    array([1, 1])
    >>> amap(lambda x,y: [x**2, y**2], [1,2], [3,4])
    array([[1, 9], [4, 16]])
    '''
    args = np.broadcast(*args)
    res = np.array([func(*arg) for arg in args])
    shape = args.shape + res.shape[1:]
    return res.reshape(shape)


def atleast_nd(a, nd, keep='right'):
    a = np.asanyarray(a)
    if a.ndim < nd:
        if keep == 'right' or keep == -1:
            shape = (1,) * (nd - a.ndim) + a.shape
        elif keep == 'left' or keep == 0:
            shape = a.shape + (1,) * (nd - a.ndim)
        else:
            raise ValueError("keep must be one of ['left', 'right', 0, -1]")
        return a.reshape(shape)
    else:
        return a


def raise_dims(a, n=0, m=0):
    a = np.asanyarray(a)
    shape = (1,) * n + a.shape + (1,) * m
    return a.reshape(shape)


def dyadic(a, b):
    """Dyadic product.
    a: shape (n1, ..., np)
    b: shape (m1, ..., mq)
    dyadic(a, b) : shape (n1, ..., np, m1, ..., mq)
    """
    a, b = np.asarray(a), np.asarray(b)
    shape = a.shape + (1,) * b.ndim
    return a.reshape(shape) * b


def shiftaxis(a, shift):
    """Roll the dimensions of an array.
    """
    a = np.asarray(a)
    if not -a.ndim <= shift < a.ndim:
        raise ValueError("shift should be in range [%d, %d)" %
                         (-a.ndim, a.ndim))
    axes = np.roll(range(a.ndim), shift)
    return a.transpose(axes)


def altcumsum(a, base=0, **kwargs):
    out = np.cumsum(a, **kwargs)
    if base is None:
        return out
    else:
        out[1:] = base + out[:-1]
        out[0] = base
        return out


def altcumprod(a, base=1, **kwargs):
    out = np.cumprod(a, **kwargs)
    if base is None:
        return out
    else:
        out[1:] = base * out[:-1]
        out[0] = base
        return out


def siground(x, n):
    x, n = float(x), int(n)
    if n <= 0:
        raise ValueError("n must be positive.")

    if x == 0:
        p = 0
    else:
        m = 10 ** floor(log10(abs(x)))
        x = round(x / m, n - 1) * m
        p = int(floor(log10(abs(x))))

    if -3 < p < n:
        return "{:.{:d}f}".format(x, n - 1 - p)
    else:
        return "{:.{:d}f}e{:+d}".format(x / 10**p, n - 1, p)


def find_numbers(string):
    """http://stackoverflow.com/a/29581287
    """
    import re
    return re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", string)


class DictToClass(object):
    def __init__(self, *args, **kwds):
        self.__dict__ = dict(*args, **kwds)


class DefaultDictToClass(object):
    def __init__(self, default_factory, *args, **kwds):
        from collections import defaultdict
        self.__dict__ = defaultdict(default_factory, *args, **kwds)

    def __getattr__(self, key):
        return self.__dict__[key]


def is_scalar(x):
    """
    >>> np.isscalar(np.array(1))
    False
    >>> is_scalar(np.array(1))
    True
    """
    if np.isscalar(x):
        return True
    elif isinstance(x, np.ndarray):
        return not x.ndim
    else:
        return False
        # return hasattr(x, "__len__")
