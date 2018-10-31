from __future__ import division, print_function, absolute_import
import numpy as np
from math import log10, floor


__all__ = ['slicer', 'argmax_nd', 'argmin_nd', 'indexed', 'argclip', 'amap',
           'atleast_nd', 'dyadic', 'altcumsum', 'altcumprod', 'siground',
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


def argmax_nd(a, axis=None):
    """Returns the indice of the maximum value.

    Examples
    --------
    a = np.random.rand(3, 4, 5)
    assert np.all(a[argmax_nd(a)] == a.max())
    assert np.all(a[argmax_nd(a, axis=1)] == a.max(axis=1))
    """
    a = np.asarray(a)
    ix = a.argmax(axis=axis)
    if axis is None:
        return np.unravel_index(ix, a.shape)
    else:
        shape = list(a.shape)
        shape.pop(axis)
        indices = list(np.indices(shape))
        indices.insert(axis, ix)
        return tuple(indices)


def argmin_nd(a, axis=None):
    """Returns the indice of the minimum value.

    Examples
    --------
    a = np.random.rand(3, 4, 5)
    assert np.all(a[argmin_nd(a)] == a.min())
    assert np.all(a[argmin_nd(a, axis=1)] == a.min(axis=1))
    """
    a = np.asarray(a)
    ix = a.argmin(axis=axis)
    if axis is None:
        return np.unravel_index(ix, a.shape)
    else:
        shape = list(a.shape)
        shape.pop(axis)
        indices = list(np.indices(shape))
        indices.insert(axis, ix)
        return tuple(indices)


def indexed(x, y, missing='raise', return_missing=False):
    """Find elements in an un-sorted array.
    Return index such that x[index] == y, the first index found is returned,
    when multiple indices satisfy this condition.

    Parameters
    ----------
    x : 1-D array_like
        Input array.
    y : array_like
        Values to search in `x`.
    missing : {'raise', 'ignore', 'mask' or int}
        The elements of `y` are present in `x` is named missing.
        If 'raise', a ValueError is raised for missing elements.
        If 'mask', a masked array is returned, where missing elements are masked out.
        If 'ignore', no missing element is assumed, and output is undefined otherwise.
        If integer, value set for missing elements.
    return_missing : bool, optional
        If True, also return the indices of the missing elements of `y`.

    Returns
    -------
    indices : ndarray, [y.shape], int
        The indices such that x[indices] == y
    indices_missing : ndarray, [y.shape], optional
        The indices such that y[indices_missing] not in x

    See Also
    --------
    searchsorted : Find elements in a sorted array.

    Notes
    -----
    This code is originally taken from
    https://stackoverflow.com/a/8251757/2144720 by HYRY
    https://github.com/EelcoHoogendoorn/Numpy_arraysetops_EP by Eelco Hoogendoorn
    """
    x, y = np.asarray(x), np.asarray(y)

    x_index = np.argsort(x)
    y_index_sorted = np.searchsorted(x[x_index], y, side='left')
    index = np.take(x_index, y_index_sorted, mode="clip")

    if missing != 'ignore' or return_missing:
        invalid = x[index] != y

    if missing != 'ignore':
        if missing == 'raise':
            if np.any(invalid):
                raise ValueError('Not all elements in `y` are present in `x`')
        elif missing == 'mask':
            index = np.ma.array(index, mask=invalid)
        else:
            index[invalid] = missing

    if return_missing:
        return index, invalid
    else:
        return index


def argclip(a, amin=None, amax=None, closed='both'):
    """argclip(a, amin, amax) == (a >= amin) & (a <= amax)

    Parameters
    ----------
    amin, amax : float
    closed : {'both', 'left', 'right', 'none'}
    """
    a = np.asarray(a)
    if closed == 'both':
        gt_min, lt_max = np.greater_equal, np.less_equal
    elif closed == 'left':
        gt_min, lt_max = np.greater_equal, np.less
    elif closed == 'right':
        gt_min, lt_max = np.greater, np.less_equal
    elif closed == 'none':
        gt_min, lt_max = np.greater, np.less
    else:
        raise ValueError(
            "keywords 'closed' should be one of 'both', 'left', 'right', 'none'.")

    if amin is None:
        if amax is None:
            return np.ones_like(a, dtype='bool')
        else:
            return lt_max(a, amax)
    else:
        if amax is None:
            return gt_min(a, amin)
        else:
            return gt_min(a, amin) & lt_max(a, amax)


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
