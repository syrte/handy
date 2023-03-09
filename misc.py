from __future__ import division, print_function, absolute_import
import numpy as np
from math import log10, floor
from copy import deepcopy


__all__ = ['slicer', 'keys', 'argmax_nd', 'argmin_nd', 'indexed', 'argclip', 'amap',
           'atleast_nd', 'assign_first', 'assign_last', 'dyadic', 'altcumsum', 'altcumprod',
           'extend_linspace', 'extend_geomspace', 'round_signif', 'almost_unique',
           'siground', 'AttrDict', 'DictToClass', 'DefaultDictToClass']


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


def keys(x):
    if hasattr(x, 'keys'):
        return list(x.keys())
    elif hasattr(x, 'dtype'):
        return x.dtype.names
    else:
        return list(x.__dict__.keys())


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


def assign_first(a, index, b):
    """a[index] = b, assign value by first occurrence of duplicate indices.
    Note that numpy itself does not guarantee the the iteration order of indexing assignment in general.
    """
    ix_unique, ix_first = np.unique(index, return_index=True)
    # np.unique: return index of first occurrence.
    # ix_unique = index[ix_first]
    # ref: https://stackoverflow.com/a/44826781/

    a[ix_unique] = b[ix_first]
    return a


def assign_last(a, index, b):
    """a[index] = b, assign value by last occurrence of duplicate indices.
    Note that numpy itself does not guarantee the the iteration order of indexing assignment in general.
    XXX: should use unique to achive this!
    """
    return assign_first(a, index[::-1], b[::-1])


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


def extend_linspace(x, min=None, max=None):
    """
    Extend a given linspace/arange array to cover [min, max].
    Updated: 2022-10-19.

    Example
        extend_linspace(np.arange(5), -1, 3)
        # array([-1,  0,  1,  2,  3,  4])
        extend_linspace(np.arange(5), None, None)
        # array([0, 1, 2, 3, 4])
        extend_linspace(np.arange(5)[::-1], min=-1)
        # array([ 4,  3,  2,  1,  0, -1])
        extend_linspace(np.arange(5), max=6)
        # array([0, 1, 2, 3, 4, 5, 6])
        extend_linspace(np.arange(5), min=6)
        # array([0, 1, 2, 3, 4])  # attention!
    """
    dx = x[1] - x[0]
    if dx < 0:
        min, max = max, min
    elif dx == 0:
        raise ValueError('dx != 0 is expected.')

    out = [x]
    if min is not None:
        x0 = np.arange(x[0] - dx, min - dx, -dx)[::-1]
        out = [x0] + out
    if max is not None:
        x1 = np.arange(x[-1] + dx, max + dx, dx)
        out = out + [x1]

    if len(out) == 1:
        return x
    else:
        return np.hstack(out)


def extend_geomspace(x, min=None, max=None):
    """
    Extend a given geomspace array to cover [min, max].
    Updated: 2022-10-19.

    Example
        r = np.geomspace(1e-6, 1e1, 1001)
        rmin, rmax = 1e-8, 1e2
        a = extend_geomspace(r, rmin, rmax)
        assert np.allclose(np.diff(np.log(a)).mean(), np.diff(np.log(r)).mean())
        assert np.allclose((a), np.geomspace(*(a[[0, -1]]), len(a)), rtol=1e-10, atol=1e-20)
        assert a[0] - rmin <= 1e-10 and a[-1] - rmax >= -1e-10
    """
    ln_min = min if min is None else np.log(min)
    ln_max = max if max is None else np.log(max)
    return np.exp(extend_linspace(np.log(x), ln_min, ln_max))


def round_signif(x, decimals):
    """
    Round to the given number of significant figures.
    ref: Scott Gigante, https://stackoverflow.com/a/59888924/2144720

    Added: 2022-10-19, updated: 2022-10-20

    x : array_like
        Input data.
    decimals : int, optional
        Number of decimal places to round to.
    """
    x = np.asfarray(x)
    x_pos = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(decimals))
    mags = 10**(decimals - np.floor(np.log10(x_pos)))
    return np.around(x * mags) / mags

    # obsolete:
    # x = np.asfarray(x)
    # str = np.array2string(x, separator=',', formatter={'float_kind': lambda x: f"{x:.{decimals}e}"})
    # return eval(f"np.array({str}, dtype=x.dtype)")


def almost_unique(x, nrel=10, nabs=None, **kwargs):
    """
    Find the unique elements of an array for given precision.
    Added: 2022-10-19, updated: 2022-10-20

    x:
        Input number or array.
    nrel:
        Number of decimals in scientific notation (significant figures - 1).
    nabs:
        Number of absolute decimals.
    kwargs:
        np.unique arguments, including 'return_index', 'return_inverse', 
        'return_counts', 'axis'
    """
    if nrel is not None:
        x = round_signif(x, decimals=nrel)
    if nabs is not None:
        x = np.around(x, decimals=nabs)

    return np.unique(x, **kwargs)


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
    return re.findall(r"[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", string)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        """
        Dict with items accessible as attributes.

        Example
            d = AttrDict(a=[0, 1])
            d.a is d['a']  # True

        Reference
            https://stackoverflow.com/a/14620633
            https://stackoverflow.com/a/15774013

        Added: 2023-03-10
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __dir__(self):
        return list(self)

    def copy(self):
        """
        New objects are created for AttrDict items during copying,
        unlike dict.copy, which will not copy its items.
        Use copy.copy for the similar behavior as dict.copy.
        """
        new = AttrDict()
        for key, value in self.items():
            if isinstance(value, AttrDict):
                new[key] = value.copy()
            else:
                new[key] = value
        return new

    def __copy__(self):
        return AttrDict(self)

    def __deepcopy__(self, memo):
        new = AttrDict()
        memo[id(self)] = new
        for key, value in self.items():
            new[key] = deepcopy(value, memo)
        return new


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
