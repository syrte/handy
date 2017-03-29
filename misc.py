from __future__ import division, print_function, absolute_import
import numpy as np
from math import log10, floor


__all__ = ['amap', 'atleast_nd', 'dyadic', 'altcumsum', 'altcumprod',
           'siground', 'DictToClass', 'DefaultDictToClass']


def amap(func, *args):
    '''array version of build-in map
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
    assert n > 0
    if x == 0:
        return ("%%.%if" % (n - 1)) % x
    m = 10 ** floor(log10(abs(x)))
    x = round(x / m, n - 1) * m
    p = floor(log10(abs(x)))
    if -3 < p < n:
        return ("%%.%if" % (n - 1 - p)) % x
    else:
        return ("%%.%ife%%+i" % (n - 1)) % (x / 10**p, p)


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
