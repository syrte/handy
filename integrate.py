from __future__ import division
import numpy as np

__all__ = ['trapz1d', 'simps1d', 'trapz2d', 'simps2d']

"""
Implementation notes
    http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html

Timing
```
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.integrate import trapz, simps
a = np.random.rand(1000, 1025)

assert np.allclose(trapz(a), trapz1d(a), atol=0, rtol=1e-8)
assert np.allclose(simps(a), simps1d(a), atol=0, rtol=1e-8)

%timeit np.sum(a)
%timeit trapz(a)
%timeit simps(a)
%timeit trapz1d(a)
%timeit simps1d(a)
```
"""


def slice_set(ix, ndim, axis):
    ix_list = [slice(None)] * ndim
    ix_list[axis] = ix
    return tuple(ix_list)


def trapz1d(y, dx=1.0, axis=-1):
    y = np.asarray(y)
    ndim = y.ndim

    ix0 = slice_set(0, ndim, axis)
    ix1 = slice_set(-1, ndim, axis)
    ix2 = slice_set(slice(1, -1), ndim, axis)

    out = ((y[ix0] + y[ix1]) + 2 * y[ix2].sum(axis)) * (dx / 2)
    return out


def simps1d(y, dx=1.0, axis=-1, even='avg'):
    y = np.asarray(y)
    ndim = y.ndim

    # when shape of y is even
    if y.shape[axis] % 2 == 0:
        if even == 'first':
            ix1 = slice_set(-1, ndim, axis)
            ix2 = slice_set(-2, ndim, axis)
            return simps1d(y[:-1], dx, axis) + 0.5 * dx * (y[ix1] + y[ix2])
        elif even == 'last':
            ix0 = slice_set(0, ndim, axis)
            ix1 = slice_set(1, ndim, axis)
            return simps1d(y[1:], dx, axis) + 0.5 * dx * (y[ix0] + y[ix1])
        else:
            return 0.5 * (simps1d(y[:-1], dx, axis, 'first') +
                          simps1d(y[1:], dx, axis, 'last'))

    ix0 = slice_set(0, ndim, axis)
    ix1 = slice_set(-1, ndim, axis)
    ixo = slice_set(slice(1, -1, 2), ndim, axis)  # odd
    ixe = slice_set(slice(2, -2, 2), ndim, axis)  # even

    out = (y[ix0] + y[ix1] + 4 * y[ixo].sum(axis) + 2 * y[ixe].sum(axis)) * (dx / 3)
    return out


def sum2d(a):
    """sum of last two dimensions
    """
    return a.reshape(*a.shape[:-2], -1).sum(-1)


def trapz2d(z, dx=1, dy=1):
    """integrate over last two dimensions

    >>> trapz2d(np.ones((5, 5)))
    16.0
    """
    z = np.asarray(z)
    ix = slice(1, -1)

    s1 = (z[..., 0, 0] + z[..., 0, -1] + z[..., -1, 0] + z[..., -1, -1])
    s2 = 2 * (z[..., 0, ix].sum(-1) + z[..., -1, ix].sum(-1) +
              z[..., ix, 0].sum(-1) + z[..., ix, -1].sum(-1))
    s3 = 4 * sum2d(z[..., ix, ix])

    out = (s1 + s2 + s3) * (dx * dy / 4)
    return out


def simps2d(z, dx=1, dy=1):
    """integrate over last two dimensions

    >>> simps2d(np.ones((5, 5)))
    16.0
    """
    z = np.asarray(z)
    nx, ny = z.shape[-2:]
    if nx % 2 != 1 or ny % 2 != 1:
        raise ValueError('input array should be odd shape')

    ixo = slice(1, -1, 2)  # odd
    ixe = slice(2, -2, 2)  # even

    # corner points, with weight 1
    s1 = (z[..., 0, 0] + z[..., 0, -1] + z[..., -1, 0] + z[..., -1, -1])

    # edges excluding corners, with weight 2 or 4
    s2 = 2 * (z[..., 0, ixe].sum(-1) + z[..., -1, ixe].sum(-1) +
              z[..., ixe, 0].sum(-1) + z[..., ixe, -1].sum(-1))
    s3 = 4 * (z[..., 0, ixo].sum(-1) + z[..., -1, ixo].sum(-1) +
              z[..., ixo, 0].sum(-1) + z[..., ixo, -1].sum(-1))

    # interior points, with weight 4, 8 or 16
    s4 = (4 * sum2d(z[..., ixe, ixe]) + 16 * sum2d(z[..., ixo, ixo]) +
          8 * sum2d(z[..., ixe, ixo]) + 8 * sum2d(z[..., ixo, ixe]))

    out = (s1 + s2 + s3 + s4) * (dx * dy / 9)
    return out
