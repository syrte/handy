from __future__ import division
import numpy as np

__all__ = ['trapz2d', 'simps2d']


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

    io = slice(1, -1, 2)
    ie = slice(2, -2, 2)

    # corner points, with weight 1
    s1 = (z[..., 0, 0] + z[..., 0, -1] + z[..., -1, 0] + z[..., -1, -1])

    # edges excluding corners, with weight 2 or 4
    s2 = 2 * (z[..., 0, ie].sum(-1) + z[..., -1, ie].sum(-1) +
              z[..., ie, 0].sum(-1) + z[..., ie, -1].sum(-1))
    s3 = 4 * (z[..., 0, io].sum(-1) + z[..., -1, io].sum(-1) +
              z[..., io, 0].sum(-1) + z[..., io, -1].sum(-1))

    # interior points, with weight 4, 8 or 16
    s4 = (4 * sum2d(z[..., ie, ie]) + 16 * sum2d(z[..., io, io]) +
          8 * sum2d(z[..., ie, io]) + 8 * sum2d(z[..., io, ie]))

    out = (s1 + s2 + s3 + s4) * (dx * dy / 9)
    return out
