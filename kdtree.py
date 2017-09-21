import numpy as np
from collections import namedtuple

__all__ = ['query_radius_periodic']


def repeat_periodic(points, boxsize):
    """Repeat data to mock periodic boundaries.
    points (m, n) -> repeated_points (m, 3**n, n)

    Each point (x1, ..., xn) will have to 3**n copies:
        (x1, ..., xn)
        (x1, ..., xn-L)
        (x1, ..., xn+L)
        ...
        (x1+L, ..., xn-L)
        (x1+L, ..., xn+L)
    """
    from itertools import product
    import numpy as np

    points = np.asarray(points)
    ndim = points.shape[-1]

    shift = np.array(list(product([0, -1, 1], repeat=ndim))) * boxsize
    repeated_points = points[..., np.newaxis, :] + shift

    return repeated_points


def query_radius_periodic(tree, points, radius, boxsize=None, merge=False):
    """
    tree: sklearn.neighbors.KDTree instance
    points : array-like
        An array of points to query.
    radius : float or array-like
        Distance within which neighbors are returned.
    boxsize : float or array-like
        Periodic boxsize.
    merge : bool
        If True, all outputs will be merged into single array.
    """
    ndim = tree.data.shape[-1]
    nrep = 3**ndim
    if points.shape[-1] != ndim:
        raise ValueError("Incompatible shape.")

    if boxsize is None:
        periodic = False
    else:
        periodic = True
        points = repeat_periodic(points, boxsize=boxsize).reshape(-1, ndim)
        if not np.isscalar(radius):
            radius = np.repeat(radius, nrep)

    idx, dis = tree.query_radius(points, radius, return_distance=True)
    cnt = np.array(map(len, idx))
    if periodic:
        cnt = cnt.reshape(-1, nrep).sum(-1)

    if merge:
        idx = np.concatenate(idx)
        dis = np.concatenate(dis)
    elif periodic:
        idx = np.array(map(np.concatenate, idx.reshape(-1, nrep)))
        dis = np.array(map(np.concatenate, dis.reshape(-1, nrep)))

    type = namedtuple('KDTreeQuery', ['count', 'index', 'distance'])
    return type(cnt, idx, dis)
