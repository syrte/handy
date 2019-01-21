import numpy as np
from sklearn.neighbors import KDTree
from numba import vectorize

PI2 = np.pi * 2
SQRT_PI2 = np.sqrt(PI2)


@vectorize(["float64(float64, float64, float64)"])
def _norm1d(xi, x, xsig):
    X = (xi - x) / xsig
    p = np.exp(-0.5 * X**2) / (SQRT_PI2 * xsig)
    return p


@vectorize(["float64(float64, float64, float64, float64, float64, float64, float64)"])
def _norm2d(xi, yi, x, y, xsig, ysig, coef=0):
    X, Y = (xi - x) / xsig, (yi - y) / ysig
    coef2_1 = 1 - coef**2
    p = np.exp(-0.5 * (X**2 + Y**2 - 2 * X * Y * coef) / coef2_1) / (PI2 * xsig * ysig * coef2_1**0.5)
    return p


class AdapKDE1D:
    def __init__(self, x, n_eps=2, n_ngb=None):
        """
        Only Gaussian kernel supported which is inefficiency for very large data set.

        n_ngb:
            sqrt(len(x)) by default.
        """
        x = np.ravel(x)
        pts = x.reshape(-1, 1)

        if n_ngb is None:
            n_ngb = int(np.sqrt(len(x)))

        eps = KDTree(pts).query(pts, n_ngb)[0].T[-1] / (n_ngb / 2)
        xsig = n_eps * eps

        self.x = x
        self.xsig = xsig

    def density(self, xi):
        xi = xi[..., None]
        p = _norm1d(xi, self.x, self.xsig)
        return p.mean(-1)


class AdapKDE2D:
    def __init__(self, x, y, n_eps=2, n_ngb=None, scale=1):
        """
        Only Gaussian kernel supported which is inefficiency for very large data set.

        n_ngb:
            sqrt(len(x)) by default.
        scale: scalar
            1 by default. Naively, one should use scale=y.std()/x.std().
        """
        x, y = np.ravel(x), np.ravel(y)

        if scale == 'std':
            scale = y.std() / x.std()

        if scale == 1:
            pts = np.vstack([x, y]).T
        else:
            pts = np.vstack([x, y / scale]).T

        if n_ngb is None:
            n_ngb = int(np.sqrt(len(x)))

        eps = KDTree(pts).query(pts, n_ngb)[0].T[-1] / np.sqrt(n_ngb / np.pi)
        kern = n_eps * eps
        xsig = kern
        ysig = kern if scale == 1 else kern * scale

        self.x = x
        self.y = y
        self.xsig = xsig
        self.ysig = ysig

    def density(self, xi, yi):
        xi, yi = xi[..., None], yi[..., None]
        p = _norm2d(xi, yi, self.x, self.y, self.xsig, self.ysig, coef=0)
        return p.mean(-1)

    def densisty_mesh(self, xi, yi):
        xi, yi = xi[..., None, None], yi[..., None, :, None]
        p = _norm2d(xi, yi, self.x, self.y, self.xsig, self.ysig, coef=0)
        return p.mean(-1)
