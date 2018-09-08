from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ['GridCubicSpline']

factorial = np.array([1, 1, 2, 6])
deriv_coeff = [np.array([1, 1, 1, 1]), np.array([3, 2, 1]), np.array([6, 2]),
               np.array([6])]
integ_power = np.array([4, 3, 2, 1], dtype=float).reshape(-1, 1)
integ_coeff = 1 / integ_power


class GridCubicSpline:
    """
    Examples
    --------
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    f1 = CubicSpline(x, y)
    f2 = GridCubicSpline(x, y)

    for k in range(4):
        assert np.allclose(f1.derivative(k)(x), f2.derivative(k))
    assert np.allclose(f1.antiderivative()(x), f2.antiderivative())

    f2.antiderivative(forward=True) + f2.antiderivative(forward=False)
    """

    def __init__(self, x, y, bc_type='not-a-knot'):
        "check docstring of scipy.interpolate.CubicSpline"
        self.spline = CubicSpline(x, y, bc_type=bc_type)
        self.x = np.asfarray(x)
        self.y = y
        self.c = self.spline.c

    def derivative(self, k=1):
        "k = 0, 1, 2, 3"
        x, c = self.x, self.c
        y_der = np.empty_like(x)

        if k < 2:
            y_der[:-1] = c[3 - k]
        else:
            y_der[:-1] = c[3 - k] * factorial[k]
        y_der[-1] = np.poly1d(c[:4 - k, -1] * deriv_coeff[k])(x[-1] - x[-2])

        return y_der

    def antiderivative(self, forward=True):
        "forward: integrate from x[0] to x,\nbackward: integrate from x[-1] to x."
        x, c = self.x, self.c
        y_int = np.empty_like(x)

        dy_int = (integ_coeff * c * np.diff(x)**integ_power).sum(0)
        if forward:
            y_int[0] = 0
            y_int[1:] = np.cumsum(dy_int)
        else:
            y_int[-1] = 0
            y_int[:-1] = np.cumsum(dy_int[::-1])[::-1]

        return y_int
