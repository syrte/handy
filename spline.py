from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline

__all__ = ['GridCubicSpline', 'CubicSplineExtrap']

factorial = np.array([1, 1, 2, 6])
deriv_coeff = [np.array([1, 1, 1, 1]),
               np.array([3, 2, 1]),
               np.array([6, 2]),
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


class CubicSplineExtrap(interpolate.PPoly):
    def __init__(self, x, y, method='spline', bc_type='not-a-knot', extrapolate='linear'):
        """
        CubicSpline with additional options for extrapolation.
        Updated: 2022-10-19

        x, y: array_like, shape (n,)
            Interpolating data.
        method: ['spline', 'akima', 'pchip']
            Interpolating method used.
        bc_type:
            Parameter of CubicSpline when method='spline'.
        extrapolate: [float, 'nan', 'const', 'linear', 'cubic'] or a 2-tuple of them
            Extrapolation type.

        Example
            from scipy.interpolate import PchipInterpolator, CubicSpline
            x = np.linspace(-0.7, 1, 11)
            a = np.linspace(-1.5, 2, 100)
            y = np.sin(x * pi)

            f0 = CubicSplineExtrap(x, y, extrapolate=('linear', 'const'))
            f1 = CubicSpline(x, y)
            f2 = PchipInterpolator(x, y)

            plt.figure(figsize=(8, 4))

            plt.subplot(121)
            plt.scatter(x, y)
            for i, f in enumerate([f0, f1, f2]):
                plt.plot(a, f(a), ls=['-', '--', ':'][i])
            plt.ylim(-2, 2)

            plt.subplot(122)
            for i, f in enumerate([f0, f1, f2]):
                plt.plot(a, f(a, nu=1) / np.pi, ls=['-', '--', ':'][i])
            plt.ylim(-2, 2)
        """
        initializer_dict = {'spline': interpolate.CubicSpline,
                            'akima': interpolate.Akima1DInterpolator,
                            'pchip': interpolate.PchipInterpolator}
        initializer = initializer_dict[method]

        if method == 'spline':
            spl = initializer(x, y, bc_type=bc_type)
        else:
            spl = initializer(x, y)

        if np.isscalar(extrapolate):
            extrapolate = (extrapolate, extrapolate)

        xs, cs = [spl.x], [spl.c]
        for i, ext in enumerate(extrapolate[:2]):
            if i == 0:
                xi, yi = x[0], y[0]
            else:
                xi, yi = x[-1], y[-1]

            if ext == 'cubic':
                continue
            elif ext == 'linear':
                di = spl(xi, nu=1)  # derivative at xi
                ci = np.array([[0, 0, di, yi]]).T
            elif ext == 'const':
                ci = np.array([[0, 0, 0, yi]]).T
            elif ext == 'nan' or np.isnan(ext):
                ci = np.array([[np.nan] * 4]).T
            else:
                ci = np.array([[0, 0, 0, float(ext)]]).T

            if i == 0:
                xs, cs = [xi, *xs], [ci, *cs]
            else:
                xs, cs = [*xs, xi], [*cs, ci]

        if len(xs) == 1:
            xs, cs = xs[0], cs[0]
        else:
            xs, cs = np.hstack(xs), np.hstack(cs)
        super().__init__(cs, xs, axis=spl.axis, extrapolate=True)
