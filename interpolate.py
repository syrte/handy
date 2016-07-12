from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

__all__ = ["EqualGridInterpolator"]


class EqualGridInterpolator(object):
    """
    Interpolation on a equal spaced regular grid in arbitrary dimensions.
    Fock from https://github.com/JohannesBuchner/regulargrid
    """

    def __init__(self, points, values, order=1, padding='constant', fill_value=np.nan):
        '''
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the equal regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        order : int
            The order of the spline interpolation, default is 1. 
            The order has to be in the range 0-5.
        padding : str
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        fill_value : number, optional
            If provided, the value to use for points outside of the interpolation domain.
        '''
        assert order in range(6)
        self.order = order
        self.fill_value = fill_value
        self.padding = padding

        values = np.asfarray(values)
        assert len(points) == values.ndim

        points = [np.asarray(p) for p in points]
        for i, p in enumerate(points):
            assert len(p) == values.shape[i]
            assert p[1] - p[0] > 0
            assert np.allclose(np.diff(p), p[1] - p[0])

        if order > 1:
            # if more speedup is needed, add keywords `output=np.float32`?
            self.coeffs = spline_filter(values, order=order)
        self.grid = tuple(points)
        self.values = values
        self.edges = tuple([p[0] for p in points])
        self.steps = tuple([(p[1] - p[0]) for p in points])

    def __call__(self, xi):
        '''
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at.
        '''
        xi = [(x - xmin) / dx
              for x, xmin, dx in zip(xi, self.edges, self.steps)]
        input = self.coeffs if self.order > 1 else self.values
        return map_coordinates(input, xi, order=self.order,
                               mode=self.padding, cval=self.fill_value,
                               prefilter=False)


if __name__ == "__main__":
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from matplotlib import pyplot as plt

    # Example 1
    f = lambda x, y: np.sin(x / 2) - np.sin(y)
    x, y = np.linspace(-2, 3, 5), np.linspace(-3, 2, 6)
    z = f(*np.meshgrid(x, y))

    xi, yi = np.meshgrid(np.linspace(-2, 3, 50), np.linspace(-3, 2, 60))
    zi1 = EqualGridInterpolator((x, y), z.T, order=1)((xi, yi))
    zi2 = RegularGridInterpolator((x, y), z.T)((xi, yi))

    assert np.allclose(zi1, zi2)

    f = lambda x, y: x * y
    mid = lambda x: (x[1:] + x[:-1]) / 2.
    x = np.linspace(0, 2, 10)
    y = np.linspace(0, 2, 15)
    x_, y_ = mid(x), mid(y)
    z = f(*np.meshgrid(x_, y_))

    # Example 2
    xi = np.linspace(0, 2, 40)
    yi = np.linspace(0, 2, 60)
    xi_, yi_ = mid(xi), mid(yi)

    zi1 = EqualGridInterpolator((x_, y_), z.T, order=3)(np.meshgrid(xi_, yi_))
    zi2 = EqualGridInterpolator((x_, y_), z.T)(np.meshgrid(xi_, yi_))
    zi3 = EqualGridInterpolator((x_, y_), z.T, padding='nearest')(np.meshgrid(xi_, yi_))

    plt.figure(figsize=(9, 9))
    plt.viridis()
    plt.subplot(221)
    plt.pcolormesh(x, y, z)
    plt.subplot(222)
    plt.pcolormesh(xi, yi, np.ma.array(zi1, mask=np.isnan(zi1)))
    plt.subplot(223)
    plt.pcolormesh(xi, yi, np.ma.array(zi2, mask=np.isnan(zi2)))
    plt.subplot(224)
    plt.pcolormesh(xi, yi, zi3)
    plt.show()
