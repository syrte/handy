from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from scipy.stats import gaussian_kde


__all__ = ['denscatter', 'circles', 'ellipses', 'rectangles', 'lines', 'cov_ellipses']


def denscatter(x, y, scale=None, sort=False, **kwargs):
    """
    scale : None, float or callable
        change the density
            z = z * scale for float, 
            z = scale(z) for callable
    sort : bool
        If `sort`, then the points with higher density are plotted on the top.
    """
    kwargs.setdefault('edgecolor', 'none')

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    if np.isscalar(scale):
        z = z * scale
    elif callable(scale):
        z = scale(z)

    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    return plt.scatter(x, y, c=z, ** kwargs)


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles. 
    Similar to plt.scatter, but the size of circles are in data scale.

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def ellipses(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of ellipses. 
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    ellipses(a, a, w=4, h=a, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, rot)
    patches = [Ellipse((x_, y_), w_, h_, rot_)
               for x_, y_, w_, h_, rot_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def rectangles(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of rectangles.

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of rectangles.
    w, h : scalar or array_like, shape (n, )
        Width, Height.
        `h` is set to be equal to `w` by default, ie. squares.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    rectangles(a, a, w=5, h=6, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w
    d = np.sqrt(np.square(w) + np.square(h)) / 2.
    t = np.deg2rad(rot) + np.arctan2(h, w)
    x, y = x - d * np.cos(t), y - d * np.sin(t)

    zipped = np.broadcast(x, y, w, h, rot)
    patches = [Rectangle((x_, y_), w_, h_, rot_)
               for x_, y_, w_, h_, rot_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def lines(xy, c='b', vmin=None, vmax=None, **kwargs):
    """
    xy : sequence of array 
        Coordinates of points in lines.
        `xy` is a sequence of array (line0, line1, ..., lineN) where
            line = [(x0, y0), (x1, y1), ... (xm, ym)]
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, linewidth(lw), linestyle(ls), norm, cmap, transform, etc.

    Returns
    -------
    collection : `~matplotlib.collections.LineCollection`
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))

    collection = LineCollection(xy, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def cov_ellipses(x, y, cov_mat=None, cov_tri=None, q=None, nsig=None, **kwargs):
    """Draw covariance error ellipses.

    Parameters
    ----------
    x, y : array (n,)
        Center of covariance ellipses.
    cov_mat : array (n, 2, 2), optional
        Covariance matrix.
    cov_tri : list of array (n,), optional
        Covariance matrix in flat form of (xvar, yvar, xycov).
    q : scalar or array
        Wanted (quantile) probability enclosed in error ellipse.
    nsig : scalar or array
        Probability in unit of standard error. Eg. `nsig = 1` means `q = 0.683`.
    kwargs :
        `ellipses` properties.
        Eg. c, vmin, vmax, alpha, edgecolor(ec), facecolor(fc), 
        linewidth(lw), linestyle(ls), norm, cmap, transform, etc.

    Reference
    ---------
    [1]: http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix
    [2]: http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    """
    from scipy.stats import norm, chi2

    if cov_mat is not None:
        cov_mat = np.asarray(cov_mat)
    elif cov_tri is not None:
        assert len(cov_tri) == 3
        cov_mat = np.array([[cov_tri[0], cov_tri[2]],
                            [cov_tri[2], cov_tri[1]]])
        cov_mat = cov_mat.transpose(range(2, cov_mat.ndim) + range(2))
        # Roll the first two dimensions (2, 2) to end.
    else:
        raise ValueError('One of `cov_mat` and `cov_tri` should be specified.')

    x, y = np.asarray(x), np.asarray(y)
    if not (cov_mat.shape[:-2] == x.shape == y.shape):
        raise ValueError('The shape of x, y and covariance are incompatible.')
    if not (cov_mat.shape[-2:] == (2, 2)):
        raise ValueError('Invalid covariance matrix shape.')

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    rho = chi2.ppf(q, 2)
    rho = rho.reshape(rho.shape + (1,) * x.ndim)  # raise dimentions

    val, vec = np.linalg.eigh(cov_mat)
    w = 2 * np.sqrt(val[..., 0] * rho)
    h = 2 * np.sqrt(val[..., 1] * rho)
    rot = np.degrees(np.arctan2(vec[..., 1, 0], vec[..., 0, 0]))

    return ellipses(x, y, w, h, rot=rot, **kwargs)
    """cov_cross
    assert q.ndim == 0
    res = []
    xy = np.stack([x, y], -1)[..., None, :]
    w_line = xy + vec[..., None, :, 0] * w[..., None, None] * np.array([[-0.5], [0.5]])
    h_line = xy + vec[..., None, :, 1] * h[..., None, None] * np.array([[-0.5], [0.5]])
    res.append(lines(w_line, **kwargs))
    res.append(lines(h_line, **kwargs))

    return res
    """
