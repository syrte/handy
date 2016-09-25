from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


__all__ = ["make_rainbow", "make_cubehelix", "show_cmap", "make_cmap_ref"]


def make_rainbow(a=0.75, b=0.2, name='custom', register=True):
    def gfunc(a, b, c=1):
        def func(x):
            return c * np.exp(-0.5 * (x - a)**2 / b**2)
        return func

    cdict = {"red": gfunc(a, b),
             "green": gfunc(0.5, b),
             "blue": gfunc(1 - a, b)
             }
    cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    if register:
        plt.register_cmap(cmap=cmap)
    return cmap


def make_cubehelix(*args, **kwargs):
    """
    make_cubehelix(start=0.5, rotation=-1.5, gamma=1.0,
                   start_hue=None, end_hue=None,
                   sat=None, min_sat=1.2, max_sat=1.2,
                   min_light=0., max_light=1.,
                   n=256., reverse=False, name='custom_cubehelix')
    """
    from palettable.cubehelix import Cubehelix
    cmap = Cubehelix.make(*args, **kwargs).mpl_colormap
    register = kwargs.setdefault("register", True)
    if register:
        plt.register_cmap(cmap=cmap)
    return cmap


def show_cmap(cmap, coeff=(0.3, 0.59, 0.11)):
    coeff = np.asarray(coeff, 'f') / np.sum(coeff)
    coeff = dict(red=coeff[0], green=coeff[1], blue=coeff[2])

    data = cmap._segmentdata
    cdict = {}
    for c, d in data.items():
        if c == 'alpha':
            continue
        if callable(d):
            cdict[c] = d
        else:
            d = np.asarray(d)
            x, y = np.repeat(d[:, 0], 2), d[:, 1:].ravel()
            cdict[c] = interp1d(x, y)

    plt.figure(figsize=(6, 4))
    x = np.linspace(0, 1, 129)

    # components
    plt.axes([0.1, 0.25, 0.8, 0.65])
    for c in cdict:
        plt.plot(x, cdict[c](x), c=c, lw=2)

    # total perceived brightness
    y = np.sum([cdict[c](x) * coeff[c] for c in cdict], 0)
    plt.plot(x, y, 'c', lw=2, ls='--')

    # total brightness
    y = np.sum([cdict[c](x) / 3. for c in cdict], 0)
    plt.plot(x, y, 'k', lw=2, ls='--')

    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.xticks([])

    # cmap
    plt.axes([0.1, 0.1, 0.8, 0.15])
    plt.imshow([x], extent=[0, 1, 0, 1], vmin=0, vmax=1, aspect='auto', cmap=cmap)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.yticks([])


def make_cmap_ref(**fig_kwargs):
    maps = [m for m in mpl.cm.datad if not m.endswith("_r")]
    maps.sort()
    n = len(maps) + 1
    a = [np.linspace(0, 1, 101)]

    fig_kwargs.setdefault("figsize", (5, 20))
    fig = plt.figure(**fig_kwargs)
    fig.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.85)

    for i, m in enumerate(maps):
        ax = plt.subplot(n, 1, i + 1)
        ax.imshow(a, cmap=plt.get_cmap(m), aspect='auto')
        ax.text(1.05, 0.2, m, fontsize=10, transform=ax.transAxes)
        ax.axis("off")
    return fig
