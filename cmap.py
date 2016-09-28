from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


__all__ = ["make_rainbow", "make_cubehelix", "show_cmap", "make_cmap_ref"]


def grayify_cmap(cmap, register=False):
    """Return a grayscale version of the colormap
    copy from https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    """
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3]**2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    cmap_g = cmap.from_list(cmap.name + "_g", colors, cmap.N)
    if register:
        plt.register_cmap(cmap=cmap_g)
    return cmap_g


def make_rainbow(a=0.75, b=0.2, name='custom_rainbow', register=False):
    """
    Use a=0.7, b=0.2 for a darker end.

    when 0.5<=a<=1.5, should have b >= (a-0.5)/2 or 0 <= b <= (a-1)/3
    when 0<=a<=0.5, should have b >= (0.5-a)/2 or 0<= b<= -a/3
    to assert the monoique

    To show the parameter dependencies interactively in notebook
    ```
    %matplotlib inline
    from ipywidgets import interact
    def func(a=0.75, b=0.2):
        cmap = gene_rainbow(a=a, b=b)
        show_cmap(cmap)
    interact(func, a=(0, 1, 0.05), b=(0.1, 0.5, 0.05))
    ```
    """
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
        plt.rc('image', cmap=cmap.name)
    return cmap


def make_cubehelix(*args, **kwargs):
    """make_cubehelix(start=0.5, rotation=-1.5, gamma=1.0,
                   start_hue=None, end_hue=None,
                   sat=None, min_sat=1.2, max_sat=1.2,
                   min_light=0., max_light=1.,
                   n=256., reverse=False, name='custom_cubehelix')
    """
    from palettable.cubehelix import Cubehelix
    cmap = Cubehelix.make(*args, **kwargs).mpl_colormap
    register = kwargs.setdefault("register", False)
    if register:
        plt.register_cmap(cmap=cmap)
        plt.rc('image', cmap=cmap.name)
    return cmap


def show_cmap(cmap, coeff=(0.3, 0.59, 0.11)):
    coeff = np.asarray(coeff, 'f').reshape(-1, 1) / np.sum(coeff)

    x = np.linspace(0, 1, 257)
    rgba = cmap(x).T

    plt.figure(figsize=(6, 4))
    plt.axes([0.1, 0.25, 0.7, 0.65])

    # components
    for c, y in zip(["red", "green", "blue"], rgba):
        plt.plot(x, y, lw=2, label=c, color=c)

    # alpha
    y = rgba[3]
    if not np.allclose(y, 1):
        plt.plot(x, y, lw=2, label="alpha", color='m', ls=":")

    # total brightness
    y = np.mean(rgba[:3], 0)
    plt.plot(x, y, 'k--', lw=2, label="L")

    # total perceived brightness
    y = np.sum(rgba[:3] * coeff, 0)
    plt.plot(x, y, 'c--', lw=2, label="L(eye)")

    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.xticks([])
    plt.legend(loc=(1, 0.1), frameon=False, handlelength=1.5)

    # cmap
    plt.axes([0.1, 0.1, 0.7, 0.15])
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
