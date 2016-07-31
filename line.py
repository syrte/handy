from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = ["abline", "ABLine2D"]


def abline(a, b, *args, **kwargs):
    """
    a, b: scalar or tuple
        Acceptable forms are
        y0, b:
            y = y0 + b * x
        (x0, y0), b:
            y = y0 + b * (x - x0)
        (x0, y0), (x1, y1):
            y = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    Additional arguments are passed to the <matplotlib.lines.Line2D> constructor.

    It will have wrong behaiver when the axis-limits are
    changed by setting ticks. This can be corrected by call
    `xlim` to reset the limits.
    Refer this issue:
        https://github.com/matplotlib/matplotlib/issues/6863
    """
    return ABLine2D(a, b, *args, **kwargs)


class ABLine2D(Line2D):
    """
    Draw a line based on a point and slope or two points. 
    Originally fock from http://stackoverflow.com/a/14348481/2144720 by ali_m
    """

    def __init__(self, a, b, *args, **kwargs):
        """
        a, b: scalar or tuple
            Acceptable forms are
            y0, b:
                y = y0 + b * x
            (x0, y0), b:
                y = y0 + b * (x - x0)
            (x0, y0), (x1, y1):
                y = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
        Additional arguments are passed to the <matplotlib.lines.Line2D> constructor.

        It will have wrong behaiver when the axis-limits are
        changed by setting ticks. This can be corrected by call
        `xlim` to reset the limits.
        Refer this issue:
            https://github.com/matplotlib/matplotlib/issues/6863
        """
        if np.isscalar(a):
            assert np.isscalar(b)
            point = (0, a)
            slope = b
        elif np.isscalar(b):
            assert len(a) == 2
            point = a
            slope = b
        else:
            assert len(a) == len(b) == 2
            point = a
            slope = (b[1] - a[1]) / np.float64(b[0] - a[0])
            # use np.float64 to get inf when dividing by zero

        if 'axes' in kwargs:
            ax = kwargs['axes']
        else:
            ax = plt.gca()
        if not ('color' in kwargs or 'c' in kwargs):
            kwargs.update(ax._get_lines.prop_cycler.next())

        super(ABLine2D, self).__init__([], [], *args, **kwargs)
        self._point = tuple(point)
        self._slope = slope

        # draw the line for the first time
        ax.add_line(self)
        self._auto_scaleview()
        self._update_lim(None)

        # connect to axis callbacks
        self.axes.callbacks.connect('xlim_changed', self._update_lim)
        self.axes.callbacks.connect('ylim_changed', self._update_lim)

    def _update_lim(self, event):
        """Update line range when the limits of the axes change."""
        (x0, y0), b = self._point, self._slope
        xlim = np.array(self.axes.get_xbound())
        ylim = np.array(self.axes.get_ybound())
        isflat = (xlim[1] - xlim[0]) * abs(b) <= (ylim[1] - ylim[0])
        if isflat:
            y = (xlim - x0) * b + y0
            self.set_data(xlim, y)
        else:
            x = (ylim - y0) / b + x0
            self.set_data(x, ylim)

    def _auto_scaleview(self):
        """Autoscale the axis view to the line.
        This will make (x0, y0) in the axes range.
        """
        self.axes.plot(*self._point).pop(0).remove()
