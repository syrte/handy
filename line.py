from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import wraps

__all__ = ["abline", "ABLine2D", "axline"]


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

        It will have wrong behavior when the axis-limits are
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

        assert "transform" not in kwargs
        if 'axes' in kwargs:
            ax = kwargs['axes']
        else:
            ax = plt.gca()
        if not ('color' in kwargs or 'c' in kwargs):
            kwargs.update(next(ax._get_lines.prop_cycler))

        super(ABLine2D, self).__init__([], [], *args, **kwargs)
        self._point = tuple(point)
        self._slope = float(slope)

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


@wraps(ABLine2D.__init__, assigned=['__doc__'], updated=[])
def abline(a, b, *args, **kwargs):
    return ABLine2D(a, b, *args, **kwargs)


def axline(a, b, **kwargs):
    """
    Add an infinite straight line across an axis.

    Parameters
    ----------
    a, b: scalar or tuple
        Acceptable forms are
        y0, b:
            y = y0 + b * x
        (x0, y0), b:
            y = y0 + b * (x - x0)
        (x0, y0), (x1, y1):
            y = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    Additional arguments are passed to the <matplotlib.lines.Line2D> constructor.

    Returns
    -------
    :class:`~matplotlib.lines.Line2D`

    Other Parameters
    ----------------
    Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
    with the exception of 'transform':
    %(Line2D)s

    Examples
    --------
    * Draw a thick red line with slope 1 and y-intercept 0::
        >>> axline(0, 1, linewidth=4, color='r')
    * Draw a default line with slope 1 and y-intercept 1::
        >>> axline(1, 1)

    See Also
    --------
    axhline : for horizontal lines
    axvline : for vertical lines

    Notes
    -----
    Currently this method does not work properly with log scaled axes.
    Taken from https://github.com/matplotlib/matplotlib/pull/9321
    """
    from matplotlib import pyplot as plt
    import matplotlib.transforms as mtransforms
    import matplotlib.lines as mlines

    if np.isscalar(a):
        if not np.isscalar(b):
            raise ValueError("Invalid line parameters.")
        point, slope = (0, a), b
    elif np.isscalar(b):
        if not len(a) == 2:
            raise ValueError("Invalid line parameters.")
        point, slope = a, b
    else:
        if not len(a) == len(b) == 2:
            raise ValueError("Invalid line parameters.")
        if b[0] != a[0]:
            point, slope = a, (b[1] - a[1]) / (b[0] - a[0])
        else:
            point, slope = a, np.inf

    ax = plt.gca()
    if "transform" in kwargs:
        raise ValueError("'transform' is not allowed as a kwarg; "
                         "axline generates its own transform.")

    if slope == 0:
        return ax.axhline(point[1], **kwargs)
    elif np.isinf(slope):
        return ax.axvline(point[0], **kwargs)

    xtrans = mtransforms.BboxTransformTo(ax.viewLim)
    viewLimT = mtransforms.TransformedBbox(
        ax.viewLim,
        mtransforms.Affine2D().rotate_deg(90).scale(-1, 1))
    ytrans = (mtransforms.BboxTransformTo(viewLimT) +
              mtransforms.Affine2D().scale(slope).translate(*point))
    trans = mtransforms.blended_transform_factory(xtrans, ytrans)
    line = mlines.Line2D([0, 1], [0, 1],
                         transform=trans + ax.transData,
                         **kwargs)
    ax.add_line(line)
    return line
