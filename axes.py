from __future__ import division
from matplotlib import pyplot as plt

__all__ = ['twin_axes']


def twin_axes(show="xy", ax=None):
    """
    Create a twin of Axes for generating a plot with a shared
    x-axis and y-axis.
    The x-axis (y-axis) of ax will have ticks on bottom (left)
    and the returned axes will have ticks on the top (right).

    It will have wrong behaiver when the axis-limits are
    changed by setting ticks. This can be corrected by call
    `xlim` to reset the limits.
    Refer this issue:
        https://github.com/matplotlib/matplotlib/issues/6863
    """
    assert show in ['x', 'y', 'xy']
    if ax is None:
        ax = plt.gca()

    ax2 = ax._make_twin_axes()
    ax2._shared_x_axes.join(ax2, ax)
    ax2._shared_y_axes.join(ax2, ax)
    ax2._adjustable = 'datalim'
    ax2.set_xlim(ax.get_xlim(), emit=False, auto=False)
    ax2.set_ylim(ax.get_ylim(), emit=False, auto=False)
    ax2.xaxis._set_scale(ax.xaxis.get_scale())
    ax2.yaxis._set_scale(ax.yaxis.get_scale())

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_offset_position('right')

    ax2.patch.set_visible(False)
    if show == 'x':
        ax2.yaxis.set_visible(False)
    elif show == 'y':
        ax2.xaxis.set_visible(False)
    return ax2
