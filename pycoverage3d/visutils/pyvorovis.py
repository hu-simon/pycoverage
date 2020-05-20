""" Python module containing visualization tools to complement the Voronoi operations in 3D.

Some other documentation here, describing this module in more detail.

"""

__author__ = "Jun Hao (Simon) Hu"
__copyright__ = "Copyright (c) 2020, Simon Hu"
__version__ = "0.2"
__maintainer__ = "Simon Hu"
__email__ = "simonhu@ieee.org"
__status__ = "Development"

import os
import sys
import warnings

warnings.simplefilter("module", PendingDeprecationWarning)

import shapely
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplt
from pycoverage3d.vorutils import pyvoro3d
from scipy._lib.decorator import decorator as _decorator


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


"""
@_decorator
def held_figure(func, obj, ax=None, **kwargs):

    # If no current axes, then create a new figure with axes.
    if ax is None:
        fig = plt.figure()
        ax = mplt.Axes3D(fig)

    is_held = ax.ishold()
    try:
        ax.hold(True)
        return func(obj, ax=ax, **kwargs)
    finally:
        ax.hold(is_held)
"""


def adjust_bounds(ax, points):
    ptp_bound = points.ptp(axis=0)
    ax.set_xlim(
        points[:, 0].min() - 0.1 * ptp_bound[0], points[:, 0].max() + 0.1 * ptp_bound[0]
    )
    ax.set_ylim(
        points[:, 1].min() - 0.1 * ptp_bound[1], points[:, 1].max() + 0.1 * ptp_bound[1]
    )
    ax.set_zlim(
        points[:, 2].min() - 0.1 * ptp_bound[2], points[:, 2].max() + 0.1 * ptp_bound[2]
    )


# @held_figure
def chull_plot_3d(chull, faces, ax=None, **kwargs):
    """
    Plot the given faces that define a convex hull in 3-D. 
    
    Parameters
    ----------
    chull : scipy.spatial.ConvexHull instance
        Original convex hull that we want to plot in 3-D.
    faces : pycoverage3d.visutils.Geom.PlanarFace instance
        Faces that define a convex hull in 3-D.
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on, by default None.

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure corresponding to the plot.
    
    See Also
    --------
    pycoverge3d.visutils.Geom.PlanarFace : PlanarFace class used to compute the faces.

    Notes
    -----
    Requires matplotlib and mpl_toolkits.
   
    """
    ax = mplt.Axes3D(plt.figure())

    pc = mplt.art3d.Poly3DCollection(faces, **kwargs)
    ax.add_collection3d(pc)

    adjust_bounds(ax, chull.points)

    return ax.figure
