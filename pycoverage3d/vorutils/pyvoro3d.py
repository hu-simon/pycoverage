""" Python file containing utilities for basic Voronoi operations in 3D.

Some other documentation here, explaining more about this module.

"""

"""
TODO: 
1. Verify that check_membership is working as expected. DONE.
2. Verify that generate_random_points is working as expected.
3. Figure out how to implement an algorithm that computes the Voronoi partition.
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
import scipy.optimize


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def check_membership(hull_points, point):
    """ Determines if ``point`` is contained in the convex polygon determined by ``hull_points``.

    Membership is checked by determining whether the point can be written as a convex combination of the points that define the convex hull. This is equivalent to solving the linear problem...finish the problem later.
    
    Parameters
    ----------
    hull_points : array-like
        Array-like containing n-dimensional coordinates of the points that define the convex hull. 
    point : array-like
        Array-like containing n-dimensional coordinates of the point for which membership is checked.

    Returns
    -------
    success : bool
        Boolean determining whether ``point`` is contained in the convex polygon determined by ``hull_points``. 

    Notes
    -----
    The input to this function is not restricted to points in 3D. In fact, this algorithm generalizes to n-dimensions.

    Note that by the definition of a convex hull, it is not neccessary that hull_points define a convex hull. The algorithm also works on any set of points, since a subset of those points will be guaranteed to define a convex hull.

    Examples
    --------
    >>> # Test if point (0.0, 0.0, 1.01) is in the unit cube. Expecting False.
    >>> unit_cube = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
    ]
    >>> point = [0.0, 0.0, 1.01]
    >>> pyvoro.check_membership(unit_cube, point)
    False
    
    >>> # Test if point (0.0, 0.0, 1.0) is in the unit cube. Expecting True.
    >>> point = [0.0, 0.0, 1.0]
    >>> pyvoro.check_membership(unit_cube, point)
    True

    See Also
    --------
    
    """
    # Convert point-to-check to numpy.ndarray for consistency.
    try:
        point = np.array(point)
    except TypeError:
        print(
            "Input must be an array-like object like a list or numpy ndarray. See documentation for example usuage"
        )

    try:
        hull_points = np.array(hull_points)
    except TypeError:
        print(
            "Input must be an array-like object like a list or numpy ndarray. See documentation for example usage."
        )

    # Set up the linear program.
    num_points = len(hull_points)
    num_dim = len(point)
    c = np.zeros(num_points)
    A = np.vstack((hull_points.T, np.ones((1, num_points))))
    b = np.hstack((point, np.ones(1)))

    # Solve the linear program to determine membership.
    lp = scipy.optimize.linprog(c, A_eq=A, b_eq=b)
    success = lp.success

    return success


def generate_random_points(chull, num_points, min_dist=0.0):
    """ Uniformly generates random tuples of the form (x, y, z) which lie in the interior of a convex polygon, using rejection sampling. 
    
    Parameters
    ----------
    chull : SciPy ConvexHull
        Polygon used for determining membership during rejection sampling.
    num_points : int
        Number of points to be generated.
    min_dist : float, optional
        Minimum separation distance enforced for each point, by default 0.0 representing no separation distance.

    Returns
    -------
    random_points : array-like
        Unordered array-like containing tuples of the form (x, y, z) representing the coordinates of the randomly sampled points. These coordinates are guaranteed to be contained in the polygon and separately by a distance of ``min_dist``. 

    Notes
    -----
    This function does not compute the packing number, which is needed to determine if it is feasbile to generate ``num_points`` points that are separated within a distance of ``min_dist``. It is the responsibility of the user to ensure that the problem is feasible.

    Unlike the 2D variant, it is not assumed that the polygon vertices are ordered in counter-clockwise order.

    Examples
    -------- 
    >>> # TODO

    See Also
    --------
    pyvoro3d.check_membership : Used for determining membership. 
    """
    vertices = chull.points
    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    min_z, max_z = vertices[:, 2].min(), vertices[:, 2].max()

    x_random = list()
    y_random = list()
    z_random = list()

    x_sample = np.random.uniform(min_x, max_x)
    y_sample = np.random.uniform(min_y, max_y)
    z_sample = np.random.uniform(min_z, max_z)
    point = np.array([x_sample, y_sample, z_sample])

    first_point_flag = False
    while first_point_flag is False:
        if check_membership(vertices, point):
            x_random.append(np.around(x_sample, 4))
            y_random.append(np.around(y_sample, 4))
            z_random.append(np.around(z_sample, 4))
            first_point_flag = True
        else:
            x_sample = np.random.uniform(min_x, max_x)
            y_sample = np.random.uniform(min_y, max_y)
            z_sample = np.random.uniform(min_z, maz_z)

    while len(x_random) < num_points:
        x_sample = np.random.uniform(min_x, max_x)
        y_sample = np.random.uniform(min_y, max_y)
        z_sample = np.random.uniform(min_z, max_z)
        point = np.array([x_sample, y_sample, z_sample])

        if check_membership(vertices, point):
            distances = np.linalg.norm(
                point.reshape(3, 1) - np.array((x_random, y_random, z_random))
            )
            min_distance = np.min(distances)
            if min_distance >= min_dist:
                x_random.append(np.around(x_sample, 4))
                y_random.append(np.around(y_sample, 4))
                z_random.append(np.around(z_sample, 4))
        else:
            continue

    random_points = [[x_random[i], y_random[i], z_random[i]] for i in range(num_points)]
    return np.array(random_points)
