"""
Python playground file used for testing algorithm ideas.

"""

import os
import sys
import time

import shapely
import numpy as np
from scipy.spatial import *
from shapely.geometry import *
import matplotlib.pyplot as plt
from pycoverage.vorutils import pyvoro


def generate_random_points(poly, num_points, min_dist=0.0, sig_dig=4):
    """
    Generates random tuples of the form (x, y) which lie in the interior of a convex hull and are separated by ``min_dist`` distance, using rejection sampling. It is assumed that the hull vertices are ordered in counter-clockwise order.
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon instance
        Convex hull used for determining membership during rejection sampling.
    num_points : int
        Number of points to be generated.
    min_dist : float, optional
        The minimum separation distance enforced for each point, by default 0.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    random_points : list
        Unordered list of tuples of the form (x, y) representing the randomly sampled points which lie within the convex hull, and are separated by a distance of ``min_dist``. 

    Notes
    -----
    This function does not compute the packing number, which is needed to determine if it is possible to generate ``num_points`` points that are separated by a distance of ``min_dist`` distance. It is the responsibility of the user to ensure that the problem is feasible.

    It is assumed that the hull vertices are ordered in counter-clockwise order.
    
    Examples
    --------
    >>> # Generates 3 points with ``min_dist`` of 1.
    >>> hull = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> random_points = geoutils.generate_random_points(hull, num_points=3, min_dist=1.)

    >>> # Fail case where the problem is not feasible. The result is an infinite runtime!
    >>> hull = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> random_points = geoutils.generate_random_points(hull, num_points=100, min_dist=100.)
    
    TODO need to fix the examples so that they work with the sig_dig option.

    """
    min_x, min_y, max_x, max_y = poly.bounds

    x_random = list()
    y_random = list()

    x_sample = np.around(np.random.uniform(min_x, max_x), sig_dig)
    y_sample = np.around(np.random.uniform(min_y, max_y), sig_dig)

    first_point = False

    while first_point is False:
        if poly.contains(Point(x_sample, y_sample)):
            x_random.append(x_sample)
            y_random.append(y_sample)
            first_point = True
        else:
            x_sample = np.around(np.random.uniform(min_x, max_x), sig_dig)
            y_sample = np.around(np.random.uniform(min_y, max_y), sig_dig)

    while len(x_random) < num_points:
        x_sample = np.around(np.random.uniform(min_x, max_x), sig_dig)
        y_sample = np.around(np.random.uniform(min_y, max_y), sig_dig)

        if poly.contains(Point(x_sample, y_sample)):
            distances = np.sqrt(
                (x_sample - np.array(x_random)) ** 2
                + (y_sample - np.array(y_random)) ** 2
            )
            min_distance = np.min(distances)
            if min_distance >= min_dist:
                x_random.append(x_sample)
                y_random.append(y_sample)
        else:
            continue

    random_points = [[x_random[i], y_random[i]] for i in range(len(x_random))]

    return np.array(random_points)


def compute_polygon_area(points, sig_dig=4):
    """
    Computes the area of a convex polygon defined by points using a shoelace formula.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coordinates in 2-D of the points defining the convex polygon. 
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    area : float
        Area of the convex polygon, defined by ``points``, truncated to ``sig_dig`` significant digits.

    Examples
    --------
    >>> # TODO

    """
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area = np.around(area, sig_dig)

    return area


def compute_polygon_centroid(points, sig_dig=4):
    """
    Computes the geometric centroid of the convex polygon, defined by ``points``, using a shoelace formula.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coordinates in 2-D of the points defining the convex polygon.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    centroid : numpy.ndarray instance
        Geometric centroid of the convex polygon, defined by ``points``, truncated to ``sig_dig`` significant digits.

    Examples
    --------
    >>> # TODO
    
    """
    x = points[:, 0]
    y = points[:, 1]
    area = compute_polygon_area(points, sig_dig)

    centroid_x = np.dot((x + np.roll(x, 1)), (x * np.roll(y, 1) - np.roll(x, 1) * y))
    centroid_y = np.dot((y + np.roll(y, 1)), (x * np.roll(y, 1) - np.roll(x, 1) * y))
    centroid = (1 / (6 * area)) * np.array([centroid_x, centroid_y])
    centroid = np.around(centroid, sig_dig)

    return centroid


def compute_enclosing_circle(points, sig_dig=4):
    """
    Computes the smallest enclosing circle that contains the convex polygon.
    
    This is computed by finding the geometric center of the polygon and then finding the largest distance between the centroid and ``points``.

    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coordinates in 2-D of the points defining the convex polygon.
    sig_dig : int, optional
        The number of siginificant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    circle : numpy.ndarray
        Array containing the coordinates (x, y, radius) that define the minimum enclosing circle.

    Examples
    --------
    >>> # TODO

    """
    centroid = compute_polygon_centroid(points, sig_dig)
    distances = np.linalg.norm(centroid - points, axis=1)
    radius = np.max(distances)
    circle = np.array([centroid[0], centroid[1], radius])

    return circle


def compute_circles(points, sig_dig=4):
    """
    Computes the distances from the centroid to the vertices of the convex polygon and returns circles whose radius are the distances.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coordinates in 2-D of the points defining the convex polygon.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    circles : numpy.ndarray
        Array containing a list of circles defined by the coordinates (x, y, radius) which determine the circles.
    
    Examples
    --------
    >>> # TODO

    """
    centroid = compute_polygon_centroid(points, sig_dig)
    distances = np.sort(np.linalg.norm(centroid - points, axis=1))
    circles = np.array(
        [[centroid[0], centroid[1], distances[i]] for i in range(len(distances))]
    )

    return circles


def plot_enclosing_circle(poly, circle):
    """
    Plots the convex polygon and the minimum enclosing circle. 
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon instance
        The convex polygon representing the original operating environment.
    circle : numpy.ndarray
        Array containing the coordinates (x, y, radius) that define the minimum enclosing circle.

    Returns
    -------
    None

    Examples
    --------
    >>> # TODO

    """
    fig, ax = plt.subplots()

    # Remove the spines.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove the x and y ticks.
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot the original polygon.
    ax.plot(*(poly.exterior.xy))

    # Plot the centroid.
    ax.scatter(circle[0], circle[1], s=3)

    # Draw the enclosing circle.
    artist_circle = plt.Circle((circle[0], circle[1]), circle[2], alpha=1, fill=False)
    ax.add_artist(artist_circle)

    min_x, min_y, max_x, max_y = poly.bounds

    ax.set_xlim([min_x - 1.1 * circle[2], max_x + 1.1 * circle[2]])
    ax.set_ylim([min_y - 1.1 * circle[2], max_y + 1.1 * circle[2]])
    ax.set_aspect("equal", adjustable="box")

    return fig, ax


def plot_all_circles(poly, circles):
    """
    Plots the convex polygon and the circles obtained from ``compute_circles()``.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        The convex polygon representing the original operating environment.
    circles : numpy.ndarray
        Array containing a list of circles defined by the coordinates (x, y, radius) which determine the circles.

    Returns
    -------
    None

    Examples
    --------
    >>> # TODO

    """
    fig, ax = plt.subplots()

    # Remove the spines.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot the original polygon.
    ax.plot(*(poly.exterior.xy))

    # Plot the centroid.
    ax.scatter(circles[0][0], circles[0][1], s=3)

    # Draw the circles.
    for circle in circles:
        artist_circle = plt.Circle(
            (circle[0], circle[1]), circle[2], alpha=1, fill=False
        )
        ax.add_artist(artist_circle)

    max_radius = np.max(circles[:, 2])
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.set_xlim([xmin - max_radius, xmax + max_radius])
    ax.set_ylim([ymin - max_radius, ymax + max_radius])
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def find_meb(points, sig_dig=4, epsilon=1e-2, stop_tol=1e-2):
    """
    Computes an approximation to the minimum enclosing ball that encloses the convex polygon defined by ``points``. 

    The minimum enclosing ball is computed using the Frank-Wolfe algorithm (insert reference here) and iteratively converges to the minimum enclosing ball.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coorddinates in 2-D of the points defining the convex polygon.
    sig_dig : int, optional
        The number of siginificant digits to truncate the computational ressults, by default 4 significant digits.
    epsilon : float, optional
        Parameter determining the acceptable error of the approximation and number of iterations, by default 1e-3.
    stop_tol : float, optional
        Stopping tolerance used for ending iterations early. 

    Returns
    -------
    circle : numpy.ndarray instance
        Array containing the coordinates (x, y, radius) that define the minimum enclosing ball.

    Examples
    --------
    >>> # TODO

    Notes
    -----
    Note that the smaller ``epsilon`` is, the more accurate the result. However, this also increases the number of iterations by a factor of two-fold, so choose this parameter wisely.

    The parameter ``stop_tol`` is used to determine if the algorithm should be terminated early. If the absolute difference between the newly computed center and the previous center is smaller than ``stop_tol``, then the algorithm is terminated early.

    """
    # Use the centroid as the initial point.
    center = compute_polygon_centroid(points, sig_dig)
    radius = np.max(np.linalg.norm(center - points, axis=1))

    # Determine the number of iterations.
    num_iters = (int)(np.ceil(1 / epsilon ** 2))

    # Iteratively compute the center of the minimum enclosing ball.
    for k in range(num_iters):
        print("Iteration {}".format(k))
        prev_center = center.copy()

        distances = np.linalg.norm(center - points, axis=1)
        idx_max = np.argmax(distances)
        center = (k / (k + 1)) * center + (1 / (k + 1)) * points[idx_max]
        radius = distances[idx_max]

        # if np.linalg.norm(center - prev_center) ** 2 < stop_tol:
        #    print("Stopping condition satisfied, terminating algorithm...")
        #    break

    circle = np.array([center[0], center[1], radius])

    return circle
