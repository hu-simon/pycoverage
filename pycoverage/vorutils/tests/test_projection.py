"""
Test to visualize the projection algorithm, as implemented in playground.py.

"""

import os
import sys
import time

import shapely
import numpy as np
from shapely.geometry import *
import matplotlib.pyplot as plt
from pycoverage.vorutils import playground


def generate_outside_points(poly, num_points, min_dist=0.0, sig_dig=4):
    pass


def test_projection():
    vertices = np.array(
        [
            [0, 0.2 * 100],
            [0.2 * -95, 0.2 * 31],
            [0.2 * -59, 0.2 * -81],
            [0.2 * 59, -0.2 * 81],
            [0.2 * 95, 0.2 * 31],
        ]
    )
    polygon = shapely.geometry.Polygon(vertices)

    # Points outside of convex polygon.
    points = 0.5 * np.array([[-40, 30], [0, 60], [-30, 0], [30, -50]])
    proj_points = list()

    start = time.time()
    for idx, point in enumerate(points):
        proj_points.append(playground.project_to_hull(vertices, point))
    end = time.time()
    print(
        "Found the projection of {} points on to the convex polygon in {} seconds.".format(
            points.shape[0], end - start
        )
    )

    # Plot the results.
    fig, ax = playground.plot_projections(polygon, points, proj_points)

    plt.savefig("example_projection.png", dpi=192)
    plt.show()


def test_speed_benchmark():
    vertices = np.array(
        [
            [0, 0.2 * 100],
            [0.2 * -95, 0.2 * 31],
            [0.2 * -59, 0.2 * -81],
            [0.2 * 59, -0.2 * 81],
            [0.2 * 95, 0.2 * 31],
        ]
    )
    """
    vertices = np.array(
        [
            [-42.8260, 43.8856],
            [-40.0353, 39.5724],
            [-55.0612, 39.4020],
            [-44.9887, 44.9299],
        ]
    )
    """
    polygon = shapely.geometry.Polygon(vertices)
    centroid = playground.compute_polygon_centroid(vertices)

    # Create points outside of the convex polygon.
    min_x, min_y, max_x, max_y = polygon.bounds
    min_bound = 5 * np.maximum(min_x, min_y)
    max_bound = 5 * np.maximum(max_x, max_y)
    points = np.random.uniform(
        centroid[0] - min_bound, centroid[1] - max_bound, size=(10, 2)
    )

    # Compute the projections of those points.
    proj_points = list()
    start = time.time()
    for idx, point in enumerate(points):
        proj_points.append(playground.project_to_hull(vertices, point))
    end = time.time()
    print(
        "Found the projection of {} points on to the convex polygon in {} seconds.".format(
            points.shape[0], end - start
        )
    )

    fig, ax = playground.plot_projections(polygon, points, proj_points)

    plt.savefig("many_projections.png", dpi=192)
    plt.show()


if __name__ == "__main__":
    # test_projection()
    test_speed_benchmark()
