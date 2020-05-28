"""
Test to visualize the algorithm for computing the minimum enclosing circle, as computed using the methods in playground.py.

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
from pycoverage.vorutils import playground


def enclose_pentagon():

    # Define the convex polygon.
    vertices = 0.2 * np.array([[0, 100], [-95, 31], [-59, -81], [59, -81], [95, 31]])
    polygon = shapely.geometry.Polygon(vertices)

    # Compute the centroid.
    area = playground.compute_polygon_area(vertices)
    centroid = playground.compute_polygon_centroid(vertices)

    # Compute the enclosing circle.
    start = time.time()
    circle = playground.find_meb(vertices)
    end = time.time()
    print("Operation took {} seconds.".format(end - start))
    print("Circle is {}.".format(circle))

    # Plot the result.
    fig, ax = playground.plot_enclosing_circle(polygon, circle)

    plt.show()


def enclose_weird_hull():

    # Define the convex polygon.
    vertices = np.array(
        [
            [-42.82601857, 43.8856823],
            [-40.03539578, 39.57246959],
            [-55.06129694, 39.40204676],
            [-44.98877434, 44.92996413],
        ]
    )
    polygon = shapely.geometry.Polygon(vertices)

    # Compute the centroid.
    area = playground.compute_polygon_area(vertices)
    centroid = playground.compute_polygon_centroid(vertices)

    # Compute the enclosing circle.
    start = time.time()
    circle = playground.find_meb(vertices)
    end = time.time()
    print("Operation took {} seconds.".format(end - start))
    print("Circle is {}.".format(circle))

    # Plot the original convex polygon, and the enclosing circle.
    fig, ax = playground.plot_enclosing_circle(polygon, circle)

    plt.show()


if __name__ == "__main__":
    enclose_pentagon()
    enclose_weird_hull()
