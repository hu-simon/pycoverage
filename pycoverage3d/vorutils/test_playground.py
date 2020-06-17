"""
Tests to visualize the algorithms in playground.py.

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
from pycoverage3d.vorutils import pyvoro3d
from pycoverage3d.vorutils import playground


def test_project_2d(blue_pos=None, red_pos=None, sig_dig=4):
    if blue_pos is None:
        blue_pos = np.around(np.random.randn(5, 3), sig_dig)
    if red_pos is None:
        red_pos = np.around(np.random.randn(1, 3), sig_dig)
    proj_blue, proj_red, proj_height = playground.project_2d(blue_pos, red_pos, sig_dig)

    print("\n")
    print("BLUE positions")
    print("------------------------------------\n")
    print(blue_pos)
    print("\n")

    print("RED positions")
    print("------------------------------------\n")
    print(red_pos)
    print("\n")

    print("Projected BLUE positions")
    print("------------------------------------\n")
    print(proj_blue)
    print("\n")

    print("Projected RED positions")
    print("------------------------------------\n")
    print(proj_red)
    print("\n")

    print("Projected Heights")
    print("------------------------------------\n")
    print(proj_height)
    print("\n")


def visualize_projection_2d(blue_pos=None, red_pos=None, sig_dig=4):
    if blue_pos is None:
        blue_pos = np.around(np.random.uniform(-7, 7, size=(5, 3)), sig_dig)
    if red_pos is None:
        red_pos = np.around(np.random.uniform(-7, 7, size=(1, 3)), sig_dig)
    proj_blue, proj_red, proj_height = playground.project_2d(blue_pos, red_pos, sig_dig)
    vec = np.ones(proj_blue.shape[0]) * red_pos[:, 2][0]
    proj_blue_plane = np.insert(proj_blue, 2, vec, axis=1)
    print(proj_blue_plane)

    # Plot the original 3-D coordinates.
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(blue_pos[:, 0], blue_pos[:, 1], blue_pos[:, 2])
    ax.scatter3D(red_pos[:, 0], red_pos[:, 1], red_pos[:, 2], color="r")

    # Plot the plane that is being projected onto.
    normal = np.array([0, 0, red_pos[:, 2][0]])
    d = -red_pos.dot(normal)
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2)

    # Plot the projected points.
    ax.scatter3D(
        proj_blue_plane[:, 0], proj_blue_plane[:, 1], proj_blue_plane[:, 2], color="g"
    )

    plt.show()


if __name__ == "__main__":
    test_project_2d()
    visualize_projection_2d()
