"""
Python code that plots the unit cube in 3-D using pyvorovis.
"""

import os
import time
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from pycoverage3d.visutils import Geom
from pycoverage3d.visutils import pyvorovis


def compute_vertices():
    vertices = list()
    phi = scipy.constants.golden_ratio

    for i in list([-1, 1]):
        for j in list([-1, 1]):
            vertices.append([0, i * 1, j * phi])
            vertices.append([i * 1, j * phi, 0])
            vertices.append([i * phi, 0, j * 1])

    return np.array(vertices)


def plot_icosahedron(vertices):
    chull = scipy.spatial.ConvexHull(vertices)
    triangles = [vertices[s] for s in chull.simplices]
    faces = Geom.PlanarFace(triangles, sig_dig=3).simplify_faces()

    fig = pyvorovis.chull_plot_3d(chull, faces, edgecolor="k", linewidths=1, alpha=0.01)

    plt.axis("off")
    plt.grid(None)

    plt.show()


if __name__ == "__main__":
    vertices = compute_vertices()
    plot_icosahedron(vertices)
