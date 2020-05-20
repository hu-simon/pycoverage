"""
Python code that plots a regular pentagonal dodecahedron in 3-D using pyvorovis.

Not working as expected...
"""

import os
import time
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplt
from pycoverage3d.visutils import Geom
from pycoverage3d.visutils import pyvorovis


def compute_vertices(R=1):
    vertices = list()
    phi = scipy.constants.golden_ratio
    scale_factor = 1 / np.sqrt(3)
    mphi = phi / np.sqrt(3)  # c
    dphi = np.sqrt(3) / phi  # b

    for i in list([-1, 1]):
        for j in list([-1, 1]):
            vertices.append([0, i * mphi * R, j * dphi * R])
            vertices.append([i * mphi * R, j * dphi * R, 0])
            vertices.append([i * dphi * R, 0, j * mphi * R])

            for k in list([-1, 1]):
                vertices.append(
                    [i * scale_factor * R, j * scale_factor * R, k * scale_factor * R]
                )

    vertices = np.array(vertices)
    return vertices


def plot_dodecahedron(vertices):
    chull = scipy.spatial.ConvexHull(vertices)
    triangles = [vertices[s] for s in chull.simplices]
    faces = Geom.PlanarFace(triangles, sig_dig=3).simplify_faces()

    fig = pyvorovis.chull_plot_3d(chull, faces, edgecolor="k", linewidths=1, alpha=0.01)

    plt.axis("off")
    plt.grid(None)

    # Try something new
    ax = mplt.Axes3D(plt.figure())
    for s in chull.simplices:
        vtx = [vertices[s[0], :], vertices[s[1], :], vertices[s[2], :]]
        tri = mplt.art3d.Poly3DCollection([vtx], linewidths=1, alpha=0.8)
        tri.set_color("salmon")
        tri.set_edgecolor("k")
        ax.add_collection3d(tri)

    plt.axis("off")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    plt.show()


if __name__ == "__main__":
    vertices = compute_vertices()
    plot_dodecahedron(vertices)
