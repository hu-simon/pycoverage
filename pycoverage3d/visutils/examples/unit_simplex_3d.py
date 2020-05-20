"""
Python code that plots the unit simplex in 3-D using pyvorovis.
"""

import os
import time
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from pycoverage3d.visutils import Geom
from pycoverage3d.visutils import pyvorovis


def main():
    R = 1
    theta = np.pi / 2  # orientation of the triangle

    vertices = np.array(
        [
            [R * np.cos(theta), R * np.sin(theta), 0],
            [
                R * np.cos(theta + (2 * np.pi / 3)),
                R * np.sin(theta + (2 * np.pi / 3)),
                0,
            ],
            [
                R * np.cos(theta + (4 * np.pi / 3)),
                R * np.sin(theta + (4 * np.pi / 3)),
                0,
            ],
            [0, 0, 1],
        ]
    )

    chull = scipy.spatial.ConvexHull(vertices)
    triangles = [vertices[s] for s in chull.simplices]
    faces = Geom.PlanarFace(triangles, sig_dig=3).simplify_faces()

    fig = pyvorovis.chull_plot_3d(chull, faces, edgecolor="k", linewidths=1, alpha=0.01)

    plt.show()


if __name__ == "__main__":
    main()
