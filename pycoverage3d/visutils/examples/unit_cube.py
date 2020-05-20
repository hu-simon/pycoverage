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


def main():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    chull = scipy.spatial.ConvexHull(vertices)
    triangles = [vertices[s] for s in chull.simplices]
    faces = Geom.PlanarFace(triangles, sig_dig=3).simplify_faces()

    fig = pyvorovis.chull_plot_3d(chull, faces, edgecolor="k", linewidths=1, alpha=0.01)

    plt.show()


if __name__ == "__main__":
    main()
