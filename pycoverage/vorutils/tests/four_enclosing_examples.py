"""
Test to visualize the algorithm for computing the minimum enclosing circle, as computed using the methods in playground.py for four examples.

"""

import os
import sys
import time

import shapely
import numpy as np
from shapely.geometry import *
import matplotlib.pyplot as plt
from pycoverage.vorutils import playground


def four_examples(vertices):
    polygons = [shapely.geometry.Polygon(vertices[i]) for i in range(len(vertices))]

    # Compute the circles.
    start = time.time()
    circles = [playground.find_meb(vertices[i]) for i in range(len(vertices))]
    end = time.time()
    print("Operation took {} seconds.".format(end - start))

    # Plot the four examples.
    fig = plt.figure(
        constrained_layout=True,
        frameon=True,
        figsize=((1680 / 1.5) / 192, (1000 / 1.5) / 192),
        dpi=192,
    )

    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax_list = [ax0, ax1, ax2, ax3]

    counter = 0
    for ax in ax_list:
        polygon = polygons[counter]
        circle = circles[counter]

        # Remove the spines.
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        # Plot the original polygon.
        ax.plot(*(polygon.exterior.xy), color="k", linewidth=0.5)

        # Plot the center.
        ax.scatter(circle[0], circle[1], s=3)

        # Draw the enclosing circle.
        artist_circle = plt.Circle(
            (circle[0], circle[1]), circle[2], alpha=1, fill=False, linewidth=0.5
        )
        ax.add_artist(artist_circle)

        # Adjust the bounds.
        min_x, min_y, max_x, max_y = polygon.bounds

        ax.set_xlim([min_x - 1.3 * circle[2], max_x + 1.3 * circle[2]])
        ax.set_ylim([min_y - 1.3 * circle[2], max_y + 1.3 * circle[2]])
        ax.set_aspect("equal", adjustable="box")

        counter += 1
    plt.savefig("four_examples.png", dpi=192)
    plt.show()


if __name__ == "__main__":
    vertices = list(
        (
            np.array(
                [
                    [0, 0.2 * 100],
                    [0.2 * -95, 0.2 * 31],
                    [0.2 * -59, 0.2 * -81],
                    [0.2 * 59, -0.2 * 81],
                    [0.2 * 95, 0.2 * 31],
                ]
            ),
            np.array(
                [
                    [-42.8260, 43.8856],
                    [-40.0353, 39.5724],
                    [-55.0612, 39.4020],
                    [-44.9887, 44.9299],
                ]
            ),
            np.array(
                [
                    [50.0, -19.1294],
                    [25.5361, -19.5973],
                    [3.7068, 7.7712],
                    [19.3939, 50.0],
                    [50.0, 50.0],
                ]
            ),
            np.array(
                [
                    [-50.0, -32.7454],
                    [8.7531, -9.3882],
                    [50.0, -39.0234],
                    [50.0, -50.0],
                    [-50.0, -50.0],
                ]
            ),
        )
    )
    four_examples(vertices)
