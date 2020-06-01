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


if __name__ == "__main__":
    test_project_2d()
