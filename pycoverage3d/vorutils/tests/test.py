""" Python file containing tests for the pyvoro3d package.

"""

import time
import unittest

import scipy.misc
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycoverage3d.vorutils import pyvoro3d


class TestCheckMembership(unittest.TestCase):
    unit_cube = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
    ]

    def test_point_outside(self):
        """ Tests if the point (0.0, 0.0, 1.01) is contained in the unit cube. Expecting False. 
        """
        point = [0.5, 0.5, 1.01]
        result = pyvoro3d.check_membership(self.unit_cube, point)
        self.assertFalse(result)

    def test_point_inside(self):
        """ Tests if the point (0.5, 0.5, 1.0) is contained in the unit cube. Expecting True.
        """
        point = [0.5, 0.5, 1.0]
        result = pyvoro3d.check_membership(self.unit_cube, point)
        self.assertTrue(result)

    def test_point_boundary(self):
        """ Tests if the point (1.0, 1.0, 1.0) is contained in the unit cube. Expecting True.
        """
        point = [1.0, 1.0, 1.0]
        result = pyvoro3d.check_membership(self.unit_cube, point)
        self.assertTrue(result)

    def test_point_origin(self):
        """ Tests if the point (0.0, 0.0, 0.0) is contained in the unit cube. Expecting True.
        """
        point = [0.0, 0.0, 0.0]
        result = pyvoro3d.check_membership(self.unit_cube, point)
        self.assertTrue(result)


class TestGenerateRandomPoints(unittest.TestCase):
    unit_cube = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
    ]
    chull = scipy.spatial.ConvexHull(unit_cube)

    '''
    def confirm_result(self, random_points, min_dist):
        """ Confirms whether the points in ``random_points`` are separated by ``min_dist``. 
        """
        confirm_sum = 0
        for idx, point in enumerate(random_points):
            diff = np.linalg.norm(
                np.array(point) - np.delete(random_points, idx, axis=0), axis=1
            )
            print(diff)
            print(diff >= min_dist)
            sum_diff = np.sum(diff >= min_dist)
            if sum_diff == len(random_points) - 1:
                confirm_sum += 1
            else:
                continue

        return confirm_sum == len(random_points)
    '''

    def confirm_result(self, random_points, min_dist):
        """ Confirms whether the points in ``random_points`` are separated by ``min_dist``.
        """
        confirm_sum = 0
        for i in range(len(random_points)):
            for j in range(len(random_points)):
                if j > i:
                    diff = np.linalg.norm(
                        np.array(random_points[i] - np.array(random_points[j]))
                    )
                    if diff >= min_dist:
                        confirm_sum += 1
                    elif np.isclose(diff, min_dist, rtol=1e-3):
                        print(
                            "Very close, difference is {}".format(
                                np.abs(diff - min_dist)
                            )
                        )
                        confirm_sum += 1
                else:
                    continue
        return confirm_sum == np.sum(np.arange(1, len(random_points)))

    '''
    def test_one(self):
        """ Generates five random points contained in the unit cube, with a minimum separation distance of 0.
        """
        num_points = 5
        min_dist = 0.0
        random_points = pyvoro3d.generate_random_points(
            self.chull, num_points=num_points, min_dist=min_dist
        )
        self.assertTrue(self.confirm_result(random_points, min_dist))
    '''

    def test_two(self):
        """ Generates ten random points contained in the unit cube, with a minimum separation distance of 0.2.
        """
        print("\n------------------------------------------------------------")
        print("Testing non-zero distance...")
        print("------------------------------------------------------------")
        for iter in range(1000):
            print("Iteration {}".format(iter))
            num_points = 10
            min_dist = 0.2
            random_points = pyvoro3d.generate_random_points(
                self.chull, num_points=num_points, min_dist=min_dist, random_seed=iter,
            )
            self.assertTrue(self.confirm_result(random_points, min_dist))

    """
    def test_fail_cases_manual(self):
        num_points = 10
        min_dist = 0.2
        random_points = pyvoro3d.generate_random_points(
            self.chull, num_points=num_points, min_dist=min_dist, random_seed=86,
        )
        print(random_points)
        self.assertTrue(self.confirm_result(random_points, min_dist))
    """


if __name__ == "__main__":
    unittest.main()
