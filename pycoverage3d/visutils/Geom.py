""" Python file containing class definitions for Geometric objects that are used for visualization purposes.

Some more information here, describing the classes in more detail.

"""

import os
import sys

import numpy as np
import scipy.spatial


class PlanarFace:
    """ A planar face of a convex polygon in 3D.

    Attributes
    ----------
    TODO

    Methods
    -------
    __init__(tri, sig_dig=5, method="convex_hull")
        Instantiates a planar face object of a convex polygon in 3D.
    compute_normal(sq) 
        Computes the normal vector for each triangle that defines the simplex.
    is_neighbor(tri1, tri2)
        Determines if two planar faces are neighbors (i.e. they share a nonempty intersection).
    order_vertices(vertices)
        Orders a set of vertices to achieve the proper orientation. 
    simplify_faces()
        Simplifies (or combines) any two planar faces that are neighbors and coplanar.
    """

    def __init__(self, tri_list, sig_dig=5, method="convex_hull"):
        """ Instantiates a PlanarFace object.
        
        Parameters
        ----------
        tri_list : list
            List containing the vertices of the simplex of the convex hull.
        sig_dig : int, optional
            The number of significant digits for computational accuracy, by default 5
        method : str, optional
            The object used to create the triangles, by default "convex_hull".

        Returns
        -------
        None

        Notes
        -----
        As of version 0.2, the only ``method`` supported is "convex_hull".
        """

        self.method = method
        self.tri_list = np.around(np.array(tri_list), sig_dig)
        self.group_idx = list(range(len(tri_list)))
        self.normals = np.around(
            [self.compute_normal(s) for s in self.tri_list], sig_dig
        )
        _, self.inv = np.unique(self.normals, return_inverse=True, axis=0)

    def compute_normal(self, sq):
        normal_vec = np.cross(sq[2] - sq[0], sq[1] - sq[0])
        return np.abs(normal_vec / np.linalg.norm(normal_vec))

    def is_neighbor(self, tri1, tri2):
        concat_vec = np.concatenate((tri1, tri2), axis=0)
        return len(concat_vec) == len(np.unique(concat_vec, axis=0)) + 2

    def order_vertices(self, vertices):
        if len(vertices) <= 3:
            # print("Not enough vertices to work with, and thus does not need ordering.")
            return vertices
        vertices = np.unique(vertices, axis=0)
        normal_vec = self.compute_normal(vertices[:3])
        perp_vec = np.cross(normal_vec, vertices[1] - vertices[0])
        dot_prod = np.dot(vertices, np.c_[vertices[1] - vertices[0], perp_vec])

        if self.method == "convex_hull":
            hull = scipy.spatial.ConvexHull(dot_prod)
            return vertices[hull.vertices]
        else:
            mean = np.mean(dot_prod, axis=0)
            d = dot_prod - mean
            s = np.arctan2(d[:, 0], d[:, 1])
            return vertices[np.argsort(s)]

    def simplify_faces(self):
        for i, tri1 in enumerate(self.tri_list):
            for j, tri2 in enumerate(self.tri_list):
                if j > i:
                    if self.is_neighbor(tri1, tri2) and self.inv[i] == self.inv[j]:
                        self.group_idx[j] = self.group_idx[i]

        groups = list()
        for i in np.unique(self.group_idx):
            vertices = self.tri_list[self.group_idx == i]
            vertices = np.concatenate([d for d in vertices])
            vertices = self.order_vertices(vertices)
            groups.append(vertices)
        return groups


""" Stupid code
def compute_vertices(R=1):
    vertices = R * np.array(
        [
            [1.21412, 0, 1.58931],
            [0.375185, 1.1547, 1.58931],
            [-0.982247, 0.713644, 1.58931],
            [-0.982247, -0.713644, 1.58931],
            [0.375185, -1.1547, 1.58931],
            [1.96449, 0, 0.375185],
            [0.607062, 1.86835, 0.375185],
            [-1.58931, 1.1547, 0.375185],
            [-1.58931, -1.1547, 0.375185],
            [0.607062, -1.86835, 0.375185],
            [1.58931, 1.1547, -0.375185],
            [-0.607062, 1.86835, -0.375185],
            [-1.96449, 0, -0.375185],
            [-0.607062, -1.86835, -0.375185],
            [1.58931, -1.1547, -0.375185],
            [0.982247, 0.713644, -1.58931],
            [-0.375185, 1.1547, -1.58931],
            [-1.21412, 0, -1.58931],
            [-0.375185, -1.1547, -1.58931],
            [0.982247, -0.713644, -1.58931],
        ]
    )
    return vertices
"""
