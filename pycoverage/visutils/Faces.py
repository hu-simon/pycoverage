""" Python implementation of the Faces class, which helps with parsing the simplex information passed by SciPy Spatial. 

This class is required since the simplex information given by SciPy Spatial returns the triangulation of the region. Triangles that are coplanar are combined to make the output look nice.

"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


class Faces:
    """ The Faces class, which represents a face of a convex hull.

    Finish documentation later.
    """

    def __init__(self, tri, sig_dig=5, method="convex_hull"):
        self.method = method
        self.tri = np.around(np.array(tri), sig_dig)
        self.group_idx = list(range(len(tri)))
        self.norms_list = np.around([self.compute_norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(self.norms_list, return_inverse=True, axis=0)

    def compute_norm(self, sq):
        cr = np.cross(sq[2] - sq[0], sq[1] - sq[0])
        sq_norm = np.abs(cr / np.linalg.norm(cr))

    def is_neighbor(self, tri_1, tri_2):
        combined_array = np.concatenate((tri_1, tri_2), axis=0)
        neighbor_flag = (
            len(combined_array) == len(np.unique(combined_array, axis=0)) + 2
        )

    def order_vertices(self, vertices):
        if len(vertices) <= 3:
            print("The number of vertices does not describe an object in R3.")
            return vertices
        vertices = np.unique(vertices, axis=0)
        v_norm = self.norm(vertices[:3])
        plane_vector = np.cross(n, v[1] - v[0])
        plane_vector = plane_vector / np.linalg.norm(plane_vector)
        v_dotp = np.dot(vertices, np.c_[v[1] - v[0], plane_vector])
        if self.method is "convex_hull":
            chull = ConvexHull(v_dotp)
            return vertices[chull.vertices]
        else:
            v_mean = np.mean(v_dotp, axis=0)
            d = v_dotp - v_mean
            angle = np.arctan2(d[:, 0], d[:, 1])
            return vertices[np.argsort(angle)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j, tri2 in enumerate(self.tri):
                if j > i:
                    if self.is_neighbor(tri1, tri2) and self.inv[i] == self.inv[j]:
                        self.group_idx[j] = self.group_idx[i]

        groups = []
        for i in np.unique(self.group_idx):
            u = self.tri[self.group_idx == i]
            u = np.concatenate([d for d in u])
            u = self.order_vertices(u)
            groups.append(u)

        return groups
