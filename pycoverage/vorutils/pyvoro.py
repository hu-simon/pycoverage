""" Python file containing utilities for basic Voronoi operations.

Some other documentation here, explaining more about this module.

"""

__author__ = "Jun Hao (Simon) Hu"
__copyright__ = "Copyright (c) 2020, Simon Hu"
__version__ = "0.2"
__maintainer__ = "Simon Hu"
__email__ = "simonhu@ieee.org"
__status__ = "Development"

import os
import sys
import warnings

warnings.simplefilter("module", PendingDeprecationWarning)

import shapely
import numpy as np
from scipy.spatial import *
from shapely.geometry import *
from shapely.ops import triangulate


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def generate_random_points(poly, num_points, min_dist=0.0, sig_dig=4):
    """
    Generates random tuples of the form (x, y) which lie in the interior of a convex hull and are separated by ``min_dist`` distance, using rejection sampling. It is assumed that the hull vertices are ordered in counter-clockwise order.
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon instance
        Convex hull used for determining membership during rejection sampling.
    num_points : int
        Number of points to be generated.
    min_dist : float, optional
        The minimum separation distance enforced for each point, by default 0.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    random_points : list
        Unordered list of tuples of the form (x, y) representing the randomly sampled points which lie within the convex hull, and are separated by a distance of ``min_dist``. 

    Notes
    -----
    This function does not compute the packing number, which is needed to determine if it is possible to generate ``num_points`` points that are separated by a distance of ``min_dist`` distance. It is the responsibility of the user to ensure that the problem is feasible.

    It is assumed that the hull vertices are ordered in counter-clockwise order.
    
    Examples
    --------
    >>> # Generates 3 points with ``min_dist`` of 1.
    >>> hull = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> random_points = geoutils.generate_random_points(hull, num_points=3, min_dist=1.)

    >>> # Fail case where the problem is not feasible. The result is an infinite runtime!
    >>> hull = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> random_points = geoutils.generate_random_points(hull, num_points=100, min_dist=100.)
    
    TODO need to fix the examples so that they work with the sig_dig option.

    """
    min_x, min_y, max_x, max_y = poly.bounds

    x_random = list()
    y_random = list()

    x_sample = np.around(np.random.uniform(min_x, max_x), sig_dig)
    y_sample = np.around(np.random.uniform(min_y, max_y), sig_dig)

    first_point = False

    while first_point is False:
        if poly.contains(Point(x_sample, y_sample)):
            x_random.append(x_sample)
            y_random.append(y_sample)
            first_point = True
        else:
            x_sample = np.around(np.random.uniform(min_x, max_x), sig_dig)
            y_sample = np.around(np.random.uniform(min_y, max_y), sig_dig)

    while len(x_random) < num_points:
        x_sample = np.around(np.random.uniform(min_x, max_x), sig_dig)
        y_sample = np.around(np.random.uniform(min_y, max_y), sig_dig)

        if poly.contains(Point(x_sample, y_sample)):
            distances = np.sqrt(
                (x_sample - np.array(x_random)) ** 2
                + (y_sample - np.array(y_random)) ** 2
            )
            min_distance = np.min(distances)
            if min_distance >= min_dist:
                x_random.append(x_sample)
                y_random.append(y_sample)
        else:
            continue

    random_points = [[x_random[i], y_random[i]] for i in range(len(x_random))]

    return np.array(random_points)


def generate_points_within_polygon(poly, num_points):
    """ Generates random tuples of the form (x,y) which lie in the interior of a polygon.

    Parameters
    ----------
    poly : shapely.geometry.Polygon instance
        Polygon used for determining membership.
    num_points : int
        Number of points to generate.
    
    Returns
    -------
    random_points : list
        Unordered list of tuples in the form (x,y) representing the randomly sampled points, which lie within the polygon.

    Notes
    -----
    Further support for this function is DEPRECATED. Please use ``generate_random_points()`` instead.
    
    This function has the same functionality as generate_random_points(), as min_dist=0 is enforced. Additionally, this function is much faster than generate_random_points() even with min_dist=0 as an argument since no check is performed on separation distance. 
    
    This function does not compute the packing number, which is needed to determine if it is possible to generate num_points points. It is up to the user to ensure this is enforced before passing arguments onto this function.

    It is assumed that the polygon vertices are ordered in counter-clockwise order.

    Examples
    --------
    >>> # Generates 3 points contained within a polygon.
    >>> polygon = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> random_points = pyvoro.generate_points_within_polygon(polygon, num_points=100)
    [[-53.10904500583265, -7.531685953052616],
     [13.908619808266579, 58.8310992632654],
     [49.05513248457106, 10.135106966482866]]

    See Also
    --------
    generate_random_points
    """

    random_points = list()

    min_x, min_y, max_x, max_y = poly.bounds

    while len(random_points) < num_points:
        point = Point(
            [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
        )
        if point.within(poly):
            random_points.append(point.xy)

    random_points = [
        [random_points[i][0][0], random_points[i][1][0]] for i in range(num_points)
    ]
    return random_points


def compute_polygon_area(points, sig_dig=4):
    """
    Computes the area of a convex polygon defined by points using a shoelace formula.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coordinates in 2-D of the points defining the convex polygon. 
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    area : float
        Area of the convex polygon, defined by ``points``, truncated to ``sig_dig`` significant digits.

    Examples
    --------
    >>> # TODO

    """
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area = np.around(area, sig_dig)

    return area


def compute_polygon_centroid(points, sig_dig=4):
    """
    Computes the geometric centroid of the convex polygon, defined by ``points``, using a shoelace formula.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coordinates in 2-D of the points defining the convex polygon.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    centroid : numpy.ndarray instance
        Geometric centroid of the convex polygon, defined by ``points``, truncated to ``sig_dig`` significant digits.

    Examples
    --------
    >>> # TODO
    
    """
    x = points[:, 0]
    y = points[:, 1]
    area = compute_polygon_area(points, sig_dig)

    centroid_x = np.dot((x + np.roll(x, 1)), (x * np.roll(y, 1) - np.roll(x, 1) * y))
    centroid_y = np.dot((y + np.roll(y, 1)), (x * np.roll(y, 1) - np.roll(x, 1) * y))
    centroid = (1 / (6 * area)) * np.array([centroid_x, centroid_y])
    centroid = np.around(centroid, sig_dig)

    return centroid


def create_finite_voronoi_2d(vor, radius=None):
    """ Given an infinite Voronoi partition, creates a finite Voronoi partition in 2D by extending the infinite ridges and taking intersections of sets.
    
    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Object containing information about the infinite Voronoi partition.
    radius : float, optional
        Float representing the distance to the "point at infinity", i.e. the point at which all the infinite ridges are extended to. If no radius is specified, we take the farthest point in the partition and multiply it by 100.
    
    Returns
    -------
    regions : list
        Indices of the vertices in the new finite Voronoi partition.
    vertices : list
        List of coordinates, in (x,y), of the new finite Voronoi partition. These are the same coordinates returned by scipy.spatial.Voronoi() but the points at infinity are also appended to this list.

    Notes
    -----
    This code has been adapated from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647. Please refer to the thread for more information.

    For best results, it is best to leave the radius argument alone. Do not be afraid of the large numbers that end up in the result. This is expected, as we are placing the point at infinity very far away, for a proper partition even in worst case scenarios.

    It is assumed that the polygon vertices are ordered in a counter-clockwise order.

    Examples
    --------
    >>> # Generate a finite 2D Voronoi object.
    >>> polygon = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> points = [[0, 0], [20, 30], [-50, 30], [-10, -10]]
    >>> vor = scipy.spatial.Voronoi(points)
    >>> regions, vorinfo = pyvoro.create_finite_voronoi_2d(vor, radius=None)
    >>> print(regions)
    [[3, 2, 0, 1], [0, 4, 5], [7, 1, 0, 6], [9, 8, 1]]
    >>> print(vorinfo)
    [[  -15.            31.66666667]
     [  -25.            15.        ]
     [ 6641.4023547  -4405.93490314]
     [ 5631.85424949 -5641.85424949]
     [ 6641.4023547  -4405.93490314]
     [  -15.          8031.66666667]
     [  -15.          8031.66666667]
     [-5681.85424949 -5641.85424949]
     [ 5631.85424949 -5641.85424949]
     [-5681.85424949 -5641.85424949]]
    """
    if vor.points.shape[1] != 2:
        raise ValueError("2D input is required.")

    regions_new = list()
    vertices_new = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 100

    # Construct a map containing all of the ridges for a single point.
    ridges_all = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges_all.setdefault(p1, []).append((p2, v1, v2))
        ridges_all.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct the infinite regions.
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        # Reconstruct the finite regions.
        if all(v >= 0 for v in vertices):
            # If Qhull reports >=0 then this is a finite region, so we should add it to the list.
            regions_new.append(vertices)
            continue

        # Reconstruct the non-finite regions.
        ridges = ridges_all[p1]
        region_new = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # This is again a finite ridge and is already in the region.
                continue

            # Compute the missing endpoint for the infinite ridge.
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            point_far = vor.vertices[v2] + direction * radius

            region_new.append(len(vertices_new))
            vertices_new.append(point_far.tolist())

        # Sort the region counterclockwise.
        sorted_vertices = np.asarray([vertices_new[v] for v in region_new])
        c = sorted_vertices.mean(axis=0)
        angles = np.arctan2(sorted_vertices[:, 1] - c[1], sorted_vertices[:, 0] - c[0])
        region_new = np.array(region_new)[np.argsort(angles)]

        regions_new.append(region_new.tolist())
    return regions_new, np.asarray(vertices_new)


def find_polygon_centroids(poly, points):
    """ Computes the new Voronoi points, which are taken to be the centroids of the previous Voronoi partitions. 
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Polygon used for determining intersection membership.
    points : list
        List of tuples (x,y) representing the current positions of agents.

    Returns
    -------
    vorinfo : list
        List containing the regions of the Voronoi partition, and the vertices of the convex hull for each region in the partition.
    centroids : list
        List of tuples (x,y) representing the new centroid positions. 

    Notes 
    -----
    It is assumed that the polygon vertices are ordered in a counter-clockwise order.

    Examples
    --------
    >>> # Find the regions of the Voronoi partition.
    >>> polygon = shapely.geometry.Polygon([[0, 100], [-95, 31], [-59, -81], [95, 31]])
    >>> points = [[0, 0], [20, 30], [-50, 30], [-10, -10]]
    >>> vorinfo, centroids = pyvoro.find_polygon_centroids(polygon, points)
    >>> print(vorinfo[0]) # Regions of the partition.
    [[3, 2, 0, 1], [0, 4, 5], [7, 1, 0, 6], [9, 8, 1]]   
    >>> print(vorinfo[1]) # Vertices defining the convex hull of the regions.
    array([[  -15.        ,    31.66666667],
           [  -25.        ,    15.        ],
           [ 6641.4023547 , -4405.93490314],
           [ 5631.85424949, -5641.85424949],
           [ 6641.4023547 , -4405.93490314],
           [  -15.        ,  8031.66666667],
           [  -15.        ,  8031.66666667],
           [-5681.85424949, -5641.85424949],
           [ 5631.85424949, -5641.85424949],
           [-5681.85424949, -5641.85424949]])
    >>> print(centroids)
    [[7.043367194231445, 1.9724402251231876], 
     [28.929774946440688, 43.64720832302304], 
     [-51.68553805648806, 27.959361526264278], 
     [-33.479077928361086, -31.697626888791707]]

    See Also
    --------
    pyvoro.create_finite_voronoi_2d : subprocedure

    TODO need to fix the examples so that they work with sig_dig option.
    """
    vor = Voronoi(points)
    regions, vertices = create_finite_voronoi_2d(vor)

    centroids = list()
    polygons = list()

    for region in regions:
        polygon_points = vertices[region]
        polygon = Polygon(polygon_points).intersection(poly)
        centroids.append(list(polygon.centroid.coords))

    centroids = [
        [centroids[i][0][0], centroids[i][0][1]] for i in range(len(centroids))
    ]
    return [regions, vertices], centroids


def find_enclosing_ball(points, sig_dig=4, epsilon=1e-2, stop_tol=1e-2):
    """
    Computes an approximation to the minimum enclosing ball that encloses the convex polygon defined by ``points``. 

    The minimum enclosing ball is computed using the Frank-Wolfe algorithm (insert reference here) and iteratively converges to the minimum enclosing ball.
    
    Parameters
    ----------
    points : numpy.ndarray instance
        Array containing tuples (x, y) representing coorddinates in 2-D of the points defining the convex polygon.
    sig_dig : int, optional
        The number of siginificant digits to truncate the computational ressults, by default 4 significant digits.
    epsilon : float, optional
        Parameter determining the acceptable error of the approximation and number of iterations, by default 1e-3.
    stop_tol : float, optional
        Stopping tolerance used for ending iterations early. 

    Returns
    -------
    circle : numpy.ndarray instance
        Array containing the coordinates (x, y, radius) that define the minimum enclosing ball.

    Examples
    --------
    >>> # TODO

    Notes
    -----
    Note that the smaller ``epsilon`` is, the more accurate the result. However, this also increases the number of iterations by a factor of two-fold, so choose this parameter wisely.

    The parameter ``stop_tol`` is used to determine if the algorithm should be terminated early. If the absolute difference between the newly computed center and the previous center is smaller than ``stop_tol``, then the algorithm is terminated early.

    """
    # Use the centroid as the initial point.
    center = compute_polygon_centroid(points, sig_dig)
    radius = np.max(np.linalg.norm(center - points, axis=1))

    # Determine the number of iterations.
    num_iters = (int)(np.ceil(1 / epsilon ** 2))

    # Iteratively compute the center of the minimum enclosing ball.
    for k in range(num_iters):
        print("Iteration {}".format(k))
        prev_center = center.copy()

        distances = np.linalg.norm(center - points, axis=1)
        idx_max = np.argmax(distances)
        center = (k / (k + 1)) * center + (1 / (k + 1)) * points[idx_max]
        radius = distances[idx_max]

        # if np.linalg.norm(center - prev_center) ** 2 < stop_tol:
        #    print("Stopping condition satisfied, terminating algorithm...")
        #    break

    circle = np.array([center[0], center[1], radius])

    return circle


def project_to_hull(vertices, point, sig_digs=4):
    """
    Projects a point onto the boundary of the convex hull.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array containing tuples (x, y) representing coordinates in 2-D of the vertices defining the convex polygon.
    point : numpy.ndarray
        Array containing tuple (x, y) representing coordinates in 2-D of the point we want to project onto the convex polygon.
    sig_digs : float, optional
        The number of significant digits to truncate the computatioanl results, by default 4 siginificant digits.

    Returns
    -------
    proj_point : list
        List containing tuple (x, y) representing the coordinates in 2-D of the projection of ``point`` onto the convex polygon defined by ``vertices``, and the distance between the point and the projection.
    
    Examples
    --------
    >>> # TODO

    Notes
    -----
    It is assumed that ``point`` represents a point that is outside the convex polygon.
    
    """
    weights = cvx.Variable(vertices.shape[0])

    obj = cvx.Minimize(cvx.sum_squares(weights @ vertices - point))
    constr = [weights >= 0, cvx.sum(weights) == 1]
    prob = cvx.Problem(obj, constr)

    value = prob.solve()
    proj_point = [
        np.around(weights.value @ vertices, sig_digs),
        np.around(value, sig_digs),
    ]

    return proj_point


def create_triangulation(poly, points, scheme=None):
    """ Computes the triangulation of the convex hull using either the defualt scheme, or by choosing the centroid as the leverage point (delaunay).
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon instance
        Polygon representing the domain to be triangulated.
    points : list
        List of points that form the convex hull. Asssumed to be in counter-clockwise order.
    scheme : string, optional
        String representing which scheme to use, by default None. If None, then the default scheme is used where the points that form the hull are used, and if "delaunay" then the points from the hull and the centroid of the polygon are used.

    Returns
    -------
    triangulation : list
        List containing the vertices that completely define the triangles that form the triangulation.
    """
    if scheme is None:
        triangulation = shapely.ops.triangulate(poly)

    elif scheme is "delaunay":
        # Obtain the points of the convex hull.
        tri_points = [
            (poly.exterior.xy[0][i], poly.exterior.xy[1][i])
            for i in range(len(poly.exterior.xy[0]))
        ]

        # Add the centroid to the list of points as well.
        tri_points.append((poly.centroid.xy[0][0], poly.centroid.xy[1][0]))
        triangulation = shapely.ops.triangulate(shapely.geometry.MultiPoint(tri_points))

    else:
        print("Selected triangulation scheme is not supported. Returning None.")
        triangulation = None

    return triangulation


def get_triangulation_points(tri):
    """ Returns a list containing tuples of the form [x1,x2] that represent the points of the triangulation. 
    
    Parameters
    ----------
    tri : list
        List containing Shapely triangulation objects, representing the triangles that make up the triangulation. 

    Returns
    -------
    points : list
        List of tuples of the form (x1, x2, ...) containing the vertices of the triangles that define the triangulation.

    Finish documentation later. 
    """
    points = list()
    for tri_idx, triangle in enumerate(tri):
        sublist = list()
        for i in range(len(triangle.exterior.xy[0]) - 1):
            sublist.append((triangle.exterior.xy[0][i], triangle.exterior.xy[1][i]))
        points.append(sublist)

    return points


def to_canonical_triangle(points):
    """
    
    Parameters
    ----------
    points : list of tuples
        List of points representing the vertices of the triangle.

    Returns
    -------

    Finish documentation later.
    """
    pass


def vectorize_lambda_function(args, derivative=False):
    """ Returns a vectorized version of the lambda function.
    
    Parameters
    ----------
    func : lambda function
        Python lambda function that represents the function we want to vectorize.
    args : dict
        Dictionary of arguments sent to the original lambda function.
    derivative : bool
        Boolean flag denoting whether or not the lambda function represents a derivative term.

    Returns
    -------
    func : lambda function
        A vectorized (using numpy.vectorize) lambda function that can be fed into quadrature rules.
    
    Notes
    -----
    * THIS FUNCTION IS PROBABLY GOING TO BE DEPRECATED.
    * The args dict takes the following categories: func_type, time_curr, time_prev, mu, sigma, step_size. The mu_x and sigma_x can also be Python lambda functions, but time_curr and time_prev must be a float and func_type must be a string.
    * The variable func_type is used to determine which lambda function we return. The values it can take on are: "joint" and "weighted".
    * The derivative flag is used to determine whether or not the derivative term is returned, instead of the non-derivative term.

    Finish documentation later & make this function prettier...
    """
    # Parse arguments dictionary.
    func_type = args["func_type"]
    t1 = args["time_curr"]
    t0 = args["time_prev"]
    mu_xt = args["mu_x"]
    sigma_xt = args["sigma_x"]
    mu_yt = args["mu_y"]
    sigma_yt = args["sigma_y"]

    if derivative is False:
        mu_x = mu_xt(t1)
        mu_y = mu_yt(t1)
        sigma_x = sigma_xt(t1)
        sigma_y = sigma_yt(t1)
        if func_type is "joint":
            return np.vectorize(
                lambda x: 1
                / (2 * np.pi * sigma)
                * np.exp(-0.5 * ((x - mu) ** 2 / (2 * sigma ** 2)))
            )
        elif func_type is "weighted":
            return np.vectorize(
                lambda x: x
                * (
                    (
                        1
                        / (2 * np.pi * sigma)
                        * np.exp(-0.5 * ((x - mu) ** 2 / (2 * sigma ** 2)))
                    )
                )
            )
        else:
            print(
                "The function type you selected is not supported. Please see documentation. Returning None."
            )
            return None
    elif derivative is True:
        mu1 = mu_t(t1)
        mu0 = mu_t(t0)
        sigma1 = sigma_t(t1)
        sigma0 = sigma_t(t0)
        h = args["step_size"]

        if func_type is "joint":
            return np.vectorize(
                lambda x: (
                    1
                    / (2 * np.pi * sigma1)
                    * np.exp(-0.5 * ((x - mu1) ** 2 / (2 * sigma1 ** 2)))
                    - 1
                    / (2 * np.pi * sigma0)
                    * np.exp(-0.5 * ((x - mu0) ** 2 / (2 * sigma0 ** 2)))
                )
                / h
            )
        elif func_type is "weighted":
            return np.vectorize(
                lambda x: x
                * (
                    1
                    / (2 * np.pi * sigma1)
                    * np.exp(-0.5 * ((x - mu1) ** 2 / (2 * sigma1 ** 2)))
                    - x
                    * (
                        1
                        / (2 * np.pi * sigma0)
                        * np.exp(-0.5 * ((x - mu0) ** 2 / (2 * sigma0 ** 2)))
                    )
                )
                / h
            )
        else:
            print(
                "The function type you selected is not supported. Please see documentation. Returning None."
            )
            return None


def multivariate_gaussian(
    x, mu, sigma, scaling_factor=10000, vectorize=False, vector=False
):
    """ Returns the value of a multi-variate Gaussian with mean mu and standard variation sigma.
    
    To prevent division by zero due to a small mass integral output, the function is multiplied by a large constant. Therefore, this is NOT a proper probability distribution. 
    
    Parameters
    ----------
    x : array-like
        The point at which the function is evaluated.
    mu : array-like
        The mean of the multivariate Gaussian.
    sigma : array-like
        The standard deviation of the multivariate Gaussian.
    vectorize : bool, optional
        Flag determining whether or not the result is vectorized using Numpy, by default False.
    
    Returns
    -------
    func : array-like
        The value of the multivariate Gaussian function at the prescribed point x. Potentially vectorized using Numpy based on the vectorized flag.

    Examples
    --------
    >>> 123
    """
    # Dimension of the Gaussian.
    dim = 2

    # Create the inverse covariance matrix.
    sigma_mat = np.diag(1.0 / sigma)
    # Compute the normalization factor.
    normal_factor = 1 / np.sqrt((2 * np.pi) ** dim * np.prod(sigma))

    """
    if vectorize:
        return np.vectorize(
            normal_factor * np.exp(-0.5 * (x - mu).T @ sigma_mat @ (x - mu))
        )
    else:
        # print(normal_factor * np.exp(-0.5 * (x - mu).T @ sigma_mat @ (x - mu)))
        return normal_factor * np.exp(-0.5 * (x - mu).T @ sigma_mat @ (x - mu))
    """
    total = 1
    if vector:
        return (
            scaling_factor
            * normal_factor
            * np.exp(-0.5 * (x - mu).T @ sigma_mat @ (x - mu))
        )
    else:
        for i in range(dim):
            total *= np.exp(-0.5 * ((x[i] - mu[i]) ** 2 / sigma[i]))
        if vectorize:
            return np.vectorize(scaling_factor * normal_factor * total)
        else:
            return scaling_factor * normal_factor * total


def weighted_center_function(x, mu, sigma, vectorize=False, vector=False):
    """ Returns the value of the weighted center of mass function, weighted by phi.
    
    To prevent division by zero due to small mass integral output, the function is multiplied by a large constant. Therefore, this is NOT a proper probability distribution.

    Parameters
    ----------
    x : array-like
        The point at which the function is evaluated.
    mu : array-like
        The mean of the multivariate Gaussian.
    sigma : array-like
        The standard deviation of the multivariate Gaussian.
    vectorize : bool, optional
        Flag determining whether or not the result is vectorized using Numpy, by default False.

    Returns
    -------
    func : array-like
        The value of the multivariate Gaussian function at the prescribed point x. Potentially vectorized using Numpy based on the vectorized flag. 

    Examples
    --------
    >>> 123
    """
    # Dimension of the Gaussian.
    dim = 2

    # Create the inverse covariance matrix.
    sigma_mat = np.diag(1.0 / sigma)
    # Compute the normalization factor.
    normal_factor = 1 / np.sqrt((2 * np.pi) ** dim * np.prod(sigma))

    total = 0
    for i in range(2):
        total += x[i] * np.exp(-0.5 * ((x[i] - mu[i]) ** 2 / sigma[i] ** 2))
    if vectorize:
        return np.vectorize(x * multivariate_gaussian(x, mu, sigma))
    else:
        return normal_factor * total


def dmultivariate_gaussian(x, mu, sigma, time_step, scheme="fd", vectorize=False):
    """ Returns the derivative of the multi-variate Gaussian, with respect to time, with mean mu and standard deviation sigma. 
    
    To prevent division by zero due to a small mass integral output, the function is multiplied by a large constant. Therefore, this is NOT a proper probability distribution.
    
    Parameters
    ----------
    x : array-like
        The point at which the function is evaluated.
    mu : array-like
        A (dim x 2, 3) array containing the mean of the multivariate Gaussian at either 2 time steps or 3 time steps. Can pass any number of time steps, but only the first 2 or 3 columns are used. 
    sigma : array-like
        A (dim x 2, 3) array containing the standard deviation of the multivariate Gaussian at either 2 time steps or 3 time steps. Can pass any number of time steps, but only the first 2 or 3 columns are used.
    time_step : float
        The time step used in the numerical scheme.
    scheme : string, optional
        The numerical scheme used, by default "fd". The options are "fd" for forward difference scheme (current value and previous value are used), and "cd" for central differencing scheme (current value, and two previous values are used). 
    vectorize : bool, optional
        Flag determining whether or not the result is vectorized using Numpy, by default False.

    Returns
    -------
    func : array-like
        The value of the multivariate Gaussian function at the prescribed point x. Potentially vectorized using Numpy based on the vectorized flag.

    Notes
    -----
    As of version 0.2, central differencing is not supported. Most likely, central differencing will never be supported.

    Examples
    --------
    >>> 123
    """
    if scheme is "fd":
        if vectorize:
            return np.vectorize(
                (
                    multivariate_gaussian(x, mu[0], sigma[0])
                    - multivariate_gaussian(x, mu[1], sigma[1])
                )
                / time_step
            )
        else:
            return (
                multivariate_gaussian(x, mu[0], sigma[0])
                - multivariate_gaussian(x, mu[1], sigma[1])
            ) / time_step
    else:
        print("The selected scheme is not supported. Returning None.")
        return None


def dweighted_center_function(x, mu, sigma, time_step, scheme="fd", vectorize=False):
    """ Returns the value of the weighted center of mass function, with respect to the function phi. 

    To prevent division by zero due to a small mass integral output, the function is multiplied by a large constant. Therefore, this is NOT a proper probability distribution.
    
    Parameters
    ----------
    x : array-like
        The point at which the function is evaluated.
    mu : array-like
        A (dim x 2, 3) array containing the mean of the multivariate Gaussian at either 2 time steps or 3 time steps. Can pass any number of time steps, but only the first 2 or 3 columns are used.
    sigma : array-like
        A (dim x 2, 3) array containing the standard deviation of the multivariate Gaussian at either 2 time steps or 3 time steps. Can pass any number of time steps, but only the first 2 or 3 columns are used. 
    time_step : float
        The time step used for the numerical approximation.
    scheme : string, optional
        The numerical scheme used, by default "fd". The options are "fd" for forward difference scheme (current value and previous value are used), and "cd" for central differencing scheme (current value, and two previous values are used).
    vectorize : bool, optional
        Flag determining whether or not the result is vectorized using Numpy, by default False.
    
    Returns
    -------
    func : array-like
        The value of the multivariate Gaussian function at the prescribed point x. Potentially vectorized using Numpy based on the vectorized flag.
    
    Examples
    --------
    >>> 123
    """
    if vectorize:
        return np.vectorize(
            x
            * dmultivariate_gaussian(
                x=x, mu=mu, sigma=sigma, time_step=time_step, scheme=scheme
            )
        )
    else:
        return x * dmultivariate_gaussian(
            x=x, mu=mu, sigma=sigma, time_step=time_step, scheme=scheme
        )


def compute_closest_point(poly, points):
    """ Finds the closest point that is contained inside the polygon. 
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Polygon representing the domain of operation.
    points : list
        List of current agent positions.

    Returns
    -------
    closest_point : tuple
        (x,y) coordinates of the closest agent.
    idx_closest_point : int
        Index of the closest point contained in the polygon.

    Notes
    -----
    If used properly, under the Voronoi partitions framework, or actually any other framework where we have a proper partition, i.e. disjoint regions with no overlap, with respect to the agent positions, then this returns a unique agent position. Otherwise, no guarantees. 
    
    The best way to use this function is essentially to treat the input poly as ONE Voronoi region and find the coordinates of the agent who is assigned this Voronoi region. 

    It is assumed that the polygon vertices ordered in a counter-clockwise order.

    Examples
    --------
    >>> # Find coordinates of agent that is assigned to a rectangle.
    >>> polygon = shapely.geometry.Polygon([[0, 0], [10, 10], [0, 10], [10, 0]])
    >>> points = [[5, 5], [20, 20], [80, 80], [100, 100]]
    >>> closest_point, idx_closest_point = pyvoro.compute_closest_point(polygon, points)
    >>> print(closest_point)
    [5, 5]
    >>> print(idx_closest_point)
    0
    """
    closest_point = None
    idx_closest_point = None
    for idx, point in enumerate(points):
        if poly.contains(Point(point)):
            closest_point = point
            idx_closest_point = idx
        else:
            continue
    if closest_point is None:
        print(poly)
        print(points)
    return closest_point, idx_closest_point


def create_transparent_cmap(cmap, N=255):
    """ Generates a transparent colormap based on a matplotlib.pyplot colormap.
    
    Parameters
    ----------
    cmap : matplotlib.pyplot.cmap instance
        Cmap object containing information about the colormap. 
    N : int, optional
        Value used in determining alpha, which determines the opacity of the transparent colormap, by default 255. The value of N should be between 0 and 255.

    Returns
    -------
    transparent_cmap : matplotlib.pyplot.cmap instance
        Cmap object containing information about the transparent colormap. 

    Examples
    --------
    >>> # Generates a transparent colormap based on the RED colormap scheme from Matplotlib.
    >>> from matplotlib import pyplot as plt
    >>> transparent_cmap = pyvoro.create_transparent_cmap(plt.cm.Reds)
    """
    transparent_cmap = cmap
    transparent_cmap._init()
    transparent_cmap._lut[:, -1] = np.linspace(0, 0.7, N + 4)
    return transparent_cmap
