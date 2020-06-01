"""
Python playground file used for testing algorithm ideas. 

Things to implement so that we can create the entire coverage control algorithm.
1. Function that approximates the hull using a bounding rectangular prism.
2. Function that computes the projection of the points in the operating domain onto the plane determined by the RED agent.
3. Functin that plots some sort of simulation data.

Steps in the simpler 3D algorithm.
1. First, compute the projection onto the plane determined by the RED agent.
2. Compute the 2D Voronoi partition based on this projection.
3. Obtain the new centroids (this gives you the coordinates in the xy-plane).
4. Move the agents according to these centroids. For the control law to apply, we should also apply the control where you go straight in the z-coordinate.

Also for Jack, create the 3D regions that he needs. These are essentially those 3D hulls.

A lot of the functions below need testing to fix some bugs.

"""

import os
import sys
import time

import shapely
import numpy as np
from scipy.spatial import *
from shapely.geometry import *
import scipy.integrate as scint
import matplotlib.pyplot as plt
from pycoverage.vorutils import pyvoro
from pycoverage3d.vorutils import pyvoro3d


def regularize_environment(vertices, sig_dig=4, create_polygon=False):
    """
    Expands the environment by creating a new environment created from the bounding box of the original convex polygon.
    
    Parameters
    ----------
    vertices : numpy.ndarray instance
        Array containing tuples (x, y, z) representing coordinates in 3-D of the vertices defining the convex polygon.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.
    create_polygon : boolean, optional
        If True, then an instance of shapely.geometry.Polygon is created using the vertices of the bounding box, by default False.

    Returns
    -------
    box_vertices : numpy.ndarray 
        Array containing the ordered vertices of the bounding box, defining the regularized domain.
    box_environment : shapely.geometry.Polygon instance or None
        A shapely.geometry.Polygon instance created by ``box_vertices``. Only created if ``create_polygon`` is True.

    Notes
    -----
    This function is not yet implemented.

    Examples
    --------

    See Also
    --------

    """
    poly = shapely.geometry.Polygon(vertices)
    min_x, min_y, max_x, max_y = poly.bounds
    return None, None


def project_2d(blue_pos, red_pos, sig_dig=4):
    """
    Projects coordinates in ``blue_pos`` to the plane determined by ``red_pos``, and returns the projections as well as the length of the projection vector.
    
    Parameters
    ----------
    blue_pos : numpy.ndarray instance
        Array containing the 3-D coordinates of the BLUE agents.
    red_pos : numpy.ndarray instance
        Array containing the 3-D coordinates of the RED agents.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 siginificant digits.
    
    Returns
    -------
    proj_blue : numpy.ndarray instance
        Array containing the 2-D coordinates of the BLUE agents.
    proj_red : numpy.ndarray instance
        Array containing the 2-D coordinates of the RED agents.
    proj_height : numpy.ndarray instance
        Array containing the height differences from the projection.

    Notes
    -----
    For now, it is assumed that there is only one RED agent that is being tracked at a time, since we are doing a projection. Another strategy needs to be developed for the situation where there are multiple RED agents.

    ``proj_height`` contains the *signed* distance vector from the RED position and the BLUE positions. For a concrete example, see the **Examples** section.

    Examples
    --------

    See Also
    --------

    """
    proj_blue = blue_pos[:, 0:2]
    proj_red = red_pos[:, 0:2]
    proj_height = np.around(red_pos[:, 2] - blue_pos[:, 2], sig_dig)

    return proj_blue, proj_red, proj_height


def compute_projected_voronoi(poly, proj_blue, sig_dig=4):
    """
    Computes the 2-D Voronoi partition using the points in ``proj_blue`` and the geometric information in ``poly``. 
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon instance
        Convex polygon representing the regularized operating environment.
    proj_blue : numpy.ndarray instance
        Array containing the 2-D coordinates of the BLUE agents.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    regions : list
        List containing the regions of the Voronoi parition.
    vertices : list
        List containing the vertices of the convex hull for each region in the partition.

    Notes
    -----
    It is assumed that the vertices of the convex polygon are ordered in a counter-clockwise order. 

    Examples
    --------

    See Also
    --------

    """
    # Compute the Voronoi partition using the 2-D coordinates.
    vor_partition = scipy.spatial.Voronoi(proj_blue)

    # Compute the finite regions.
    vor_regions, vor_vertices = pyvoro.create_finite_voronoi_2d(vor_partition)

    return regions, vertices


def gaussian_function(pos, mu, sigma, scale=1.0, sig_dig=4):
    """
    Returns the value of the Gaussian function with mean ``mu``, variance ``sigma``, and scaling factor ``scale``.
    
    Parameters
    ----------
    pos : numpy.ndarray instance
        Array containing the coordinates in 2-D of the current BLUE agent position.
    mu : numpy.ndarray instance
        Array containing the mean of the Gaussian function.
    sigma : numpy.ndarray instance
        Array containing the variance of the Gaussian function.
    scale : float, optional
        Scaling factor used for ensuring that numerical issues do not arise, by default 1.0.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    val : float
        The value of the Gaussian function with mean ``mu``, variance ``sigma``, and scaling factor ``scale``.
    
    Notes
    -----
    Note that due to the scaling factor ``scale``, this does not return a proper probability distribution (i.e. the range of this function is not in [0,1]) so please take note of this. The scaling factor is used to ensure that there are no numerical accuracy issues.

    Examples
    --------

    See Also
    --------

    """
    dim = mu.shape[1]
    inv_sigma_mat = np.around(np.diag(1.0 / sigma), sig_dig)
    normal_factor = np.around(1 / np.sqrt((2 * np.pi) ** dim * np.prod(sigma)), sig_dig)
    val = np.around(
        normal_factor
        * np.exp(-0.5 * (pos - mu).T @ inv_sigma_mat @ (pos - mu), sig_dig,)
    )

    return val


def dgaussian_function(pos, mu, sigma, step_size, scale=1.0, sig_dig=4):
    """
    Returns the estimate of the derivative of the Gaussian function, parameterized by mean ``mu`, variance ``sigma``, and scaling factor ``scale``.

    The estimation is performed using finite difference methods.
    
    Parameters
    ----------
    pos : numpy.ndarray instance
        Array containing the coordinates in 2-D of the current BLUE agent positions.
    mu : numpy.ndarray instance
        Array containing the mean of the Gaussian function at the current position, and the previous position, respectively.
    sigma : numpy.ndarray instance
        Array containing the variance of the Gaussian function at the current position, and the previous position, respectively.
    step_size : float
        The step size used for the finite differences algorithm.
    scale : float, optional
        Scaling factor used to ensure that numerical issues do not arise, by default 1.0.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    val : float
        The estimate of the time-derivative of the Gaussian function, parametrized by mean ``mu``, variance ``sigma``, and scaling factor ``scale``, which is computed using finite differences.
    
    Notes
    -----
    Note that due to the scaling factor ``scale``, this does not return a proper probability distribution (i.e. the range of this function is not in [0,1]) so please take note of this. The scaling factor is used to ensure that there are no numerical accuracy issues.

    To prevent numerical accuracy issues, the user should ensure that the step size is small. The estimate is computed using finite differences so the larger the step size, the less accurate the estimate, but additionally to prevent overflow, the step size should be chosen carefully.

    Examples
    --------

    See Also
    --------
    pycoverage3d.vorutils.pyvoro3d.gaussian_function : subroutine
    
    """
    val = gaussian_function(
        pos, mu[1, :], sigma[1, :], scale=scale, sig_dig=sig_dig
    ) + gaussian_function(pos, mu[0, :], sigma[0, :], scale=scale, sig_dig=sig_dig)
    return np.around(val, sig_dig)


def tracking_gaussian(pos, mu, sigma, scale=1.0, sig_dig=4):
    """
    Returns the value of the weighted Gaussian function, parametrized by mean ``mu``, variance ``sigma`` and scaling factor ``scale``. 
    
    Parameters
    ----------
    pos : numpy.ndarray instance
        Array containing the coordinates in 2-D of the current BLUE agent position.
    mu : numpy.ndarray instance
        Array containing the mean of the Gaussian function at the current position, and the previous position, respectively.
    sigma : numpy.ndarray instance
        Array containing the variance of the Gaussian function at the current position, and the previous position, respectively.
    scale : float, optional
        Scaling factor used to ensure that numerical issues do no arise, by default 1.0.
    sig_dig : int, optional
        The number of significant digits to trunate the computational result, by default 4 significant digits.

    Returns
    -------
    val : float
        The value of the weighted Gaussian function, parametrized by mean ``mu``, variance ``sigma`` and scaling factor ``scale``.

    Notes
    -----

    Examples
    --------

    See Also
    --------

    """
    val = pos * gaussian_function(pos, mu, sigma, scale=scale, sig_dig=sig_dig)

    return val


def dtracking_function(pos, mu, sigma, scale=1.0, sig_dig=4):
    """
    Returns the estimate of the derivative of the weighted Gaussian function, parametrized by mean ``mu``, variance ``sigma``, and scaling factor ``scale``.
    
    Parameters
    ----------
    pos : numpy.ndarray instance
        Array containing the coordinates in 2-D of the current BLUE agent positions.
    mu : numpy.ndarray instance
        Array containing the mean of the Gaussian function at the current position, and the previous position, respectively.
    sigma : numpy.ndarray instance
        Array containing the variance of the Gaussian function at the current position, and the previous position, respectively.
    step_size : float
        The step size used for the finite differences algorithm.
    scale : float, optional
        Scaling factor used to ensure that numerical issues do not arise, by default 1.0.
    sig_dig : int, optional
        The number of significant digits to truncate the numerical results, by default 4 significant digits.

    Returns
    -------
    val : float
        The estimate of the time-derivative of the Gaussian function, parametrized by mean ``mu``, variance ``sigma``, and scaling factor ``scale``, which is computed using finite differences.

    Notes
    -----

    Examples
    --------

    See Also
    --------
    pycoverage3d.vorutils.pyvoro3d.dgaussian_function : subprocedure

    """
    val = pos * (
        dgaussian_function(pos, mu, sigma, step_size, scale=scale, sig_dig=sig_dig)
    )

    return np.around(val, sig_dig)


def compute_centroids_2d_single_step(
    vor_cell,
    vorinfo,
    proj_blue,
    proj_red,
    sigma,
    step_size,
    alpha,
    scale=1.0,
    sig_dig=4,
):
    """
    Computes the new centroid for coverage control, using the 2-D projected Voronoi partition using information about the BLUE and RED positions.
    
    Parameters
    ----------
    vor_cell : shapely.geometry.Polygon instance
        Convex polygon representing the current Voronoi cell.
    vorinfo : list
        List containing the regions of the Voronoi partition, and the vertices of the convex hull for each region in the partition.
    proj_blue : numpy.ndarray instance
        Array containing the 2-D coordinates of the current BLUE agent that is in the Voronoi cell.
    proj_red : numpy.ndarray instance
        Array containing the 2-D coordinates of the current RED agent position, and the previous RED agent position. 
    sigma : numpy.ndarray instance
        Array containing the current and previous covariance estimate of the RED agent.
    step_size : float
        The step size used for finite differences.
    alpha : float
        The step size used for determining the size of the gradient step.
    scale : float, optional
        Scaling factor used to ensure that numerical issues do not arise, by default 1.0.
    sig_dig : int, optional
        The number of significant digits to truncate the computational results, by default 4 significant digits.

    Returns
    -------
    proj_centroid : numpy.ndarray instance
        Array containing the 2-D coordinates of the centroid locations, computed using information from the RED agent. 

    Notes
    -----
    The ``scale`` argument is passed to ``pycoverage3d.vorutils.pyvoro3d.gaussian_function``. Thus, if ``scale`` is not unity then the returned value is not a proper probability value.

    Examples
    --------

    See Also
    --------
    scipy.integrate.quad : subroutine for quadrature
    pycoverage3d.vorutils.pyvoro3d.gaussian_function : subroutine
    pycoverage3d.vorutils.pyvoro3d.dgaussian_function : subroutine
    pycoverage3d.vorutils.pyvoro3d.tracking_function : subroutine
    pycoverage3d.vorutils.pyvoro3d.dtracking_function : subroutine

    """
    # Compute the new position for the agent.
    min_x, min_y, max_x, max_y = vor_cell.bounds

    # Compute the mass.
    mass_x, _ = scint.quad(
        gaussian_function,
        min_x,
        max_x,
        args=(proj_red[0, :][0], sigma[0, :][0], scale, sig_dig),
    )
    mass_y, _ = scint.quad(
        gaussian_function,
        min_y,
        max_y,
        args=(proj_red[0, :][1], sigma[0, :][1], scale, sig_dig),
    )

    # Compute the derivative of the mass.
    dmass_x, _ = scint.quad(
        dgaussian_function,
        min_x,
        max_x,
        args=(proj_red, sigma, step_size, scale, sig_dig),
    )
    dmass_y, _ = scint.quad(
        dgaussian_function,
        min_y,
        max_y,
        args=(proj_red, sigma, step_size, scale, sig_dig),
    )

    # Compute the new center of mass.
    cent_x, _ = scint.quad(
        tracking_gaussian,
        min_x,
        max_x,
        args=(proj_red[0, :][0], sigma[0, :][0], scale, sig_dig),
    )
    cent_y, _ = scint.quad(
        tracking_gaussian,
        min_y,
        max_y,
        args=(proj_red[0, :][1], sigma[0, :][1], scale, sig_dig),
    )
    cent_x = cent_x / mass_x
    cent_y = cent_y / mass_y

    # Compute the derivative of the centers of mass.
    dcent_x, _ = scint.quad(
        dtracking_function,
        min_x,
        max_x,
        args=(proj_red, sigma, step_size, scale, sig_dig),
    )
    dcent_y, _ = scint.quad(
        dtracking_function,
        min_y,
        max_y,
        args=(proj_red, sigma, step_size, scale, sig_dig),
    )
    dcent_x = (dcent_x - mass_x * cent_x) / mass_x
    dcent_y = (dcent_y - mass_y * cent_y) / mass_y

    # Compute the closest point information, used to compute the new centroid positions.
    closest_point, _ = pyvoro.compute_closest_point(vor_cell, proj_blue)

    # Compute the gradient direction.
    dclosest_point = np.zeros(len(closest_point))
    kappa_x = (dcent_x / (closest_point[0] - cent_x)) - (dmass_x / mass_x)
    kappa_y = (dcent_y / (closest_point[1] - cent_y)) - (dmass_y / mass_y)
    kappa = np.linalg.norm(np.array([kappa_x, kappa_y]))
    dclosest_point[0] = dcent_x - (kappa + (dmass_x / mass_x)) * (
        closest_point[0] - cent_x
    )
    dclosest_point[1] = dcent_y - (kappa + (dmass_y / mass_y)) * (
        closest_point[1] - cent_y
    )

    # Compute the new centroid positions using gradient descent.
    new_x = closest_point[0] + alpha * dclosest_point[0] / np.linalg.norm(
        dclosest_point[0]
    )
    new_y = closest_point[1] + alpha * dclosest_point[1] / np.linalg.norm(
        dclosest_point[1]
    )
    proj_centroid = np.around(np.array([new_x, new_y]), sig_dig)

    return proj_centroid
