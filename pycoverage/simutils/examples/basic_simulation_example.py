""" A basic, low-fidelity simulation that uses the BasicSimulation class from the MissionControl module. 

This is a low-fidelity simulation to test the coverage control tracking algorithm.

"""

import time

import numpy as np
from pycoverage.simutils.MissionControl import BasicSimulation

# Auxiliary functions that are used for the simulations.
# These are Gaussian functions, not necessarily a probability distribution.
def joint_xyt(x, y, t, mu_x, mu_y, sigma_x, sigma_y):
    return (1 / 2 * np.pi * sigma_x(t) * sigma_y(t)) * np.exp(
        -0.5
        * (((x - mu_x(t)) ** 2) / (sigma_x(t)) + ((y - mu_y(t)) ** 2) / (sigma_y(t)))
    )


def joint_xt(x, t, mu_x, sigma_x):
    return (1 / (2 * np.pi * sigma_x(t))) * np.exp(
        -0.5 * ((x - mu_x(t)) ** 2 / (sigma_x(t)))
    )


def weighted_xt(x, t, mu_x, sigma_x):
    return x * (
        (1 / (2 * np.pi * sigma_x(t)))
        * np.exp(-0.5 * ((x - mu_x(t)) ** 2 / (sigma_x(t))))
    )


def cost_function_xt(x, t, closest_point, mu_x, sigma_x):
    return -((x - closest_point) ** 2) * joint_xt(x, t, mu_x, sigma_x)


def djoint_xt(x, t1, t0, mu_x, sigma_x, h):
    return (joint_xt(x, t1, mu_x, sigma_x) + joint_xt(x, t0, mu_x, sigma_x)) / h


def dweighted_xt(x, t1, t0, mu_x, sigma_x, h):
    return x * ((joint_xt(x, t1, mu_x, sigma_x) + joint_xt(x, t0, mu_x, sigma_x)) / h)


def basic_example():
    # Define the operating environment as a set of vertices that define the convex hull.
    vertices = [[-50, 50], [-50, -50], [50, -50], [50, 50]]
    env_params = {"points": vertices}

    # Define the simulation parameters.
    kappa = 1
    alpha = 1
    dimension = 2
    num_frames = 100
    num_blue = 5
    min_dist = 5
    sim_params = {
        "kappa": kappa,
        "alpha": alpha,
        "init_points": None,
        "tracking_schemes": "new",
        "dimension": dimension,
        "num_frames": num_frames,
        "num_blue": num_blue,
        "min_dist": 10,
    }

    # Define the adversarial parameters.
    mu_ax = 50
    mu_bx = -50
    mu_ay = -50
    mu_by = 50
    sigma_ax = 50
    sigma_bx = 25
    sigma_ay = 50
    sigma_by = 25

    heat_params = {
        "mu_xt": lambda t: mu_ax * (1 - t / num_frames) + mu_bx * (t / num_frames),
        "mu_yt": lambda t: mu_ay * (1 - t / num_frames) + mu_by * (t / num_frames),
        "sigma_xt": lambda t: sigma_ax * (1 - t / num_frames)
        + sigma_bx * (t / num_frames),
        "sigma_yt": lambda t: sigma_ay * (1 - t / num_frames)
        + sigma_by * (t / num_frames),
        "joint_xt": joint_xt,
        "djoint_xt": djoint_xt,
        "joint_xyt": joint_xyt,
        "weighted_xt": weighted_xt,
        "dweighted_xt": dweighted_xt,
        "cost_xt": cost_function_xt,
    }

    # Initialize the BasicSimulation object.
    simulation = BasicSimulation(sim_params, env_params, heat_params)

    # Obtain the simulation data.
    simulation.obtain_simulation_data()

    # Create the visualization.
    simulation.create_simulation(pause_rate=0.01)


if __name__ == "__main__":
    basic_example()
