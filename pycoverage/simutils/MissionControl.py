""" Python class used for creating low and high fidelity simulations.

Some more words here to further explain this module.

"""

import time

import shapely
import numpy as np
import quadpy as qp
from scipy.spatial import *
from shapely.geometry import *
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pycoverage.vorutils import pyvoro


class MissionControl:
    def __init__(self, sim_info):
        pass


class BasicSimulation:
    """ A high-level low-fidelity class for creating simulations. 

    Attributes
    ----------
    sim_params : dict
        Dictionary containing the simulation parameters.
    env_params : dict 
        Dictionary containing the environment parameters.
    heat_params : dict
        Dictionary containing the adversarial parameters.
    operating_env : Shapely Polygon
        Polygon representing the operating environment.
    cost_data : array-like
        Array-like object containing information about the cost value from the simulation.
    vori_data : array-like
        Array-like object containing information about the Voronoi partitions from the simulation.
    path-data : array-like
        Array-like object containing information about the agent coordinates from the simulation.


    Methods
    -------
    __init__(sim_params, env_params, heat_params)
        Instantiates a BasicSimulation object.
    obtain_simulation_data(path_to_folder=None)
        Gathers data for visualizing the simulation.
    create_simulation(pause_rate=0.1)
        Creates a visualization using matplotlib.
    """

    def __init__(self, sim_params, env_params, heat_params):
        """ Instantiates a BasicSimulation object.

        Parameters
        ----------
        sim_params : dict
            Dictionary containing the simulation parameters.
        env_params : dict
            Dictionary containing the environment parameters.
        heat_params : dict
            Dictionary containing the adversarial parameters.
        """
        self.sim_params = sim_params
        self.env_params = env_params
        self.heat_params = heat_params
        self.operating_env = shapely.geometry.Polygon(env_params["points"])
        self.num_frames = self.sim_params["num_frames"]
        self.cost_data = list()
        self.vori_data = list()
        self.path_data = np.zeros(
            (self.sim_params["num_frames"], self.sim_params["num_blue"], 2)
        )

    def obtain_simulation_data(self, path_to_folder=None):
        """ Gathers data for visualizing the simulation.

        Parameters
        ----------
        path_to_folder : str, optional
            Absolute path to the directory used to store the data, by default None.
        """

        # Obtain the bounds of the bounding box for the operating environment.
        min_x, min_y, max_x, max_y = self.operating_env.bounds

        # Initialize the agent positions.
        if self.sim_params["init_points"] is None:
            deploy_points = pyvoro.generate_random_points(
                self.operating_env,
                num_points=self.sim_params["num_blue"],
                min_dist=self.sim_params["min_dist"],
            )
        else:
            deploy_points = self.sim_params["init_points"]

        # Populate the path data with the initial points.
        self.path_data[0, 0 : self.sim_params["num_blue"], 0] = np.array(
            [deploy_points[i][0] for i in range(len(deploy_points))]
        )
        self.path_data[0, 0 : self.sim_params["num_blue"], 1] = np.array(
            [deploy_points[i][1] for i in range(len(deploy_points))]
        )

        # Collect data for the visualization.
        points = deploy_points.copy()
        for k in range(self.sim_params["num_frames"]):
            print("Frame {}".format(str(k)))

            # Compute the Voronoi tesselation.
            vorinfo, _ = pyvoro.find_polygon_centroids(self.operating_env, points)
            self.vori_data.append(vorinfo)

            new_pos = list()
            frame_cost = 0

            mu = np.array([self.heat_params["mu_xt"](k), self.heat_params["mu_yt"](k)])
            sigma = np.array(
                [self.heat_params["sigma_xt"](k), self.heat_params["sigma_xt"](k)]
            )

            # Insert finite difference mu and sigma here.

            # Compute the new positions for each agent (for each region).
            for region in vorinfo[0]:
                vertices = vorinfo[1][region]
                polygon_region = shapely.geometry.Polygon(vertices).intersection(
                    self.operating_env
                )

                # Print the coordinates for some other use...
                coords = polygon_region.exterior.xy
                coords = [[coords[0][i], coords[1][i]] for i in range(len(coords[0]))]
                print(coords)

                min_x, min_y, max_x, max_y = polygon_region.bounds

                # Compute the mass.
                mass_x, _ = scint.quad(
                    self.heat_params["joint_xt"],
                    min_x,
                    max_x,
                    args=(k, self.heat_params["mu_xt"], self.heat_params["sigma_xt"]),
                )
                mass_y, _ = scint.quad(
                    self.heat_params["joint_xt"],
                    min_y,
                    max_y,
                    args=(k, self.heat_params["mu_yt"], self.heat_params["sigma_yt"]),
                )

                # Compute the mass derivative.
                dmass_x, _ = scint.quad(
                    self.heat_params["djoint_xt"],
                    min_x,
                    max_x,
                    args=(
                        k,
                        k - 1,
                        self.heat_params["mu_xt"],
                        self.heat_params["sigma_xt"],
                        1 / self.num_frames,
                    ),
                )
                dmass_y, _ = scint.quad(
                    self.heat_params["djoint_xt"],
                    min_y,
                    max_y,
                    args=(
                        k,
                        k - 1,
                        self.heat_params["mu_yt"],
                        self.heat_params["sigma_yt"],
                        1 / self.num_frames,
                    ),
                )

                # Compute the center of mass.
                cent_x, _ = scint.quad(
                    self.heat_params["weighted_xt"],
                    min_x,
                    max_x,
                    args=(k, self.heat_params["mu_xt"], self.heat_params["sigma_xt"]),
                )
                cent_y, _ = scint.quad(
                    self.heat_params["weighted_xt"],
                    min_y,
                    max_y,
                    args=(k, self.heat_params["mu_yt"], self.heat_params["sigma_yt"]),
                )

                cent_x = cent_x / mass_x
                cent_y = cent_y / mass_y

                # Compute the derivative of the centers of mass.
                dcent_x, _ = scint.quad(
                    self.heat_params["dweighted_xt"],
                    min_x,
                    max_x,
                    args=(
                        k,
                        k - 1,
                        self.heat_params["mu_xt"],
                        self.heat_params["sigma_xt"],
                        1 / self.num_frames,
                    ),
                )
                dcent_y, _ = scint.quad(
                    self.heat_params["dweighted_xt"],
                    min_y,
                    max_y,
                    args=(
                        k,
                        k - 1,
                        self.heat_params["mu_yt"],
                        self.heat_params["sigma_yt"],
                        1 / self.num_frames,
                    ),
                )

                dcent_x = (dcent_x - mass_x * cent_x) / mass_x
                dcent_y = (dcent_y - mass_y * cent_y) / mass_y

                # Compute the centers of mass assuming that the density is uniform.
                cent_unif = list(polygon_region.centroid.coords)

                # Compute the closest point information and then use that information to compute the new positions.
                closest_point, _ = pyvoro.compute_closest_point(polygon_region, points)

                # Compute the gradient direction and then use that information to compute the new positions.
                dclosest_point = np.zeros((len(closest_point)))
                kappa_x = (dcent_x / (closest_point[0] - cent_x)) - (dmass_x / mass_x)
                kappa_y = (dcent_y / (closest_point[1] - cent_y)) - (dmass_y / mass_y)
                kappa = np.linalg.norm(np.array([kappa_x, kappa_y]))
                dclosest_point[0] = dcent_x - (kappa + (dmass_x / mass_x)) * (
                    closest_point[0] - cent_x
                )
                dclosest_point[1] = dcent_y - (kappa + (dmass_y / mass_y)) * (
                    closest_point[1] - cent_y
                )
                new_x = closest_point[0] + self.sim_params["alpha"] * dclosest_point[
                    0
                ] / np.linalg.norm(dclosest_point[0])
                new_y = closest_point[1] + self.sim_params["alpha"] * dclosest_point[
                    1
                ] / np.linalg.norm(dclosest_point[1])

                new_pos.append([new_x, new_y])

                # Compute the contribution to the total cost.
                total_cost_x, _ = scint.quad(
                    self.heat_params["cost_xt"],
                    min_x,
                    max_x,
                    args=(
                        k,
                        closest_point[0],
                        self.heat_params["mu_xt"],
                        self.heat_params["sigma_xt"],
                    ),
                )
                total_cost_y, _ = scint.quad(
                    self.heat_params["cost_xt"],
                    min_y,
                    max_y,
                    args=(
                        k,
                        closest_point[1],
                        self.heat_params["mu_yt"],
                        self.heat_params["sigma_yt"],
                    ),
                )

                frame_cost += total_cost_x + total_cost_y

            self.cost_data.append(frame_cost)

            # Update the new positions.
            points = new_pos.copy()

            # Append the data to path_data.
            self.path_data[k, 0 : self.sim_params["num_blue"], 0] = np.array(
                [new_pos[i][0] for i in range(len(new_pos))]
            )
            self.path_data[k, 0 : self.sim_params["num_blue"], 1] = np.array(
                [new_pos[i][1] for i in range(len(new_pos))]
            )

        # Optional, save the data locally.
        if path_to_folder is not None:
            np.save(path_to_folder + "path.npy", self.path_data)
            np.save(path_to_folder + "cost.npy", self.cost_data)
            np.save(path_to_folder + "vori.npy", self.vori_data)

    def create_simulation(self, pause_rate=0.1):
        """ Creates a visualization using matplotlib.

        Parameters
        ----------
        pause_rate : float, optional
            The pause rate passed onto plt.pause(), by default 0.1.
        """
        fig = plt.figure(
            constrained_layout=True,
            figsize=((1680 / 1) / 192, (1000 / 1) / 192),
            dpi=192,
            frameon=True,
        )

        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[:, 0])
        vx = fig.add_subplot(gs[:, 1])

        for spine in ax.spines.values():
            spine.set_visible(False)

        for spine in vx.spines.values():
            spine.set_visible(False)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        vx.set_aspect("equal")
        vx.set_xticks([])
        vx.set_yticks([])

        # Create the transparent color map for plotting the RED agent.
        transparent_cmap = pyvoro.create_transparent_cmap(plt.cm.Reds)

        # Plot the main environment for visualization.
        ext_x, ext_y = self.operating_env.exterior.xy
        ax.plot(ext_x, ext_y, color="black", linewidth=0.5)
        vx.plot(ext_x, ext_y, color="black", linewidth=0.5)

        # Plot the initial point and get the maximum rectangular range for plotting.
        # initp, = ax.plot(self.path_data[0, 0:self.sim_params["num_blue"], 0], self.path_data[0, 0:self.sim_params["num_blue"], 1], "g*", markersize=1)
        min_x, max_x = ax.get_xlim()
        min_y, max_y = ax.get_ylim()

        # Create the grids for visualization.
        x_grid, y_grid = np.mgrid[min_x:max_x, min_y:max_y]
        time_range = np.linspace(0, 1, self.num_frames)

        for idx in range(1, self.path_data.shape[0]):
            gauss_val = self.heat_params["joint_xyt"](
                x_grid,
                y_grid,
                idx,
                self.heat_params["mu_xt"],
                self.heat_params["mu_yt"],
                self.heat_params["sigma_xt"],
                self.heat_params["sigma_yt"],
            ).ravel()
            cbf = ax.contourf(
                x_grid,
                y_grid,
                gauss_val.reshape(x_grid.shape[0], y_grid.shape[1]),
                15,
                cmap=transparent_cmap,
            )

            vlines = list()
            (pts,) = ax.plot(
                self.path_data[idx, 0 : self.sim_params["num_blue"], 0],
                self.path_data[idx, 0 : self.sim_params["num_blue"], 1],
                "bo",
                markersize=2,
                alpha=0.6,
            )

            if idx > 0:
                (ptsvax,) = vx.plot(
                    self.path_data[idx - 1, 0 : self.sim_params["num_blue"], 0],
                    self.path_data[idx - 1, 0 : self.sim_params["num_blue"], 1],
                    "bo",
                    markersize=2,
                    alpha=0.5,
                )

            for region in self.vori_data[idx][0]:
                polygon_points = self.vori_data[idx][1][region]
                poly = Polygon(polygon_points).intersection(self.operating_env)

                (vline,) = vx.plot(*poly.exterior.xy, color="black", linewidth=0.5)
                vlines.append(vline)

            if idx is not self.num_frames - 1:
                plt.show(block=False)
                plt.pause(pause_rate)
            else:
                plt.show(block=True)

            pts.remove()
            ptsvax.remove()

            [p.remove() for _, p in enumerate(vlines)]

            for coll in cbf.collections:
                coll.remove()

            del pts
            del vline
            del ptsvax
