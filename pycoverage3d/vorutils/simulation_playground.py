"""
A playground for creating simulations.

"""

import os
import sys
import time

import shapely
import numpy as np
import scipy.spatial
import shapely.geometry
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pycoverage3d.vorutils import pyvoro3d
from pycoverage3d.vorutils import playground


class SimulationInstance:
    """
    A low-fidelity simulation class for testing the 3-D case.

    """

    def __init__(self, sim_params, env_params, heat_params):
        """
        Instantiates an instance of the SimulationInstance class.
        
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

        self.num_frames = self.sim_params["num_frames"]
        self.num_agents = self.sim_params["num_blue_agents"]
        self.operating_hull = scipy.spatial.ConvexHull(self.env_params["vertices"])

        self.projected_vertices = np.unique(self.env_params["vertices"][:, 0:2], axis=0)
        # Need to figure out some way to order the vertices above but OK.
        self.projected_vertices = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
        self.projected_env = shapely.geometry.Polygon(self.projected_vertices)

        self.path_data = np.zeros((self.num_frames, self.num_agents, 3))
        self.red_path = np.zeros((self.num_frames, 3))

    def obtain_simulation_data(self, path_to_folder=None):
        """
        Obtains data for visualizing the simulation.
        
        Parameters
        ----------
        path_to_folder : str, optional
            Absolute path to the directory used to store the data, by default None. If None, then the simulation is not saved locally.

        """
        # Initialize the agent positions.
        if self.sim_params["init_points"] is None:
            deploy_points = pyvoro3d.generate_random_points(
                self.operating_hull, num_points=self.num_agents, min_dist=0.3
            )
        else:
            deploy_points = self.sim_params["init_points"]

        # Populate the path data with the initial points.
        self.path_data[0, 0 : self.num_agents, 0] = deploy_points[:, 0]
        self.path_data[0, 0 : self.num_agents, 1] = deploy_points[:, 1]
        self.path_data[0, 0 : self.num_agents, 2] = deploy_points[:, 2]
        self.red_path[0, :] = np.array(
            [
                self.heat_params["mu_xt"](0),
                self.heat_params["mu_yt"](0),
                self.heat_params["mu_zt"](0),
            ]
        )

        # Collect data for the simulation.
        blue_pos = deploy_points
        for k in range(self.num_frames):
            print("Frame {}".format(str(k)))

            # Initial red position.
            red_pos = np.array(
                [
                    [
                        self.heat_params["mu_xt"](k),
                        self.heat_params["mu_yt"](k),
                        self.heat_params["mu_zt"](k),
                    ]
                ]
            )
            self.red_path[k, :] = np.array(
                [
                    self.heat_params["mu_xt"](k),
                    self.heat_params["mu_yt"](k),
                    self.heat_params["mu_zt"](k),
                ]
            )

            # Projection onto 2-D.
            proj_blue, proj_red, proj_height = playground.project_2d(
                blue_pos, red_pos, sig_dig=4
            )

            # Compute the 2-D Voronoi diagram using the projection.
            regions, vertices = playground.compute_projected_voronoi(
                self.projected_env, proj_blue, sig_dig=4
            )

            """
            # Plot the Voronoi diagrams to confirm that it is what we expect. Delete this code later.
            if k == 1:
                fig, ax = plt.subplots()
                ext_x, ext_y = self.projected_env.exterior.xy
                ax.plot(ext_x, ext_y, color="black", linewidth=0.5)
                ax.scatter(proj_blue[:, 0], proj_blue[:, 1])
                for region in regions:
                    poly_points = vertices[region]
                    poly = shapely.geometry.Polygon(poly_points).intersection(
                        self.projected_env
                    )

                    (vline,) = ax.plot(*poly.exterior.xy, color="black", linewidth=0.5)
                plt.show()
            """

            # Set up the mean and std matrices.
            mu = np.array(
                [
                    [
                        self.heat_params["mu_xt"](k),
                        self.heat_params["mu_yt"](k),
                        self.heat_params["mu_zt"](k),
                    ],
                    [
                        self.heat_params["mu_xt"](k - 1),
                        self.heat_params["mu_yt"](k - 1),
                        self.heat_params["mu_zt"](k - 1),
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        self.heat_params["sigma_xt"](k),
                        self.heat_params["sigma_yt"](k),
                        self.heat_params["sigma_zt"](k),
                    ],
                    [
                        self.heat_params["sigma_xt"](k - 1),
                        self.heat_params["sigma_yt"](k - 1),
                        self.heat_params["sigma_zt"](k - 1),
                    ],
                ]
            )

            # Iterate through the regions.
            new_pos = list()
            for region in regions:
                poly_vertices = vertices[region]
                polygon_region = shapely.geometry.Polygon(poly_vertices).intersection(
                    self.projected_env
                )

                (
                    proj_centroids,
                    closest_point,
                    idx,
                ) = playground.compute_centroids_2d_single_step(
                    polygon_region,
                    [regions, vertices],
                    proj_blue,
                    mu,
                    sigma,
                    step_size=1.0 / self.num_frames,
                    alpha=0.05,
                    scale=1.0,
                    sig_dig=4,
                )
                new_z = blue_pos[:, 2][idx] + 0.9 * proj_height[idx]
                new_pos.append([proj_centroids[0], proj_centroids[1], new_z])

            blue_pos = np.array(new_pos.copy())
            self.path_data[k, 0 : self.num_agents, 0] = np.array(
                [new_pos[i][0] for i in range(len(new_pos))]
            )
            self.path_data[k, 0 : self.num_agents, 1] = np.array(
                [new_pos[i][1] for i in range(len(new_pos))]
            )
            self.path_data[k, 0 : self.num_agents, 2] = np.array(
                [new_pos[i][2] for i in range(len(new_pos))]
            )

        if path_to_folder is not None:
            np.save(path_to_folder + "path.npy", self.path_data)
            np.save(path_to_folder + "red_path.npy", self.red_path)

    def create_simulation_movie(self, pause_rate=0.1, data=False):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_zlim([0, 5])
        for k in range(self.num_frames):
            # ax.scatter3D(
            #    self.path_data[k][:, 0],
            #    self.path_data[k][:, 1],
            #    self.path_data[k][:, 2],
            #    color="g",
            # )
            pts1 = ax.scatter3D(
                self.path_data[k][:, 0],
                self.path_data[k][:, 1],
                self.path_data[k][:, 2],
                color="b",
            )
            pts2 = ax.scatter3D(
                self.red_path[k][0],
                self.red_path[k][1],
                self.red_path[k][2],
                color="r",
            )
            plt.show(block=False)
            plt.pause(pause_rate)
            pts1.remove()
            pts2.remove()


def main():
    # Define the environment parameters.
    vertices = 5 * np.array(
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
    env_params = {"vertices": vertices}

    # Define the simulation parameters.
    kappa = 1
    alpha = 0.1
    num_frames = 100
    num_blue = 3
    sim_params = {
        "init_points": None,
        "num_frames": num_frames,
        "num_blue_agents": num_blue,
    }

    mu_a = [5, 5, 5]
    mu_b = [0, 0, 0]
    sigma_a = [1, 1, 1]
    sigma_b = [1, 1, 1]

    mu_xt = lambda t: mu_a[0] * (1 - t / num_frames) + mu_b[0] * (t / num_frames)
    mu_yt = lambda t: mu_a[1] * (1 - t / num_frames) + mu_b[1] * (t / num_frames)
    mu_zt = lambda t: mu_a[2] * (1 - t / num_frames) + mu_b[2] * (t / num_frames)

    sigma_xt = lambda t: sigma_a[0] * (1 - t / num_frames) + sigma_b[0] * (
        t / num_frames
    )
    sigma_yt = lambda t: sigma_a[1] * (1 - t / num_frames) + sigma_b[1] * (
        t / num_frames
    )
    sigma_zt = lambda t: sigma_a[2] * (1 - t / num_frames) + sigma_b[2] * (
        t / num_frames
    )

    heat_params = {
        "mu_xt": mu_xt,
        "mu_yt": mu_yt,
        "mu_zt": mu_zt,
        "sigma_xt": sigma_xt,
        "sigma_yt": sigma_yt,
        "sigma_zt": sigma_zt,
    }

    simulation = SimulationInstance(sim_params, env_params, heat_params)
    simulation.obtain_simulation_data(
        "/Users/administrator/Desktop/pycoverage/pycoverage3d/vorutils/"
    )
    simulation.create_simulation_movie(pause_rate=0.5)


if __name__ == "__main__":
    main()
