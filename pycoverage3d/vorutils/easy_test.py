import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def main():
    blue_path = np.load("./path.npy")
    red_path = np.load("./red_path.npy")

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

    for k in range(100):
        # ax.scatter3D(
        #    blue_path[k][:, 0], blue_path[k][:, 1], blue_path[k][:, 2], color="g"
        # )
        pts1 = ax.scatter3D(
            blue_path[k][:, 0], blue_path[k][:, 1], blue_path[k][:, 2], color="b"
        )
        pts2 = ax.scatter3D(red_path[k][0], red_path[k][1], red_path[k][2], color="r")

        plt.show(block=False)
        plt.pause(0.5)
        pts1.remove()
        pts2.remove()


if __name__ == "__main__":
    main()
