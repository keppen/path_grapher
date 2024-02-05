from decorator import Decorator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Axes3D


class Plotter(Decorator):
    def __init__(self, pot_obj, to_freeze: list = [], values: list = []):
        Decorator.__init__(self, pot_obj)
        self._free_dim_idx = self._init_free_dim_idx(to_freeze)
        self._free_shape = tuple(
            self.pot_obj.shape[_] for _ in self.free_dim_idx)
        self._free_dim = len(self.free_shape)
        self._bool_array = self._init_bool_array(to_freeze, values)

    @property
    def free_dim(self):
        return self._free_dim

    @property
    def bool_array(self):
        return self._bool_array

    @property
    def meshslice_dim(self):
        return self.meshslice.shape

    @property
    def meshslice(self):
        if self.bool_array.all():
            return self.pot_obj.meshgrid

        meshslice = []
        for i in self.free_dim_idx:
            meshslice.append(
                self.pot_obj.meshgrid[i][self.bool_array].reshape(
                    self.free_shape)
            )
        return np.array(meshslice)

    @property
    def potential_slice(self):
        if self.bool_array.all():
            return self.pot_obj.potential

        return self.pot_obj.potential[self.bool_array].reshape(self.free_shape)

    @property
    def free_dim_idx(self):
        return self._free_dim_idx

    @property
    def free_shape(self):
        return self._free_shape

    def _init_free_dim_idx(self, to_freeze):
        return tuple([_ for _ in range(self.pot_obj.dim) if _ not in to_freeze])

    def _init_bool_array(self, to_freeze: list = None, values: list = None):
        """ """
        if not to_freeze:
            return np.ones(self.pot_obj.shape, dtype=bool)

        error = self.out_of_bounds(to_freeze, values)
        if error:
            raise ValueError(error)

        bool_arrays = []
        for freezed_dim_idx, value in zip(to_freeze, values):
            bool_array = self.pot_obj.meshgrid[freezed_dim_idx] == value
            bool_arrays.append(bool_array)

        result = np.ones_like(bool_arrays[0], dtype=bool)
        for bool_array in bool_arrays:
            np.logical_and(result, bool_array, out=result)

        return result

    def out_of_bounds(self, to_freeze: list = None, values: list = None):
        if len(to_freeze) != len(values):
            return f"Number of dimentions {to_freeze} are not equal to number of values {values} to freeze."

        if any([_ > self.pot_obj.dim for _ in to_freeze]):
            return "Too high dimention to freeze."

        if any(
            [
                j < self.pot_obj.ranges[i][0] or j > self.pot_obj.ranges[i][1]
                for i, j in zip(to_freeze, values)
            ]
        ):
            return "Value to freeze at is out of range."

        return False

    def suface_path_2D(self, paths: list = None, name: str = "surface") -> None:
        """
        Print 2D surface with path as a scatterplot.

        mesh - np.meshgrid() / np.mgrid[], 2-Dimmnetional
        potential - np.array, potential.shape == mesh.shape[0]
        path - list of tuples, len(tuples) == potential.shape[0]
        """

        if self.free_dim != 2:
            print("Not valid dimention. Should be equal to 2.")
            exit(1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x, y = [*self.meshslice]

        ax.plot_surface(x, y, self.potential_slice, cmap="viridis", alpha=0.2)

        if paths:
            for path in paths:
                X = [point.cart[0] for point in path]
                Y = [point.cart[1] for point in path]
                Z = [point.pot for point in path]

                ax.scatter(X, Y, Z, alpha=0.25)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # plt.show()

        plt.savefig(f"{name}_plot.png", dpi=500)
        # Close the plot
        plt.close()

    def suface_points_2D(self, points: list = None, name: str = "points") -> None:
        """
        Print 2D surface with path as a scatterplot.

        mesh - np.meshgrid() / np.mgrid[], 2-Dimmnetional
        potential - np.array, potential.shape == mesh.shape[0]
        path - list of tuples, len(tuples) == potential.shape[0]
        """

        if self.free_dim != 2:
            print("Not valid dimention. Should be equal to 2.")
            exit(1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x, y = [*self.meshslice]

        ax.plot_surface(x, y, self.potential_slice, cmap="viridis", alpha=0.2)

        if points:
            print(points)
            for i in range(len(points)):
                X = points[i].cart[0]
                Y = points[i].cart[1]
                Z = points[i].pot

                ax.scatter(X, Y, Z)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # plt.show()

        plt.savefig(f"{name}_plot.png", dpi=500)
        # Close the plot
        plt.close()

    def path_3D(self, mesh: np.array, path: list = None):
        """
        Plot a only path as a lineplot in 3D.
        Volumetrc data cannot be plotted by matplotlib.

        mesh - np.meshgrid() / np.mgrid[], 3-Dimmnetional
        path - list of tuples, len(tuples) == mesh.shape[0][0]
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x = [coord[0] for coord in path]
        y = [coord[1] for coord in path]
        z = [coord[2] for coord in path]

        ax.plot(x, y, z)

        # ax.scatter(x, y, z, c=values, cmap="viridis", norm=norm)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # plt.show()

        plt.savefig("path_plot" + ".png", dpi=1200)
        # close the plot
        plt.close()

    def linear_potential_1D(self, path, name: str = "surface"):
        """
        Plot a potential versus number reaction variable.
        """
        X = list(range(len(path)))
        Y = path.pot

        fig = plt.figure()
        ax = plt.axes()

        ax.plot(X, Y)

        plt.savefig(f"{name}-plot.png")
        plt.close()

    def operate(self):
        ...
