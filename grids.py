import numpy as np
from itertools import product
from interface import Interface
from collections_objects import Point, Path


class Grid:
    def __init__(self, linespaces):
        self._shape = tuple([_[2] for _ in linespaces])
        self._ranges = tuple([(_[0], _[1]) for _ in linespaces])
        self._dim = len(self.shape)
        self._shape_offsets = self._init_shape_offsets(self.shape)
        self._axis_arrays = self._init_axis_arrays(linespaces)
        self._neighbor_offsets = self._init_neighbor_offsets(self.dim)

    @property
    def shape(self):
        return self._shape

    @property
    def ranges(self):
        return self._ranges

    @property
    def dim(self):
        return self._dim

    @property
    def nnodes(self):
        nnodes = 1
        for number in self.shape:
            nnodes *= number
        return nnodes

    @property
    def shape_offsets(self):
        return self._shape_offsets

    @property
    def axis_arrays(self):
        return self._axis_arrays

    @property
    def neighbors_offsets(self):
        return self._neighbor_offsets

    @property
    def linespaces(self):
        """
        Grid may me initalized by classmethod.
        That way grid is not defined linespace explicitly.

        So that makes it possible to get grid.linespaces
        as if self.linespaces were defiend.

        @property defines a getter. It is decorator which
        produces __get__, __set__ etc. methods. So it is descriptor.
        """
        linespaces = []
        ranges = self.ranges
        for i, number in enumerate(self.shape):
            start, stop = ranges[i]
            linespaces.append((start, stop, number))
        return tuple(linespaces)

    @staticmethod
    def _init_shape_offsets(shape):
        offsets = []
        dim = len(shape)
        for i in range(dim):
            offset = 1
            for j in range(i + 1, dim):
                offset *= shape[j]
            offsets.append(offset)
        return tuple(offsets)

    @staticmethod
    def _init_axis_arrays(linespaces):
        axis_arrays = []
        dim = len(linespaces)
        base_shape = [1] * dim
        for i in range(dim):
            a_array = np.linspace(*(linespaces[i]))
            array_shape = base_shape[:]
            array_shape[i] = linespaces[i][2]
            axis_arrays.append(a_array.reshape(array_shape))
        return tuple(axis_arrays)

    @staticmethod
    def _init_neighbor_offsets(dim):
        s = (-1, 0, 1)
        neighbour_offsets = list(product(s, repeat=dim))
        to_pop = neighbour_offsets.index(tuple([0] * dim))
        neighbour_offsets.pop(to_pop)
        return np.array(neighbour_offsets)

    @classmethod
    def from_nnodes(cls, nnodes, ranges, weights=None):
        """
        Create grid object from the number of points.
        Weights may manipulate size of grid vectors.
        """
        dim = len(ranges)
        if not weights:
            weights = [1] * dim
        weights = np.array(weights)
        numbers = np.copy(weights)
        while nnodes > np.prod(numbers):
            numbers += weights

        linespaces = [[ranges[_][0], ranges[_][1], numbers[_]] for _ in range(dim)]
        return cls(linespaces)

    @classmethod
    def from_ranges(cls, ranges, resolutions):
        """
        Create grid object from the start and stop.
        Weights may manipulate size of grid vectors.
        """
        from math import floor

        linespaces = []

        for ranges, resolution in zip(ranges, resolutions):
            start, limit = ranges
            if start > limit:
                start, limit = limit, start
            number = abs(floor((limit - start) / float(resolution)))
            stop = start + number * resolution
            linespaces.append((start, stop, number))
        return cls(linespaces)

    def cart_to_grid(self, cart):
        grid = []
        for i in range(self.dim):
            axis_array = self.axis_arrays[i].ravel()
            grid.append(np.argmin(abs(axis_array - cart[i])))
        return tuple(grid)

    def grid_to_idx(self, grid):
        return sum([point * offset for point, offset in zip(grid, self.shape_offsets)])

    def idx_to_grid(self, idx):
        grid = []
        reminder = idx
        for offset in self.shape_offsets:
            quotient, reminder = divmod(reminder, offset)
            grid.append(quotient)
        return tuple(grid)

    def grid_to_cart(self, grid):
        return tuple(
            [float(self.axis_arrays[i].ravel()[j]) for i, j in enumerate(grid)]
        )

    def neighbors_from_grid(self, grid):
        neighbors = grid + self.neighbors_offsets
        return [
            self.grid_to_idx(n)
            for n in neighbors[
                np.all((self.shape > neighbors) & (neighbors >= 0), axis=1)
            ]
        ]

    def neighbors_from_idx(self, idx):
        return self.neighbors_from_grid(self.idx_to_grid(idx))


class Potential(Interface, Grid):
    def __init__(self, linespaces, potential):
        Grid.__init__(self, linespaces)
        self._potential_1D = potential.ravel()
        print("Created pot_obj".upper())

    @property
    def potential_1D(self):
        return self._potential_1D

    @property
    def potential(self):
        return self.potential_1D.reshape(self.shape)

    @property
    def surface(self):
        meshgrid = [_.ravel() for _ in self.meshgrid]
        meshgrid.append(self.potential_1D)
        return np.array(meshgrid).vstack()

    @property
    def meshgrid(self):
        return np.meshgrid(*self.axis_arrays, indexing="ij")

    @classmethod
    def from_file(cls, filepath: str):
        surface = np.loadtxt(filepath)
        return cls.from_surface(surface)

    @classmethod
    def from_cube(cls, filepath: str):
        from libread import read_cube

        volumetric_data, _ = read_cube(filepath)
        surface = list(zip(volumetric_data.keys(), volumetric_data.values()))
        return cls.from_surface(surface)

    @classmethod
    def from_potential(cls, potential, nnodes, ranges, weights):
        linespaces = Grid.from_nnodes(nnodes, ranges, weights)
        return cls(potential, linespaces)

    @classmethod
    def from_surface(cls, surface):
        linespace = []
        potential = surface[:, -1]
        for dim in range(surface.shape[-1] - 1):
            axis_array = np.unique(surface[:, dim])
            linespace.append([axis_array[0, axis_array[-1], axis_array.size]])

        cls(potential, linespace)

    @classmethod
    def from_npz(cls, filepath: str, name: str):
        surface = np.load(filepath + "/" + name)
        return cls.from_surface(surface)

    def idx_to_Point(self, idx: int):
        grid = self.idx_to_grid(idx)
        cart = self.grid_to_cart(grid)
        potential = self.potential_1D[idx]
        has_nan_neighbor = False
        for neigbor in self.neighbors_from_idx(idx):
            if np.isnan(self.potential_1D[neigbor]):
                has_nan_neighbor = True
        return Point(
            idx=idx,
            grid=grid,
            cart=cart,
            pot=potential,
            has_nan_neighbor=has_nan_neighbor,
        )

    def idx_to_Path(self, path: list):
        points = []
        for idx in path:
            points.append(self.idx_to_Point(idx))
        return Path(points)

    def save_potential(self, name):
        np.save(name, self.surface)

    def copy_potential(self):
        potential = self.potential_1D.reshape(self.shape)
        copy_pot_1D = self.__class__(self.linespaces, potential)
        return copy_pot_1D

    @staticmethod
    def is_smooth(potential):
        """Check if the potenetial is smooth"""
        ...

    @staticmethod
    def surface_smoothing(potential):
        pass

    @staticmethod
    def gradients(pot_obj):
        axis_arrays = [_.ravel() for _ in pot_obj.axis_arrays]
        return np.gradient(pot_obj.potential, *axis_arrays)

    def operate(self):
        msg = f"""shape:\t{self.shape}
ranges:\t{self.ranges}
dimention:\t{self.dim}
number of nodes:\t{self.nnodes}
"""
        return msg


if __name__ == "__main__":
    """
    [..., x3, x2, x1, energy]
    """
    n = 1
    if n == 0:
        linespaces = [[0, 10, 10], [0, 10, 10]]
        grid = Grid(linespaces)
        assert grid.shape == (10, 10)
        assert grid.ranges == ((0, 10), (0, 10))
        assert grid.dim == 2
        # print(grid.shape_offsets)
        assert grid.shape_offsets == (10, 1)
        assert grid.linespaces == ((0, 10, 10), (0, 10, 10))
        # print(grid.axis_arrays)
        assert grid.nnodes == 100
        # print(grid.neighbors_offsets)

        nnodes = 200
        ranges = [[0, 10], [0, 10]]
        weights = [1, 2]
        grid = Grid.from_nnodes(nnodes, ranges, weights)
        assert grid.shape == (10, 20)
        assert grid.ranges == ((0, 10), (0, 10))
        assert grid.dim == 2
        assert grid.shape_offsets == (20, 1)
        assert grid.linespaces == ((0, 10, 10), (0, 10, 20))
        assert grid.nnodes == 200

        assert grid.cart_to_grid((2, 2)) == (2, 4)
        assert grid.grid_to_idx((3, 3)) == 63
        assert grid.idx_to_grid((41)) == (2, 1)
        # print(
        #     grid.grid_to_cart((2, 1)),
        #     (grid.axis_arrays[0].ravel()[2], grid.axis_arrays[1].ravel()[1]),
        # )
        # print(grid.neighbors_from_grid((2, 1)))
        # print(grid.neighbors_from_idx(41))
    if n == 1:
        linespaces = [[0, 1, 3 + n] for n in range(3)]

        # def pot(x, y, z):
        #     return x**2 + y**2 - z**2

        print("linespaces = =", linespaces)
        mesh = np.meshgrid(*[np.linspace(*_) for _ in linespaces], indexing="ij")
        # print(mesh.shape)
        print(*mesh, sep="\nnext dim\n".upper())
        mesh_slice = np.array(mesh)

        dim_to_find = 0
        dim_slice = mesh_slice[dim_to_find].shape
        dim_slice = list(dim_slice)
        # print(dim_slice)
        dim_slice.pop(dim_to_find)

        bool_array1 = mesh_slice[dim_to_find] == 0
        print("bool array 1\n", bool_array1)
        # print(mesh_slice[1][bool_array].reshape(dim_slice), "\n")
        # print(mesh_slice[2][bool_array].reshape(dim_slice), "\n")
        # print(mesh_slice[3][bool_array].reshape(dim_slice), "\n")

        dim_to_find = 1
        dim_slice = mesh_slice[dim_to_find].shape
        dim_slice = list(dim_slice)
        # print(dim_slice)
        dim_slice.pop(dim_to_find)

        bool_array2 = mesh_slice[dim_to_find] == 0
        print("bool_array 2\n", bool_array2)
        print("\nRESULT\n")
        print(bool_array1.shape, bool_array2.shape)
        # print(np.logical_and(bool_array1, bool_array2))
        print(bool_array1 & bool_array2)
