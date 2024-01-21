from decorator import Decorator
from grids import Potential
import numpy as np


class Minimize:
    def __init__(self, nwalkers: int = None, start_points: tuple = None):
        """
        start_points in fromart
        - if idx
            (int, int, int, ...)
        - if coord
            ((int, int, int, ...), ...)
        - if cart
            ((float, float, float, ...)
        """
        print("Created minimizer".upper())
        self._nwalkers = nwalkers
        self._start_points = start_points

    @property
    def nwalkers(self):
        if self._start_points is not None:
            return len(self.start_points)
        elif self._start_points is None and self._nwalkers:
            return self._nwalkers
        else:
            return 1

    @property
    def start_points(self):
        if self._start_points is not None:
            return self._start_points
        return [np.random.randint(self.pot_obj.nnodes) for _ in range(self.nwalkers)]

    @classmethod
    def from_gradient(cls, pot_obj: Potential):
        min_bool = cls._init_min_bool(pot_obj)
        min_coords = []
        for i in range(pot_obj.dim):
            min_coords.append(pot_obj.meshgrid[i][min_bool])
        min_coords = tuple(zip(*min_coords))
        return cls(pot_obj, nwalkers=len(min_coords), start_points=min_coords)

    @staticmethod
    def _init_min_bool(pot_obj: Potential):
        dims = pot_obj.dim
        gradients = Potential.gradients(pot_obj)
        gradient_signs = np.sign(gradients)
        axis = tuple([_ for _ in range(dims)])
        sign_change = []

        for i in range(dims):
            roll_direction = [0] * dims
            roll_direction[i] = 1
            rolled_signs = np.roll(
                gradient_signs[i],
                tuple(roll_direction),
                axis=axis,
            )
            changes = rolled_signs - gradient_signs[i] != 0
            changes = gradient_signs[i] == 1 & changes
            sign_change.append(changes)

        min_bool = np.ones(pot_obj.shape, dtype=bool)
        for i in sign_change:
            np.logical_and(min_bool, i, out=min_bool)
        return min_bool

    def to_idx(self, start_point):
        if isinstance(start_point, int):
            return start_point
        if isinstance(start_point, tuple):
            if all([isinstance(_, float) for _ in start_point]):
                start_point = self.pot_obj.cart_to_grid(start_point)
            return self.pot_obj.grid_to_idx(start_point)

    def operate(self):
        start_points = self.start_points
        for i in range(self.nwalkers):
            min_idx = self.minimize(start_points[i])
            self.pot_obj._min_list.append(min_idx)

        self.pot_obj._min_list = set(self.pot_obj._min_list)
        self.pot_obj._min_list = [
            self.pot_obj.idx_to_Point(idx) for idx in self.pot_obj._min_list
        ]

        return self.pot_obj

    def minimize(self):
        ...


class Numerical_Search(Minimize, Decorator):
    def __init__(self, potential: Potential, nwalkers: int = None):
        Decorator.__init__(self, potential)
        Minimize.__init__(self, nwalkers)

    def minimize(self):
        ...


class Point_Search(Minimize, Decorator):
    def __init__(self, pot_obj: Potential, nwalkers: int = None, start_points=None):
        Decorator.__init__(self, pot_obj)
        Minimize.__init__(self, nwalkers, start_points)

    def minimize(self, start_point):
        idx = self.to_idx(start_point)
        return self._minimize(idx)

    def _minimize(self, idx):
        min_idx = idx
        min_pot = self.pot_obj.potential_1D[min_idx]

        # print(
        #     "New minimize",
        #     min_idx,
        #     self.pot_obj.idx_to_grid(idx),
        #     self.pot_obj.potential_1D[min_idx],
        #     sep="\t",
        # )

        found_minimum = False
        while not found_minimum:
            found_minimum = True
            for neighbor_idx in self.pot_obj.neighbors_from_idx(min_idx):
                # print(
                #     "New neighbor",
                #     neighbor_idx,
                #     self.pot_obj.idx_to_grid(neighbor_idx),
                #     self.pot_obj.potential_1D[neighbor_idx],
                #     sep="\t",
                # )

                neighbor_pot = self.pot_obj.potential_1D[neighbor_idx]
                if neighbor_pot < min_pot:
                    # print("lower potential point", neighbor_pot)

                    min_idx = neighbor_idx
                    min_pot = neighbor_pot
                    found_minimum = False
        return min_idx
