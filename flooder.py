from heapq import heappush as hpush, heappop as hpop
from decorator import Decorator
from grids import Potential
import numpy as np


class Flodder(Decorator):
    def __init__(self, pot_obj: Potential):
        Decorator.__init__(self, pot_obj)
        self._sorted_idx = np.argsort(self.pot_obj.potential_1D)
        self._ranks_idx = np.argsort(self._sorted_idx)

    @property
    def sorted_idx(self):
        return self._sorted_idx

    @property
    def ranks_idx(self):
        return self._ranks_idx

    def _init_flood(self, idx1):
        self.path_idx = np.zeros(self.pot_obj.nnodes, dtype=np.int64)
        self.color = np.zeros(self.pot_obj.nnodes, dtype=np.int64)
        self.heap = []

        self.path_idx[:] = -1
        self.color[:] = 0
        self.color[idx1] = 1
        hpush(self.heap, self.ranks_idx[idx1])

    def flood(self, start_idx: int):
        self._init_flood(start_idx)
        self.fill()

        for end_point in self.pot_obj.min_list:
            end_idx = end_point.idx
            if end_idx == start_idx:
                continue
            path = self.follow_path(start_idx, end_idx)
            self.pot_obj._path_list.append(self.pot_obj.idx_to_Path(path))

    def follow_path(self, start: int, end: int):
        path = []
        path.append(start)
        while self.path_idx[path[-1]] != -1:
            path.append(self.path_idx[path[-1]])
        path.append(end)
        while self.path_idx[path[-1]] != -1:
            path.append(self.path_idx[path[-1]])
        return tuple(set(path))

    def fill(self):
        """
        How to make it versitale.
        It may stop when two points have met but also make it possible to
        flood all the nodes.
        """
        while True:
            if len(self.heap) == 0:
                print("The heap is empty.")

            current_node = self.sorted_idx[hpop(self.heap)]
            current_color = self.color[current_node]

            for neighbor_node in self.pot_obj.neighbors_from_idx(current_node):
                neighbor_color = self.color[neighbor_node]

                if self.color.all():
                    print("All nodes has been visited.")
                    return None

                if neighbor_color == current_color:
                    continue

                self.path_idx[neighbor_node] = current_node
                self.color[neighbor_node] = current_color
                hpush(self.heap, self.ranks_idx[neighbor_node])

        return None

    def operate(self):
        return self.pot_obj
