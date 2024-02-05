from abc import ABC, abstractmethod
from collections_objects import Graph
from itertools import product, combinations


class Interface(ABC):
    _min_list: list = []
    _path_list: list = []
    _graph = Graph()

    @property
    def min_list(self):
        return self._min_list

    @property
    def path_list(self):
        return self._path_list

    @property
    def graph(self):
        return self._graph

    @abstractmethod
    def operate(self, *args, **kargs):
        ...

    @classmethod
    @abstractmethod
    def from_file(cls, *args, **kargs):
        ...

    @classmethod
    @abstractmethod
    def from_cube(cls, *args, **kargs):
        ...

    @classmethod
    @abstractmethod
    def from_npz(cls, *args, **kargs):
        ...

    def __init_redundancy(self):
        self.redundant = []
        self.npaths = len(self.path_list)
        self.path_idx = list(range(self.npaths))

    def __del_redundancy(self):
        del self.redundant
        del self.npaths
        del self.path_idx

    def __remove_composites(self):
        self.__init_redundancy()
        for i, j in product(self.path_idx, repeat=2):
            if i in self.redundant or j in self.redundant:
                continue

            print(f"Progress: {i}/{self.npaths}", end="\r")
            short_path = self.path_list[i]
            long_path = self.path_list[j]

            if len(short_path) > len(long_path):
                continue

            if short_path in long_path:
                if len(short_path) == len(long_path):
                    continue
                self.redundant.append(j)

    def __remove_duplicates(self):
        self.__init_redundancy()
        for i, j in combinations(self.path_idx, 2):
            if self.path_list[i] in self.path_list[j]:
                self.redundant.append((i, j))
        self.redundant = [i for i, j in self.redundant]

    def __pop_form_paths(self):
        self.redundant.sort(reverse=True)
        for idx in self.redundant:
            self._path_list.pop(idx)

    def remove_excess_paths(self):
        print("Removing composite paths...")
        self.__remove_composites()
        self.__pop_form_paths()

        print("Removing duplicate paths...")
        self.__remove_duplicates()
        self.__pop_form_paths()

        self.__del_redundancy()

    def init_graph(self):
        for min in self.min_list:
            self._graph.append(min)

        for path in self.path_list:
            ends = path.ends
            ends_idx = [self.graph.idx.index(i.idx) for i in ends]
            self._graph.add_edge(
                ends_idx[0], ends_idx[1], path.point_max_difference(ends[0])
            )
            self._graph.add_edge(
                ends_idx[1], ends_idx[0], path.point_max_difference(ends[1])
            )

    def __partition_function(self, microstates):
        import math

        temp = 300
        k_boltz = 1.380649e-23
        kcal = 4.1868

        partition_function = 0
        for microstate in microstates:
            partition_function += math.exp(-microstate / k_boltz / temp / kcal)

    def __change_prob(self, node_idx):
        ends_points = [self.graph[i] for i, _ in self.graph.edges[node_idx]]
        ...
