import numpy as np
import collections.abc
import itertools


class Point(object):
    attribute = {
        "cart": np.nan,
        "grid": np.nan,
        "idx": np.nan,
        "pot": np.nan,
        "has_nan_neighbor": False,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(Point.attribute)
        self.__dict__.update(kwargs)

    def __repr__(self):
        return repr(self.__dict__)


class Path(collections.abc.MutableSequence):
    __lenghts = [1.0, 2 ** (1 / 2)]

    def __init__(self, points: list):
        self.__nmin = len(points)
        self.__start_point, self.__end_point = self._init_ends(points)
        self.__points = self._init_path(points)

    @property
    def ends(self):
        return self.__start_point, self.__end_point

    def __getitem__(self, index):
        return self.__points[index]

    def __setitem__(self, index, value):
        self.__points[index] = value

    def __delitem__(self, index):
        del self.__points[index]

    def __len__(self):
        return len(self.__points)

    def insert(self, index, value):
        self.__points.insert(index, value)

    def __contains__(self, short_path):
        distance_matix = []
        for i in short_path:
            for j in self.__points:
                distance_matix.append(self._distance(i.grid, j.grid))
        distance_matix = np.array(distance_matix).reshape(
            len(short_path), len(self.__points)
        )
        distance_sum = np.sum(np.min(distance_matix, axis=1))
        if distance_sum < 0.5 * len(short_path):
            return True
        return False

    def _distance(self, grid1, grid2):
        return np.linalg.norm(np.array(grid1) - np.array(grid2))

    def _adjacency_matrix(self, points):
        adjacency_matrix = []
        for i, j in itertools.product(list(range(self.__nmin)), repeat=2):
            lenght = self._distance(points[i].grid, points[j].grid)
            lenghts_bool = False
            if lenght in self.__lenghts:
                lenghts_bool = True
            adjacency_matrix.append(lenghts_bool)
        return np.array(adjacency_matrix, dtype=bool).reshape(self.__nmin, self.__nmin)

    def _init_ends(self, points):
        ends = []
        for i, row in enumerate(self._adjacency_matrix(points)):
            if np.sum(row) == 1:
                ends.append(i)
        start_point, end_point = map(lambda i: points[i], ends)
        if np.linalg.norm(start_point.grid) > np.linalg.norm(end_point.grid):
            start_point, end_point = end_point, start_point
        return start_point, end_point

    def _init_path(self, points):
        adj_matrix = self._adjacency_matrix(points)

        point_idx = points.index(self.__start_point)

        sorted_points = [point_idx]
        while points[point_idx] != self.__end_point:
            point_idx = [i for i, j in enumerate(
                adj_matrix[point_idx]) if j == True]
            point_idx = [i for i in point_idx if i not in sorted_points][0]
            sorted_points.append(point_idx)
        return np.array(points)[sorted_points]

    def __getattr__(self, attribute_name):
        if attribute_name in Point.attribute:
            values = []
            for point in self.__points:
                values.append(getattr(point, attribute_name, None))
            return values
        else:
            raise AttributeError(attribute_name)

    def activation_energy(self, point: Point):
        return max(self.pot) - point.pot

    def __repr__(self):
        return repr(self.idx)


class Graph:
    def __init__(self, nodes: list = None):
        self.__nodes = tuple(nodes) if nodes else []
        self.__adj_list = [] * len(self.__nodes)

    @property
    def nodes(self):
        return self.__nodes

    @property
    def nodes_idx(self):
        return list(range(len(self.__nodes)))

    def __getitem__(self, index):
        return self.__nodes[index]

    def __setitem__(self, index, value):
        self.__nodes[index] = value

    def __delitem__(self, index):
        del self.__adj_list[index]
        self.__nodes[index] = Point()
        for idx in self.nodes_idx:
            if index not in self.__adj_list[idx]:
                continue
            del self.__adj_list[self.__adj_list[idx].index(index)]

    def __len__(self):
        return len(self.__nodes)

    def append(self, node: Point):
        self.__nodes.append(node)
        self.__adj_list.append([])

    def __getattr__(self, attribute_name):
        if attribute_name in Point.attribute:
            values = []
            for point in self.__nodes:
                values.append(getattr(point, attribute_name, None))
            return values
        else:
            raise AttributeError(attribute_name)

    def __repr__(self):
        msg = ""
        for idx in self.nodes_idx:
            msg += f"{idx} --> {self.__adj_list[idx]}\n"

        return msg

    def add_edge(self, origin_idx, destination_idx, weight):
        self.__adj_list[origin_idx].append([destination_idx, weight])

    def BFS(self, node, border=list()):
        """Breath First Search walker. A generator.
        A border is an atom not added to queue,
        it makes walker stop searching beyond it.

        Parameters:
            node (dict): an atom
            border: an list of atoms

        Yields:
            node_path (list): [atom, [list of atoms ids]] -> [node, path]
        """
        path = [node["id"]]
        node_path = [node, path]
        queue = [node_path]
        while queue:
            node, path = queue.pop(0)
            node["visited"] = True
            for neighbor in node["bonds"]:
                # neighbor = self._findbyID(neighbor)
                node_path = [neighbor, path + [neighbor["id"]]]
                if not neighbor["visited"]:
                    if neighbor["id"] in border:
                        # print('found border', neighbor)
                        neighbor["visited"] = True
                        yield node_path
                        continue
                    yield node_path

                    queue.append(node_path)

    def Dijkstra(self, start):
        """
        Finds the shortest paths from the start node to all other nodes in the graph.
        """
        queue = [(0, start, [])]
        distances = {node: float("infinity") for node in self.nodes_idx}
        paths = {node: [] for node in self.__nodes}
        distances[start] = 0

        while queue:
            current_distance, current_node, path = heapq.heappop(queue)
            path = path + [current_node]

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.__adj_list[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = path
                    heapq.heappush(queue, (distance, neighbor, path))

        return distances, paths

    # # Set up queue
    # queue = [(0, start)]
    # # Set all node distances (values) to infinite
    # distances = {node: float("infinity") for node in graph.nodes}
    # # Set start node distance to 0
    # distances[start] = 0
    #
    # while queue:
    #     current_distance, current_node = heapq.heappop(queue)
    #
    #     if current_distance > distances[current_node]:
    #         continue
    #
    #     for neighbor, weight in graph.edges[current_node].items():
    #         distance = current_distance + weight
    #
    #         if distance < distances[neighbor]:
    #             distances[neighbor] = distance
    #             heapq.heappush(queue, (distance, neighbor))
    #
    # return distances

    # def Bellman_Ford(self, start):
    #     """
    #     Finds the shortest paths from the start node to all other nodes in the graph.
    #     """
    #
    #     # Set all node distances (values) to infinite
    #     distances = {node: float("infinity") for node in self.nodes}
    #     # Set start node distance to 0
    #     distances[start] = 0
    #
    #     """
    #     It visitas all nodes, calculates distance from starting node.
    #     It does in in N -1 cycles.
    #     """
    #     for _ in range(len(self.nodes) - 1):
    #         for node in self.nodes:
    #             for neighbor, weight in self.edges[node].items():
    #                 new_distance = distances[node] + weight
    #                 # print(f"New distance form {new_distance}")
    #                 previous_distance = distances[neighbor]
    #                 # print(f"Old distance {previous_distance}")
    #                 if new_distance < previous_distance:  # or \
    #                     # distances[node] == float('infinity'):
    #                     distances[neighbor] = new_distance
    #
    #     # for node in graph.nodes:
    #     #     for neighbor, weight in graph.edges[node].items():
    #     #         new_distance = distances[node] + weight
    #     #         previous_distance = distances[neighbor]
    #     #         assert new_distance >= previous_distance, "There is negative cycle"
    #
    #     return distances
