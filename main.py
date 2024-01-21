from grids import Potential
from minimizer import Point_Search
from plotter import Plotter
from flooder import Flodder
import numpy as np


class main:
    def __init__(self):
        pass

    """
    Choose potential surface represtatnion

    Grid -
    Graph - I do not know if it is nessesary, but for the
            sake of implementation, why not.
    """

    def _surface_representation(self):
        pass

    """
    Choose minimum finiding algorithm

    Numerical analysis
    Neighbor check
    """

    def _minimize_function(self):
        pass

    """
    Choose path finding algorithm

    Flood
    ???
    """

    def _minpath_function(self):
        pass


class client:
    def convert_idx_to_cart(self, obj, list_idx):
        to_grid = obj.idx_to_grid
        to_cart = obj.grid_to_cart

        list_cart = [to_cart(to_grid(_)) for _ in list_idx]
        # print(list_idx)
        # print(list_cart)
        pot = [obj._potential_1D[_] for _ in list_idx]
        return [[a, b, c] for [a, b], c in zip(list_cart, pot)]

    def client_1(self):
        def f(x, y):
            return np.sin(np.sqrt(x**2 + y**2)) + 1 / 10 * x + 1 / 10 * y

        # def f(x, y):
        #     return np.zeros(x.shape)
        linspaces = [[-10, 10, 50], [-10, 10, 50]]

        print("Linespaces: ", linspaces)
        mesh = np.meshgrid(*[np.linspace(*_)
                           for _ in linspaces], indexing="ij")
        pot_obj = Potential(linspaces, f(*mesh))
        print("Created pot_obj")

        print(pot_obj.operate())

        minimizer = Point_Search.from_gradient(
            pot_obj,
        )
        print("Created minimizer")
        minimizer = minimizer.operate()

        print(f"minima: {minimizer.min_list}")
        flooder = Flodder(minimizer)
        print(("Created flooder"))
        for min in flooder.pot_obj.min_list:
            print(f"minimum : {min}")
            flooder.flood(min)
        flooder.remove_excess_paths()
        path = flooder.pot_obj.path_list

        from collections_objects import Graph

        graph = Graph()
        for min in flooder.pot_obj.min_list:
            graph.append(flooder.pot_obj.idx_to_Point(min))

        for p in path:
            ps, pe = p.ends
            ps_idx = graph.idx.index(ps.idx)
            pe_idx = graph.idx.index(pe.idx)
            weight = p.activation_energy(ps)
            graph.add_edge(ps_idx, pe_idx, weight)
            ps_idx, pe_idx = pe_idx, ps_idx
            weight = p.activation_energy(pe)
            graph.add_edge(ps_idx, pe_idx, weight)
        print(graph.pot, pe.pot, ps.pot)

        print(graph)
        flooder_1 = flooder.operate()

        plotter = Plotter(flooder_1)
        print("Created plotter")

        plotter.suface_path_2D(
            path,
            name="refined",
        )
        print(flooder_1.min_list)
        test = []
        for p in path:
            test.extend(p.idx)
        print(
            f"all points in paths = {len(test)}\n duplicates removed = {len(set(test))}"
        )
        # for p in path:
        #     print(p.idx)
        plotter.suface_points_2D(
            flooder_1.min_list,
            name="minimia",
        )

        for i, p in enumerate(path):
            plotter.linear_potential_1D(p, name=f"{i}-path")
        # plotter.suface_path_2D(
        #     flooder.idx_to_Path(set(list_idx)),
        #     name="refined-set",
        # )
        # list_idx = []
        # for i, path in enumerate(flooder_1.path_list):
        #     list_idx.extend(path.idx)
        # print(
        #     f"n points: all points {len(list_idx)}, removed duplicates  {len(set(list_idx))}"
        # )
        # plotter.suface_path_2D(
        #     flooder.idx_to_Path(list_idx),
        #     name="path-all",
        # )
        # return pot_obj, minimizer, flooder, plotter

    def client_2(self):
        # def f(x, y):
        #     return np.sin(np.sqrt(x**2 + y**2)) + 1 / 10 * x + 1 / 10 * y

        # def f(x, y, z):
        # return np.sin(x) / 10 + np.cos(y) / 5 / np.cos(z) / 15

        def f(x, y):
            return np.sin(np.sqrt(x**2 + y**2)) + 1 / 10 * x + 1 / 10 * y

        linspaces = [
            [-15, 15, 50],
            [-15, 15, 50],
            # [-1, 1, 10],
            # [4, 4, 6],
        ]
        # linspaces = [[-1, 1, 21]]
        print("Linespaces: ", linspaces)

        mesh = np.meshgrid(*[np.linspace(*_)
                           for _ in linspaces], indexing="ij")
        pot_obj = Potential(linspaces, f(*mesh))
        print(pot_obj.operate())

        minimizer = Point_Search(
            pot_obj,
            nwalkers=1,
        )

        minimizer = minimizer.operate()
        meshgrid = minimizer.meshgrid
        potential = minimizer.potential

        # potential = np.array([_ for _ in range(60)]).reshape(potential.shape)

        print("raw axis arrays")
        print(minimizer.axis_arrays)
        print("\n\n")

        axis_arrays = [j.ravel() for i, j in enumerate(minimizer.axis_arrays)]

        print("potential and shape")
        print(potential)
        print(potential.shape)
        print("\n\n")

        print("modified arrays axis")
        print(axis_arrays)
        print("\n\n")

        d = np.array(np.gradient(potential, *axis_arrays))

        print("graidents")
        print(d)
        print("graidents shape")
        print(d.shape)
        print("\n\n")

        sign_d = np.sign(d)
        print("signs")
        print(sign_d)
        print("\n\n")

        rolled_0 = np.roll(sign_d[0], (1, 0), axis=(0, 1))
        rolled_1 = np.roll(sign_d[1], (0, 1), axis=(0, 1))
        # rolled_2 = np.roll(potential, (0, 0), axis=(0, 1))

        print("rolled")
        print(
            rolled_0,
            rolled_1,
            # rolled_2,
            sep="\n\nnext\n\n",
        )

        dx_0 = sign_d[0] - rolled_0 != 0
        dy_0 = sign_d[1] - rolled_1 != 0

        dx_0 = np.logical_and(dx_0, sign_d[0] == 1)
        dy_0 = np.logical_and(dy_0, sign_d[1] == 1)

        print("dx, dy")
        print(
            dx_0,
            dy_0,
            sep="\n\n next \n\n",
        )

        print("all conditions")
        mins = np.logical_and(dx_0, dy_0)
        print(mins)

        print("gradients values")
        d_x = d[0][mins]
        d_y = d[1][mins]
        print(d_x, d_y, sep="\n\n")

        z = potential[mins]
        x = meshgrid[0][mins]
        y = meshgrid[1][mins]
        print(x, y, z)
        points = list(zip(x, y, z))
        print(points)

        # plotter = Plotter(minimizer)
        # plotter.suface_2D(path=points, name="gradient_control")

        minimize_test = Point_Search.from_gradient(pot_obj)
        start_points = minimize_test.start_points
        to_grid = pot_obj.cart_to_grid
        to_idx = pot_obj.grid_to_idx
        start_points = [to_idx(to_grid(i)) for i in start_points]
        start_points = sorted(set(start_points))
        start_points = self.convert_idx_to_cart(minimize_test, start_points)

        print(f"start points:\n\n{start_points}")

        min_list = minimize_test.operate()._min_list
        min_list = self.convert_idx_to_cart(minimize_test, min_list)
        print(sorted(min_list))

        print(
            f"""start points:
              number of points {len(start_points)}"""
        )
        print(
            f"""min list:
              number of points {len(min_list)}"""
        )
        diff = [_ for _ in start_points if _ not in min_list]
        print(diff)

        # plotter = Plotter(minimizer)
        # plotter.suface_2D(path=start_points, name="gradient_start")
        # plotter.suface_2D(path=min_list, name="gradient_minimized")


if __name__ == "__main__":
    # print("Mes shape:", mesh[0].shape)
    # print(*mesh, sep="\nnext dim\n".upper())
    # print("Potential")
    # print(pot(*mesh))

    client_class = client()
    client_class.client_1()
    # client_class.client_2()
