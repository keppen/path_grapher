# import minimizer
# import plotter
# import flooder
from grids import Grid
import unittest


class grids_tester(unittest.TestCase):
    def setUp(self):
        self.linspaces = ((0, 10, 5), (0, 10, 10), (0, 10, 15))
        self.grid = Grid(self.linspaces)

    @property
    def msg(self):
        return f"\nChecked:\t{self.checked}\nResult:  \t{self.result}"

    def test_shape(self):
        self.checked = self.grid.shape
        self.result = (5, 10, 15)
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_ranges(self):
        self.checked = self.grid.ranges
        self.result = ((0, 10), (0, 10), (0, 10))
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_dim(self):
        self.checked = self.grid.dim
        self.result = 3
        self.assertEqual(self.checked, self.result, self.msg)

    def test_nnodes(self):
        self.checked = self.grid.nnodes
        self.result = 5 * 10 * 15
        self.assertEqual(self.checked, self.result, self.msg)

    def test_shape_offsets(self):
        self.checked = self.grid.shape_offsets
        self.result = (150, 15, 1)
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_axis_arrays(self):
        self.checked = self.grid.axis_arrays
        self.result = (5, 1, 1)
        self.assertSequenceEqual(
            tuple(self.checked[0].shape), self.result, self.msg)
        self.result = (1, 10, 1)
        self.assertSequenceEqual(
            tuple(self.checked[1].shape), self.result, self.msg)
        self.result = (1, 1, 15)
        self.assertSequenceEqual(
            tuple(self.checked[2].shape), self.result, self.msg)

    def test_from_nodes(self):
        nnodes = 5 * 10 * 15
        ranges = ((0, 10), (0, 10), (0, 10))
        weights = (5, 10, 15)
        from_nodes = Grid.from_nnodes(nnodes, ranges, weights)
        self.checked = from_nodes.linespaces
        self.result = self.linspaces
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_from_ranges(self):
        ranges = ((0, 10), (0, 10), (0, 10))
        resolution = (2, 1, 2 / 3)
        from_nodes = Grid.from_ranges(ranges, resolution)
        self.checked = from_nodes.linespaces
        self.result = self.linspaces
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_cart_to_grid(self):
        point = (1.0, 2.0, 3.0)
        self.checked = self.grid.cart_to_grid(point)
        self.result = (0, 2, 4)
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_grid_to_idx(self):
        point = (0, 2, 4)
        self.checked = self.grid.grid_to_idx(point)
        self.result = 34
        self.assertEqual(self.checked, self.result, self.msg)

    def test_idx_to_grid(self):
        point = 34
        self.checked = self.grid.idx_to_grid(point)
        self.result = (0, 2, 4)
        self.assertSequenceEqual(self.checked, self.result, self.msg)

    def test_grid_to_cart(self):
        point = (0, 2, 4)
        self.checked = self.grid.grid_to_cart(point)
        self.checked = [round(_, 2) for _ in self.checked]
        self.result = (0.0, 2.22, 2.86)
        self.assertSequenceEqual(self.checked, self.result, self.msg)


if __name__ == "__main__":
    unittest.main()
