#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import unittest
import numpy

from openfermion.utils import Grid
from openfermion.utils._grid import OrbitalSpecificationError
from openfermion.utils._testing_utils import EqualsTester


class GridTest(unittest.TestCase):

    def test_orbital_id(self):

        # Test in 1D with spin.
        grid = Grid(dimensions=1, length=5, scale=1.0)
        input_coords = [0, 1, 2, 3, 4]
        tensor_factors_up = [1, 3, 5, 7, 9]
        tensor_factors_down = [0, 2, 4, 6, 8]

        test_output_up = [grid.orbital_id(i, 1) for i in input_coords]
        test_output_down = [grid.orbital_id(i, 0) for i in input_coords]

        self.assertEqual(test_output_up, tensor_factors_up)
        self.assertEqual(test_output_down, tensor_factors_down)

        with self.assertRaises(OrbitalSpecificationError):
            grid.orbital_id(6, 1)

        # Test in 2D without spin.
        grid = Grid(dimensions=2, length=3, scale=1.0)
        input_coords = [(0, 0), (0, 1), (1, 2)]
        tensor_factors = [0, 3, 7]
        test_output = [grid.orbital_id(i) for i in input_coords]
        self.assertEqual(test_output, tensor_factors)

    def test_position_vector(self):

        # Test in 1D.
        grid = Grid(dimensions=1, length=4, scale=4.)
        test_output = [grid.position_vector(i)[0]
                       for i in range(grid.length[0])]
        correct_output = [-2, -1, 0, 1]
        self.assertEqual(correct_output, test_output)

        # Test in 2D.
        grid = Grid(dimensions=2, length=3, scale=3.)
        test_input = []
        test_output = []
        for i in range(3):
            for j in range(3):
                test_input += [(i, j)]
                test_output += [grid.position_vector((i, j))]
        correct_output = numpy.array([[-1., -1.], [-1., 0.], [-1., 1.],
                                      [0., -1.], [0., 0.], [0., 1.],
                                      [1., -1.], [1., 0.], [1., 1.]])
        self.assertAlmostEqual(0., numpy.amax(test_output - correct_output))

    def test_momentum_vector(self):
        grid = Grid(dimensions=1, length=3, scale=2. * numpy.pi)
        test_output = [grid.momentum_vector(i)
                       for i in range(grid.length[0])]
        correct_output = [-1., 0, 1.]
        self.assertEqual(correct_output, test_output)

        grid = Grid(dimensions=1, length=2, scale=2. * numpy.pi)
        test_output = [grid.momentum_vector(i)
                       for i in range(grid.length[0])]
        correct_output = [-1., 0.]
        self.assertEqual(correct_output, test_output)

        grid = Grid(dimensions=1, length=11, scale=2. * numpy.pi)
        for i in range(grid.length[0]):
            self.assertAlmostEqual(
                -grid.momentum_vector(i),
                grid.momentum_vector(grid.length[0] - i - 1))

        # Test in 2D.
        grid = Grid(dimensions=2, length=3, scale=2. * numpy.pi)
        test_input = []
        test_output = []
        for i in range(3):
            for j in range(3):
                test_input += [(i, j)]
                test_output += [grid.momentum_vector((i, j))]
        correct_output = numpy.array([[-1, -1], [-1, 0], [-1, 1],
                                      [0, -1], [0, 0], [0, 1],
                                      [1, -1], [1, 0], [1, 1]])
        self.assertAlmostEqual(0., numpy.amax(test_output - correct_output))

    def test_grid_indices(self):
        g1 = Grid(dimensions=2, length=4, scale=1.0)
        self.assertListEqual(g1.grid_indices(0, spinless=True),
                             [0, 0])

        for elmt in g1.grid_indices(0, spinless=True):
            self.assertIsInstance(elmt, int)

        # Check that out-of-range qubit IDs raise correct errors.
        with self.assertRaises(OrbitalSpecificationError):
            _ = g1.grid_indices(-1, spinless=True)

        with self.assertRaises(OrbitalSpecificationError):
            _ = g1.grid_indices(16, spinless=True)

        with self.assertRaises(OrbitalSpecificationError):
            _ = g1.grid_indices(17, spinless=True)

        # Check that the spinful grid allows appropriate qubit IDs.
        self.assertListEqual(g1.grid_indices(31, spinless=False),
                             [3, 3])

        # Check correct ordering of the list output.
        self.assertListEqual(g1.grid_indices(17, spinless=False),
                             [0, 2])
        self.assertListEqual(g1.grid_indices(8, spinless=True),
                             [0, 2])

    def test_initialize(self):
        g1 = Grid(dimensions=3, length=3, scale=1.0)
        scale_matrix = numpy.diag([1.0] * 3)
        g2 = Grid(dimensions=3, length=(3, 3, 3), scale=scale_matrix)
        self.assertEqual(g1, g2)

    def test_no_errors_in_call(self):
        # No exception
        _ = Grid(dimensions=1, length=1, scale=1.0)
        _ = Grid(dimensions=2, length=3, scale=0.01)
        _ = Grid(dimensions=23, length=34, scale=45.0)

    def test_preconditions_raise_value_error(self):
        nan = float('nan')

        with self.assertRaises(ValueError):
            _ = Grid(dimensions=0, length=0, scale=1.0)
        with self.assertRaises(ValueError):
            _ = Grid(dimensions=1, length=1, scale=1)
        with self.assertRaises(ValueError):
            _ = Grid(dimensions=1, length=1, scale=0.0)
        with self.assertRaises(ValueError):
            _ = Grid(dimensions=1, length=1, scale=-1.0)
        with self.assertRaises(ValueError):
            _ = Grid(dimensions=1, length=1, scale=nan)

        with self.assertRaises(ValueError):
            _ = Grid(dimensions=1, length=-1, scale=1.0)
        with self.assertRaises(ValueError):
            _ = Grid(dimensions=-1, length=1, scale=1.0)

    def test_position_and_momentum_vector_orbital_specification_error(self):
        with self.assertRaises(OrbitalSpecificationError):
            g = Grid(dimensions=1, length=2, scale=1.0)
            _ = g.position_vector((10, ))
        with self.assertRaises(OrbitalSpecificationError):
            g = Grid(dimensions=1, length=2, scale=1.0)
            _ = g.momentum_vector((10, ))

    def test_properties(self):
        g = Grid(dimensions=2, length=3, scale=5.0)
        self.assertEqual(g.num_points, 9)
        self.assertEqual(g.volume_scale(), 25)
        self.assertEqual(list(g.all_points_indices()), [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ])

    def test_equality(self):
        eq = EqualsTester(self)
        eq.make_equality_pair(lambda: Grid(dimensions=5, length=3, scale=0.5))
        eq.add_equality_group(Grid(dimensions=4, length=3, scale=0.5))
        eq.add_equality_group(Grid(dimensions=5, length=4, scale=0.5))
        eq.add_equality_group(Grid(dimensions=5, length=3, scale=0.25))
