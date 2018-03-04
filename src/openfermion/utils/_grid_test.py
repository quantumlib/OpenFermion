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

from __future__ import absolute_import

import unittest

from openfermion.utils import Grid
from openfermion.utils._testing_utils import EqualsTester


class GridTest(unittest.TestCase):

    def test_preconditions(self):
        nan = float('nan')

        # No exception
        _ = Grid(dimensions=0, length=0, scale=1.0)
        _ = Grid(dimensions=1, length=1, scale=1.0)
        _ = Grid(dimensions=2, length=3, scale=0.01)
        _ = Grid(dimensions=234, length=345, scale=456.0)

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

    def test_properties(self):
        g = Grid(dimensions=2, length=3, scale=5.0)
        self.assertEqual(g.num_points(), 9)
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

    def test_repr(self):
        self.assertEqual(
            repr(Grid(dimensions=5, length=3, scale=0.5)),
            'Grid(dimensions=5, length=3, scale=0.5)')

    def test_str(self):
        self.assertEqual(
            str(Grid(dimensions=5, length=4, scale=0.5)),
            '[-0.25, -0.125, 0.0, 0.125]**5')

    def test_equality(self):
        eq = EqualsTester(self)
        eq.make_equality_pair(lambda: Grid(dimensions=5, length=3, scale=0.5))
        eq.add_equality_group(Grid(dimensions=4, length=3, scale=0.5))
        eq.add_equality_group(Grid(dimensions=5, length=4, scale=0.5))
        eq.add_equality_group(Grid(dimensions=5, length=3, scale=0.25))
