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
"""Tests for Fenwick tree module."""
from __future__ import absolute_import

import unittest

from openfermion.transforms._fenwick_tree import FenwickNode, FenwickTree


class FenwickTreeTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_fenwick_tree_structure(self):
        """A lookup test on 5-qubit fenwick tree.

        Test:    Verifies structure of the Fenwick Tree on 5 sites."""
        f = FenwickTree(5)
        self.assertEqual(f.root.children[0].index, 2)
        self.assertEqual(f.root.children[1].index, 3)
        self.assertEqual(f.root.children[0].children[0].index, 1)
        self.assertEqual(f.root.children[0].children[0].children[0].index, 0)

    def test_fenwick_tree_ancestors(self):
        """Ancestor test. Check validity of the get_update_set(j) method on 8
        qubit register. Note that root is the last node.

        Test:     Verifies integrity of ancestors of nodes within the tree."""
        f = FenwickTree(8)
        self.assertEqual(len(f.get_update_set(7)), 0)

        # Is the parent of the middle child the root?
        self.assertEqual(f.get_update_set(3)[0], f.root)

        # Are the ancestors chained correctly?
        self.assertEqual(f.get_update_set(0)[0], f.get_node(1))
        self.assertEqual(f.get_update_set(0)[1], f.get_node(3))

    def test_fenwick_tree_children(self):
        """Children test. Checks get_F(j) on 8 qubit register.

        Test:     Verifies integrity of child nodes of the root."""
        f = FenwickTree(8)
        self.assertEqual(f.get_node(7).children[0], f.get_node(3))
        self.assertEqual(f.get_node(7).children[1], f.get_node(5))
        self.assertEqual(f.get_node(7).children[2], f.get_node(6))
        self.assertEqual(len(f.get_children_set(7)), 3)

    def test_fenwick_tree_ancestor_children(self):
        """Ancestor children test. Checks get_remainder_set(j) on 8 qubit
        register.

        Tests:    Checks the example given in the paper."""
        f = FenwickTree(16)
        self.assertEqual(f.get_remainder_set(9)[0].index, 7)


if __name__ == '__main__':
    unittest.main()
