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

"""Tests for indexing.py."""

import unittest
from openfermion.utils.indexing import up_index, down_index, up_then_down


class IndexingTest(unittest.TestCase):

    def test_up_index(self):
        self.assertEqual(up_index(0), 0)
        self.assertEqual(up_index(1), 2)
        self.assertEqual(up_index(2), 4)
        self.assertEqual(up_index(5), 10)

    def test_down_index(self):
        self.assertEqual(down_index(0), 1)
        self.assertEqual(down_index(1), 3)
        self.assertEqual(down_index(2), 5)
        self.assertEqual(down_index(5), 11)

    def test_up_then_down(self):
        # Test even num_modes (6 modes)
        # Even indices: 0->0, 2->1, 4->2
        # Odd indices: 1->3, 3->4, 5->5
        self.assertEqual(up_then_down(0, 6), 0)
        self.assertEqual(up_then_down(2, 6), 1)
        self.assertEqual(up_then_down(4, 6), 2)
        self.assertEqual(up_then_down(1, 6), 3)
        self.assertEqual(up_then_down(3, 6), 4)
        self.assertEqual(up_then_down(5, 6), 5)

        # Test odd num_modes (5 modes)
        # halfway = ceil(2.5) = 3
        # Even indices: 0->0, 2->1, 4->2
        # Odd indices: 1->3, 3->4
        self.assertEqual(up_then_down(0, 5), 0)
        self.assertEqual(up_then_down(2, 5), 1)
        self.assertEqual(up_then_down(4, 5), 2)
        self.assertEqual(up_then_down(1, 5), 3)
        self.assertEqual(up_then_down(3, 5), 4)
