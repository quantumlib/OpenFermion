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
"""tests for operator_tapering.py"""

import unittest
from openfermion.ops.operators import FermionOperator, BosonOperator
from openfermion.transforms.repconversions.operator_tapering import (
    freeze_orbitals, prune_unused_indices)


class FreezeOrbitalsTest(unittest.TestCase):

    def test_freeze_orbitals_nonvanishing(self):
        op = FermionOperator(((1, 1), (1, 0), (0, 1), (2, 0)))
        op_frozen = freeze_orbitals(op, [1])
        expected = FermionOperator(((0, 1), (1, 0)), -1)
        self.assertEqual(op_frozen, expected)

    def test_freeze_orbitals_vanishing(self):
        op = FermionOperator(((1, 1), (2, 0)))
        op_frozen = freeze_orbitals(op, [], [2])
        self.assertEqual(len(op_frozen.terms), 0)


class PruneUnusedIndicesTest(unittest.TestCase):

    def test_prune(self):
        for LadderOp in (FermionOperator, BosonOperator):
            op = LadderOp(((1, 1), (8, 1), (3, 0)), 0.5)
            op = prune_unused_indices(op)
            expected = LadderOp(((0, 1), (2, 1), (1, 0)), 0.5)
            self.assertTrue(expected == op)