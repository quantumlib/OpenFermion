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
"""tests for contextuality.py"""
import unittest
from openfermion.ops.operators import QubitOperator, FermionOperator

from openfermion.functionals.contextuality import is_contextual


class IsContextualTest(unittest.TestCase):

    def setUp(self):
        self.x1 = QubitOperator('X1', 1.)
        self.x2 = QubitOperator('X2', 1.)
        self.x3 = QubitOperator('X3', 1.)
        self.x4 = QubitOperator('X4', 1.)
        self.z1 = QubitOperator('Z1', 1.)
        self.z2 = QubitOperator('Z2', 1.)
        self.x1x2 = QubitOperator('X1 X2', 1.)
        self.y1y2 = QubitOperator('Y1 Y2', 1.)

    def test_raises_exception(self):
        with self.assertRaises(TypeError):
            is_contextual(FermionOperator())

    def test_empty_qubit_operator(self):
        self.assertFalse(is_contextual(QubitOperator()))

    def test_noncontextual_two_qubit_hamiltonians(self):
        self.assertFalse(is_contextual(self.x1 + self.x2))
        self.assertFalse(is_contextual(self.x1 + self.x2 + self.z2))
        self.assertFalse(is_contextual(self.x1 + self.x2 + self.y1y2))

    def test_contextual_two_qubit_hamiltonians(self):
        self.assertTrue(is_contextual(self.x1 + self.x2 + self.z1 + self.z2))
        self.assertTrue(is_contextual(self.x1 + self.x1x2 + self.z1 + self.z2))
        self.assertTrue(is_contextual(self.x1 + self.y1y2 + self.z1 + self.z2))

    def test_contextual_hamiltonians_with_extra_terms(self):
        self.assertTrue(
            is_contextual(self.x1 + self.x2 + self.z1 + self.z2 + self.x3 +
                          self.x4))
        self.assertTrue(
            is_contextual(self.x1 + self.x1x2 + self.z1 + self.z2 + self.x3 +
                          self.x4))
        self.assertTrue(
            is_contextual(self.x1 + self.y1y2 + self.z1 + self.z2 + self.x3 +
                          self.x4))

    def test_commuting_hamiltonian(self):
        self.assertFalse(is_contextual(self.x1 + self.x2 + self.x3 + self.x4))