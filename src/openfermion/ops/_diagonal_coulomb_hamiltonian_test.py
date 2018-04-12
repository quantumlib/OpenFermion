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
from __future__ import division

import unittest

from openfermion.transforms import get_fermion_operator
from openfermion.utils._testing_utils import random_diagonal_coulomb_hamiltonian

class DiagonalCoulombHamiltonianTest(unittest.TestCase):

    def test_multiply(self):
        n_qubits = 5
        op1 = random_diagonal_coulomb_hamiltonian(n_qubits)
        op2 = op1 * 1.5
        op3 = 1.5 * op1
        self.assertEqual(get_fermion_operator(op1) * 1.5,
                         get_fermion_operator(op2),
                         get_fermion_operator(op3))

    def test_divide(self):
        n_qubits = 5
        op1 = random_diagonal_coulomb_hamiltonian(n_qubits)
        op2 = op1 / 1.5
        self.assertEqual(get_fermion_operator(op1) / 1.5,
                         get_fermion_operator(op2))
