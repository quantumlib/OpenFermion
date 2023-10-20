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

from openfermion.transforms.opconversions import get_fermion_operator
from openfermion.ops.representations import DiagonalCoulombHamiltonian
from openfermion.testing.testing_utils import random_diagonal_coulomb_hamiltonian


class DiagonalCoulombHamiltonianTest(unittest.TestCase):
    def test_init(self):
        one_body = numpy.array([[2.0, 3.0], [3.0, 5.0]])
        two_body = numpy.array([[7.0, 11.0], [11.0, 13.0]])
        constant = 17.0
        op = DiagonalCoulombHamiltonian(one_body, two_body, constant)
        self.assertTrue(numpy.allclose(op.one_body, numpy.array([[9.0, 3.0], [3.0, 18.0]])))
        self.assertTrue(numpy.allclose(op.two_body, numpy.array([[0.0, 11.0], [11.0, 0.0]])))
        self.assertAlmostEqual(op.constant, 17.0)

    def test_multiply(self):
        n_qubits = 5
        op1 = random_diagonal_coulomb_hamiltonian(n_qubits)
        op2 = op1 * 1.5
        op3 = 1.5 * op1
        self.assertEqual(
            get_fermion_operator(op1) * 1.5, get_fermion_operator(op2), get_fermion_operator(op3)
        )

    def test_divide(self):
        n_qubits = 5
        op1 = random_diagonal_coulomb_hamiltonian(n_qubits)
        op2 = op1 / 1.5
        self.assertEqual(get_fermion_operator(op1) / 1.5, get_fermion_operator(op2))

    def test_exceptions(self):
        mat1 = numpy.array([[2.0, 3.0], [3.0, 5.0]])
        mat2 = numpy.array([[2.0, 3.0 + 1.0j], [3 - 1.0j, 5.0]])
        mat3 = numpy.array([[2.0, 3.0], [4.0, 5.0]])
        with self.assertRaises(ValueError):
            _ = DiagonalCoulombHamiltonian(mat1, mat2)
        with self.assertRaises(ValueError):
            _ = DiagonalCoulombHamiltonian(mat2, mat3)
        with self.assertRaises(ValueError):
            _ = DiagonalCoulombHamiltonian(mat3, mat1)
