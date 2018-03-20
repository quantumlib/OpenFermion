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

"""Tests for _conversion.py."""
from __future__ import absolute_import

import copy
import numpy
import unittest

from openfermion.ops import (FermionOperator, InteractionOperator,
                             normal_ordered, QubitOperator)
from openfermion.ops._interaction_operator import InteractionOperatorError
from openfermion.ops._quadratic_hamiltonian import QuadraticHamiltonianError
from openfermion.transforms import *
from openfermion.utils import *


class GetInteractionOperatorTest(unittest.TestCase):

    def test_get_molecular_operator(self):
        coefficient = 3.
        operators = ((2, 1), (3, 0), (0, 0), (3, 1))
        op = FermionOperator(operators, coefficient)

        molecular_operator = get_interaction_operator(op)
        fermion_operator = get_fermion_operator(molecular_operator)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(normal_ordered(op).isclose(fermion_operator))

    def test_get_interaction_operator_bad_input(self):
        with self.assertRaises(TypeError):
            get_interaction_operator('3')

    def test_get_interaction_operator_too_few_qubits(self):
        with self.assertRaises(ValueError):
            get_interaction_operator(FermionOperator('3^ 2^ 1 0'), 3)

    def test_get_interaction_operator_bad_1body_term(self):
        with self.assertRaises(InteractionOperatorError):
            get_interaction_operator(FermionOperator('1^ 0^'))

    def test_get_interaction_operator_bad_2body_term(self):
        with self.assertRaises(InteractionOperatorError):
            get_interaction_operator(FermionOperator('3^ 2 1 0'))

    def test_get_interaction_operator_nonmolecular_term(self):
        with self.assertRaises(InteractionOperatorError):
            get_interaction_operator(FermionOperator('3^ 2 1'))

    def test_get_molecular_data(self):
        """Test conversion to MolecularData from InteractionOperator"""


class GetQuadraticHamiltonianTest(unittest.TestCase):
    def setUp(self):
        self.hermitian_op = FermionOperator((), 1.)
        self.hermitian_op += FermionOperator('1^ 1', 3.)
        self.hermitian_op += FermionOperator('1^ 2', 3. + 4.j)
        self.hermitian_op += FermionOperator('2^ 1', 3. - 4.j)
        self.hermitian_op += FermionOperator('3^ 4^', 2. + 5.j)
        self.hermitian_op += FermionOperator('4 3', 2. - 5.j)

        self.hermitian_op_pc = FermionOperator((), 1.)
        self.hermitian_op_pc += FermionOperator('1^ 1', 3.)
        self.hermitian_op_pc += FermionOperator('1^ 2', 3. + 4.j)
        self.hermitian_op_pc += FermionOperator('2^ 1', 3. - 4.j)
        self.hermitian_op_pc += FermionOperator('3^ 4', 2. + 5.j)
        self.hermitian_op_pc += FermionOperator('4^ 3', 2. - 5.j)

        self.hermitian_op_bad_term = FermionOperator('1^ 1 2', 3.)
        self.hermitian_op_bad_term += FermionOperator('2^ 1^ 1', 3.)

        self.not_hermitian_1 = FermionOperator('2^ 0^')
        self.not_hermitian_2 = FermionOperator('3^ 0^')
        self.not_hermitian_2 += FermionOperator('3 0', 3.)
        self.not_hermitian_3 = FermionOperator('2 0')
        self.not_hermitian_4 = FermionOperator('4 0')
        self.not_hermitian_4 += FermionOperator('4^ 0^', 3.)
        self.not_hermitian_5 = FermionOperator('2^ 3', 3.)
        self.not_hermitian_5 += FermionOperator('3^ 2', 2.)

    def test_get_quadratic_hamiltonian_hermitian(self):
        """Test properly formed quadratic Hamiltonians."""
        # Non-particle-number-conserving without chemical potential
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(
            normal_ordered(self.hermitian_op).isclose(fermion_operator))

        # Non-particle-number-conserving chemical potential
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op,
                                                 chemical_potential=3.)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(
            normal_ordered(self.hermitian_op).isclose(fermion_operator))

        # Particle-number-conserving
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op_pc)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(
            normal_ordered(self.hermitian_op_pc).isclose(fermion_operator))

    def test_get_quadratic_hamiltonian_hermitian_bad_term(self):
        """Test an operator with non-quadratic terms."""
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.hermitian_op_bad_term)

    def test_get_quadratic_hamiltonian_not_hermitian(self):
        """Test non-Hermitian operators."""
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.not_hermitian_1)
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.not_hermitian_2)
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.not_hermitian_3)
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.not_hermitian_4)
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.not_hermitian_5)

    def test_get_quadratic_hamiltonian_bad_input(self):
        """Test improper input."""
        with self.assertRaises(TypeError):
            get_quadratic_hamiltonian('3')

    def test_get_quadratic_hamiltonian_too_few_qubits(self):
        """Test asking for too few qubits."""
        with self.assertRaises(ValueError):
            get_quadratic_hamiltonian(FermionOperator('3^ 2^'), n_qubits=3)


class GetSparseOperatorQubitTest(unittest.TestCase):

    def test_sparse_matrix_Y(self):
        term = QubitOperator(((0, 'Y'),))
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [1j, -1j])
        self.assertEqual(list(sparse_operator.indices), [1, 0])
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_ZX(self):
        coefficient = 2.
        operators = ((0, 'Z'), (1, 'X'))
        term = QubitOperator(operators, coefficient)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [2., 2., -2., -2.])
        self.assertEqual(list(sparse_operator.indices), [1, 0, 3, 2])
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_ZIZ(self):
        operators = ((0, 'Z'), (2, 'Z'))
        term = QubitOperator(operators)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data),
                         [1, -1, 1, -1, -1, 1, -1, 1])
        self.assertEqual(list(sparse_operator.indices), list(range(8)))
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_combo(self):
        qop = (QubitOperator(((0, 'Y'), (1, 'X')), -0.1j) +
               QubitOperator(((0, 'X'), (1, 'Z')), 3. + 2.j))
        sparse_operator = get_sparse_operator(qop)

        self.assertEqual(list(sparse_operator.data),
                         [3 + 2j, 0.1, 0.1, -3 - 2j,
                          3 + 2j, -0.1, -0.1, -3 - 2j])
        self.assertEqual(list(sparse_operator.indices),
                         [2, 3, 2, 3, 0, 1, 0, 1])

    def test_sparse_matrix_zero_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator((), 0.0), 1)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (2, 2))

    def test_sparse_matrix_zero_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator((), 0.0), 5)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (32, 32))

    def test_sparse_matrix_identity_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(()), 1)
        self.assertEqual(list(sparse_operator.data), [1] * 2)
        self.assertEqual(sparse_operator.shape, (2, 2))

    def test_sparse_matrix_identity_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(()), 5)
        self.assertEqual(list(sparse_operator.data), [1] * 32)
        self.assertEqual(sparse_operator.shape, (32, 32))

    def test_sparse_matrix_linearity(self):
        identity = QubitOperator(())
        zzzz = QubitOperator(tuple((i, 'Z') for i in range(4)), 1.0)

        sparse1 = get_sparse_operator(identity + zzzz)
        sparse2 = get_sparse_operator(identity, 4) + get_sparse_operator(zzzz)

        self.assertEqual(list(sparse1.data), [2] * 8)
        self.assertEqual(list(sparse1.indices),
                         [0, 3, 5, 6, 9, 10, 12, 15])
        self.assertEqual(list(sparse2.data), [2] * 8)
        self.assertEqual(list(sparse2.indices),
                         [0, 3, 5, 6, 9, 10, 12, 15])


class GetSparseOperatorFermionTest(unittest.TestCase):

    def test_sparse_matrix_zero_n_qubit(self):
        sparse_operator = get_sparse_operator(FermionOperator.zero(), 4)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (16, 16))
