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

import unittest

import numpy

from openfermion.hamiltonians import fermi_hubbard
from openfermion.ops import (BosonOperator,
                             DiagonalCoulombHamiltonian,
                             FermionOperator,
                             QuadOperator,
                             QubitOperator)
from openfermion.transforms import jordan_wigner
from openfermion.utils import is_hermitian, normal_ordered

from openfermion.ops._interaction_operator import InteractionOperatorError
from openfermion.ops._quadratic_hamiltonian import QuadraticHamiltonianError
from openfermion.utils._testing_utils import (
        random_hermitian_matrix,
        random_quadratic_hamiltonian)

from openfermion.transforms._conversion import (
        get_boson_operator,
        get_diagonal_coulomb_hamiltonian,
        get_fermion_operator,
        get_interaction_operator,
        get_quad_operator,
        get_quadratic_hamiltonian,
        get_sparse_operator)


class GetInteractionOperatorTest(unittest.TestCase):

    def test_get_molecular_operator(self):
        coefficient = 3.
        operators = ((2, 1), (3, 0), (0, 0), (3, 1))
        op = FermionOperator(operators, coefficient)

        molecular_operator = get_interaction_operator(op)
        fermion_operator = get_fermion_operator(molecular_operator)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(normal_ordered(op) == fermion_operator)

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
            normal_ordered(self.hermitian_op) == fermion_operator)

        # Non-particle-number-conserving chemical potential
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op,
                                                 chemical_potential=3.)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(
            normal_ordered(self.hermitian_op) == fermion_operator)

        # Particle-number-conserving
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op_pc)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(
            normal_ordered(self.hermitian_op_pc) == fermion_operator)

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

    def test_ignore_incompatible_terms(self):

        ferm_op = (FermionOperator('0^ 2') + FermionOperator('2^ 0') +
                   FermionOperator('1^ 0^ 2') + FermionOperator('1^ 0^ 2 1') +
                   FermionOperator('0^ 0 1^ 1') + FermionOperator('1^ 2^ 1 2'))
        converted_op = get_quadratic_hamiltonian(
                ferm_op,
                ignore_incompatible_terms=True)
        self.assertTrue(numpy.allclose(converted_op.hermitian_part,
                        numpy.array([[0, 0, 1],
                                     [0, 0, 0],
                                     [1, 0, 0]])))


class GetDiagonalCoulombHamiltonianTest(unittest.TestCase):

    def test_hubbard(self):
        x_dim = 4
        y_dim = 5
        tunneling = 2.
        coulomb = 3.
        chemical_potential = 7.
        magnetic_field = 11.
        periodic = False

        hubbard_model = fermi_hubbard(x_dim, y_dim, tunneling, coulomb,
                                      chemical_potential, magnetic_field,
                                      periodic)

        self.assertTrue(
                normal_ordered(hubbard_model) ==
                normal_ordered(
                    get_fermion_operator(
                        get_diagonal_coulomb_hamiltonian(hubbard_model))))

    def test_random_quadratic(self):
        n_qubits = 5
        quad_ham = random_quadratic_hamiltonian(n_qubits, True)
        ferm_op = get_fermion_operator(quad_ham)
        self.assertTrue(
                normal_ordered(ferm_op) ==
                normal_ordered(
                    get_fermion_operator(
                        get_diagonal_coulomb_hamiltonian(ferm_op))))

    def test_ignore_incompatible_terms(self):

        ferm_op = (FermionOperator('0^ 2') + FermionOperator('2^ 0') +
                   FermionOperator('1^ 0^ 2') + FermionOperator('1^ 0^ 2 1') +
                   FermionOperator('0^ 0 1^ 1') + FermionOperator('1^ 2^ 1 2'))
        converted_op = get_diagonal_coulomb_hamiltonian(
                ferm_op,
                ignore_incompatible_terms=True)
        self.assertTrue(numpy.allclose(converted_op.one_body,
                        numpy.array([[0, 0, 1],
                                     [0, 0, 0],
                                     [1, 0, 0]])))
        self.assertTrue(numpy.allclose(converted_op.two_body,
                        numpy.array([[0, 0.5, 0],
                                     [0.5, 0, -0.5],
                                     [0, -0.5, 0]])))

    def test_exceptions(self):
        op1 = QubitOperator()
        op2 = FermionOperator('0^ 3') + FermionOperator('3^ 0')
        op3 = FermionOperator('0^ 1^')
        op4 = FermionOperator('0^ 1^ 2^ 3')
        op5 = FermionOperator('0^ 3')
        op6 = FermionOperator('0^ 0 1^ 1', 1.j)
        with self.assertRaises(TypeError):
            _ = get_diagonal_coulomb_hamiltonian(op1)
        with self.assertRaises(ValueError):
            _ = get_diagonal_coulomb_hamiltonian(op2, n_qubits=2)
        with self.assertRaises(ValueError):
            _ = get_diagonal_coulomb_hamiltonian(op3)
        with self.assertRaises(ValueError):
            _ = get_diagonal_coulomb_hamiltonian(op4)
        with self.assertRaises(ValueError):
            _ = get_diagonal_coulomb_hamiltonian(op5)
        with self.assertRaises(ValueError):
            _ = get_diagonal_coulomb_hamiltonian(op6)


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


class GetSparseOperatorBosonTest(unittest.TestCase):
    def setUp(self):
        self.hbar = 1.
        self.d = 4
        self.b = numpy.diag(numpy.sqrt(numpy.arange(1, self.d)), 1)
        self.bd = self.b.conj().T
        self.q = numpy.sqrt(self.hbar/2)*(self.b + self.bd)

    def test_sparse_matrix_ladder(self):
        sparse_operator = get_sparse_operator(BosonOperator('0'), trunc=self.d)
        self.assertTrue(numpy.allclose(sparse_operator.toarray(), self.b))
        self.assertEqual(sparse_operator.shape, (self.d, self.d))

    def test_sparse_matrix_quad(self):
        sparse_operator = get_sparse_operator(QuadOperator('q0'), trunc=self.d)
        self.assertTrue(numpy.allclose(sparse_operator.toarray(), self.q))
        self.assertEqual(sparse_operator.shape, (self.d, self.d))

    def test_sparse_matrix_error(self):
        with self.assertRaises(TypeError):
            _ = get_sparse_operator(1)


class GetSparseOperatorDiagonalCoulombHamiltonianTest(unittest.TestCase):

    def test_diagonal_coulomb_hamiltonian(self):
        n_qubits = 5
        one_body = random_hermitian_matrix(n_qubits, real=False)
        two_body = random_hermitian_matrix(n_qubits, real=True)
        constant = numpy.random.randn()
        op = DiagonalCoulombHamiltonian(one_body, two_body, constant)

        op1 = get_sparse_operator(op)
        op2 = get_sparse_operator(jordan_wigner(get_fermion_operator(op)))
        diff = op1 - op2
        discrepancy = 0.
        if diff.nnz:
            discrepancy = max(abs(diff.data))
        self.assertAlmostEqual(discrepancy, 0.)


class GetQuadOperatorTest(unittest.TestCase):

    def setUp(self):
        self.hbar = 0.5

    def test_invalid_op(self):
        op = QuadOperator()
        with self.assertRaises(TypeError):
            _ = get_quad_operator(op)

    def test_zero(self):
        b = BosonOperator()
        q = get_quad_operator(b)
        self.assertTrue(q == QuadOperator.zero())

    def test_identity(self):
        b = BosonOperator('')
        q = get_quad_operator(b)
        self.assertTrue(q == QuadOperator.identity())

    def test_creation(self):
        b = BosonOperator('0^')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') - 1j*QuadOperator('p0')
        expected /= numpy.sqrt(2*self.hbar)
        self.assertTrue(q == expected)

    def test_annihilation(self):
        b = BosonOperator('0')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') + 1j*QuadOperator('p0')
        expected /= numpy.sqrt(2*self.hbar)
        self.assertTrue(q == expected)

    def test_two_mode(self):
        b = BosonOperator('0^ 2')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') - 1j*QuadOperator('p0')
        expected *= (QuadOperator('q2') + 1j*QuadOperator('p2'))
        expected /= 2*self.hbar
        self.assertTrue(q == expected)

    def test_two_term(self):
        b = BosonOperator('0^ 0') + BosonOperator('0 0^')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = (QuadOperator('q0') - 1j*QuadOperator('p0')) \
            * (QuadOperator('q0') + 1j*QuadOperator('p0')) \
            + (QuadOperator('q0') + 1j*QuadOperator('p0')) \
            * (QuadOperator('q0') - 1j*QuadOperator('p0'))
        expected /= 2*self.hbar
        self.assertTrue(q == expected)

    def test_q_squared(self):
        b = self.hbar*(BosonOperator('0^ 0^') + BosonOperator('0 0')
                       + BosonOperator('') + 2*BosonOperator('0^ 0'))/2
        q = normal_ordered(
            get_quad_operator(b, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q0 q0')
        self.assertTrue(q == expected)

    def test_p_squared(self):
        b = self.hbar*(-BosonOperator('1^ 1^') - BosonOperator('1 1')
                       + BosonOperator('') + 2*BosonOperator('1^ 1'))/2
        q = normal_ordered(
            get_quad_operator(b, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('p1 p1')
        self.assertTrue(q == expected)


class GetBosonOperatorTest(unittest.TestCase):

    def setUp(self):
        self.hbar = 0.5

    def test_invalid_op(self):
        op = BosonOperator()
        with self.assertRaises(TypeError):
            _ = get_boson_operator(op)

    def test_zero(self):
        q = QuadOperator()
        b = get_boson_operator(q)
        self.assertTrue(b == BosonOperator.zero())

    def test_identity(self):
        q = QuadOperator('')
        b = get_boson_operator(q)
        self.assertTrue(b == BosonOperator.identity())

    def test_x(self):
        q = QuadOperator('q0')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = BosonOperator('0') + BosonOperator('0^')
        expected *= numpy.sqrt(self.hbar/2)
        self.assertTrue(b == expected)

    def test_p(self):
        q = QuadOperator('p2')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = BosonOperator('2') - BosonOperator('2^')
        expected *= -1j*numpy.sqrt(self.hbar/2)
        self.assertTrue(b == expected)

    def test_two_mode(self):
        q = QuadOperator('p2 q0')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = -1j*self.hbar/2 \
            * (BosonOperator('0') + BosonOperator('0^')) \
            * (BosonOperator('2') - BosonOperator('2^'))
        self.assertTrue(b == expected)

    def test_two_term(self):
        q = QuadOperator('p0 q0') + QuadOperator('q0 p0')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = -1j*self.hbar/2 \
            * ((BosonOperator('0') + BosonOperator('0^'))
               * (BosonOperator('0') - BosonOperator('0^'))
               + (BosonOperator('0') - BosonOperator('0^'))
               * (BosonOperator('0') + BosonOperator('0^')))
        self.assertTrue(b == expected)
