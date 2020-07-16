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

import unittest
import os

import numpy
import pytest
import scipy

from openfermion.hamiltonians import fermi_hubbard
from openfermion.ops import (BosonOperator,
                             DiagonalCoulombHamiltonian,
                             FermionOperator,
                             MajoranaOperator,
                             QuadOperator,
                             QubitOperator)
from openfermion.transforms import jordan_wigner
from openfermion.utils import (is_hermitian,
                               normal_ordered,
                               jw_number_restrict_operator,
                               jw_sz_restrict_operator,
                               sparse_eigenspectrum)

from openfermion.hamiltonians._molecular_data import MolecularData
from openfermion.ops._interaction_operator import InteractionOperatorError
from openfermion.ops._quadratic_hamiltonian import QuadraticHamiltonianError
from openfermion.utils._testing_utils import (random_hermitian_matrix,
                                              random_quadratic_hamiltonian)
from openfermion.config import THIS_DIRECTORY

from openfermion.transforms._conversion import (
    get_boson_operator, get_diagonal_coulomb_hamiltonian, get_fermion_operator,
    get_interaction_operator, get_majorana_operator,
    _fermion_operator_to_majorana_operator, _fermion_term_to_majorana_operator,
    get_quad_operator, get_quadratic_hamiltonian, get_sparse_operator,
    get_number_preserving_sparse_operator, _iterate_basis_)


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
        converted_op = get_quadratic_hamiltonian(ferm_op,
                                                 ignore_incompatible_terms=True)
        self.assertTrue(
            numpy.allclose(converted_op.hermitian_part,
                           numpy.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])))


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
            normal_ordered(hubbard_model) == normal_ordered(
                get_fermion_operator(
                    get_diagonal_coulomb_hamiltonian(hubbard_model))))

    def test_random_quadratic(self):
        n_qubits = 5
        quad_ham = random_quadratic_hamiltonian(n_qubits, True)
        ferm_op = get_fermion_operator(quad_ham)
        self.assertTrue(
            normal_ordered(ferm_op) == normal_ordered(
                get_fermion_operator(get_diagonal_coulomb_hamiltonian(
                    ferm_op))))

    def test_ignore_incompatible_terms(self):

        ferm_op = (FermionOperator('0^ 2') + FermionOperator('2^ 0') +
                   FermionOperator('1^ 0^ 2') + FermionOperator('1^ 0^ 2 1') +
                   FermionOperator('0^ 0 1^ 1') + FermionOperator('1^ 2^ 1 2'))
        converted_op = get_diagonal_coulomb_hamiltonian(
            ferm_op, ignore_incompatible_terms=True)
        self.assertTrue(
            numpy.allclose(converted_op.one_body,
                           numpy.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])))
        self.assertTrue(
            numpy.allclose(
                converted_op.two_body,
                numpy.array([[0, 0.5, 0], [0.5, 0, -0.5], [0, -0.5, 0]])))

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


def test_get_fermion_operator_majorana_operator():
    a = MajoranaOperator((0, 3), 2.0) + MajoranaOperator((1, 2, 3))
    op = get_fermion_operator(a)
    expected_op = (-2j*(FermionOperator(((0, 0), (1, 0)))
                        - FermionOperator(((0, 0), (1, 1)))
                        + FermionOperator(((0, 1), (1, 0)))
                        - FermionOperator(((0, 1), (1, 1))))
                   - 2*FermionOperator(((0, 0), (1, 1), (1, 0)))
                   + 2*FermionOperator(((0, 1), (1, 1), (1, 0)))
                   + FermionOperator((0, 0))
                   - FermionOperator((0, 1)))
    assert normal_ordered(op) == normal_ordered(expected_op)


def test_get_fermion_operator_wrong_type():
    with pytest.raises(TypeError):
        _ = get_fermion_operator(QubitOperator())


class GetMajoranaOperatorTest(unittest.TestCase):
    """Test class get Majorana Operator."""

    def test_raises(self):
        """Test raises errors."""
        with self.assertRaises(TypeError):
            get_majorana_operator(1.0)
        with self.assertRaises(TypeError):
            _fermion_operator_to_majorana_operator([1.0])
        with self.assertRaises(TypeError):
            _fermion_term_to_majorana_operator(1.0)

    def test_get_majorana_operator_fermion_operator(self):
        """Test conversion FermionOperator to MajoranaOperator."""
        fermion_op = (-2j * (FermionOperator(
            ((0, 0), (1, 0))) - FermionOperator(
                ((0, 0), (1, 1))) + FermionOperator(
                    ((0, 1), (1, 0))) - FermionOperator(
                        ((0, 1), (1, 1)))) - 2 * FermionOperator(
                            ((0, 0), (1, 1), (1, 0))) + 2 * FermionOperator(
                                ((0, 1), (1, 1), (1, 0))) + FermionOperator(
                                    (0, 0)) - FermionOperator((0, 1)))

        majorana_op = get_majorana_operator(fermion_op)
        expected_op = (MajoranaOperator((0, 3), 2.0) + MajoranaOperator(
            (1, 2, 3)))
        self.assertTrue(majorana_op == expected_op)

    def test_get_majorana_operator_diagonalcoulomb(self):
        """Test get majorana from Diagonal Coulomb."""
        fermion_op = (FermionOperator('0^ 1', 1.0) +
                      FermionOperator('1^ 0', 1.0))

        diagonal_ham = get_diagonal_coulomb_hamiltonian(fermion_op)

        self.assertTrue(
            get_majorana_operator(diagonal_ham) == get_majorana_operator(
                fermion_op))


class GetNumberPreservingSparseOperatorIntegrationTestLiH(unittest.TestCase):

    def setUp(self):
        # Set up molecule.
        geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H1-Li1_sto-3g_singlet_1.45')
        self.molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        self.molecule.load()

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

        self.hubbard_hamiltonian = fermi_hubbard(
            2, 2, 1.0, 4.0,
            chemical_potential=.2,
            magnetic_field=0.0,
            spinless=False)

    def test_number_on_reference(self):
        sum_n_op = FermionOperator()
        sum_sparse_n_op = get_number_preserving_sparse_operator(
            sum_n_op,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        space_size = sum_sparse_n_op.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        for i in range(self.molecule.n_qubits):
            n_op = FermionOperator(((i, 1), (i, 0)))
            sum_n_op += n_op

            sparse_n_op = get_number_preserving_sparse_operator(
                n_op,
                self.molecule.n_qubits,
                self.molecule.n_electrons,
                spin_preserving=False)

            sum_sparse_n_op += sparse_n_op

            expectation = reference.dot(sparse_n_op.dot(reference))

            if i < self.molecule.n_electrons:
                assert expectation == 1.0
            else:
                assert expectation == 0.0

        convert_after_adding = get_number_preserving_sparse_operator(
            sum_n_op,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        assert scipy.sparse.linalg.norm(convert_after_adding -
                                        sum_sparse_n_op) < 1E-9

        assert reference.dot(sum_sparse_n_op.dot(reference)) - \
            self.molecule.n_electrons < 1E-9

    def test_space_size_correct(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        space_size = sparse_ham.shape[0]

        # Naive Hilbert space size is 2**12, or 4096.
        assert space_size == 225

    def test_hf_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        space_size = sparse_ham.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        sparse_hf_energy = reference.dot(sparse_ham.dot(reference))

        assert numpy.linalg.norm(sparse_hf_energy -
                                 self.molecule.hf_energy) < 1E-9

    def test_one_body_hf_energy(self):
        one_body_part = self.molecular_hamiltonian
        one_body_part.two_body_tensor = numpy.zeros_like(
            one_body_part.two_body_tensor)

        one_body_fop = get_fermion_operator(one_body_part)
        one_body_regular_sparse_op = get_sparse_operator(one_body_fop)

        make_hf_fop = FermionOperator(((3, 1), (2, 1), (1, 1), (0, 1)))
        make_hf_sparse_op = get_sparse_operator(make_hf_fop, n_qubits=12)

        hf_state = numpy.zeros((2**12))
        hf_state[0] = 1.0
        hf_state = make_hf_sparse_op.dot(hf_state)

        regular_sparse_hf_energy = \
            (hf_state.dot(one_body_regular_sparse_op.dot(hf_state))).real

        one_body_sparse_op = get_number_preserving_sparse_operator(
            one_body_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        space_size = one_body_sparse_op.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        sparse_hf_energy = reference.dot(one_body_sparse_op.dot(reference))

        assert numpy.linalg.norm(sparse_hf_energy -
                                 regular_sparse_hf_energy) < 1E-9

    def test_ground_state_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which='SA')

        assert numpy.abs(eig_val[0] - self.molecule.fci_energy) < 1E-9

    def test_doubles_are_subset(self):
        reference_determinants = [
            [True, True, True, True, False, False,
             False, False, False, False, False, False],
            [True, True, False, False, False, False,
             True, True, False, False, False, False]
        ]

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=2,
                                    spin_preserving=True)))
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1])

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=4,
                                    spin_preserving=True)))
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1])

            for item in doubles_int_state_array:
                assert item in all_int_state_array

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=2,
                                    spin_preserving=True)))
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1])

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=4,
                                    spin_preserving=False)))
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1])

            for item in doubles_int_state_array:
                assert item in all_int_state_array

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=2,
                                    spin_preserving=False)))
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1])

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=4,
                                    spin_preserving=False)))
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1])

            for item in doubles_int_state_array:
                assert item in all_int_state_array

    def test_full_ham_hermitian(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_full_ham_hermitian_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_singles_simple_one_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (1, 0)))
        fop_conj = FermionOperator(((1, 1), (3, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_op -
                                        sparse_op_conj.getH()) < 1E-9

    def test_singles_simple_two_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (8, 1), (1, 0), (4, 0)))
        fop_conj = FermionOperator(((4, 1), (1, 1), (8, 0), (3, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_op -
                                        sparse_op_conj.getH()) < 1E-9

    def test_singles_repeating_two_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (1, 1), (5, 0), (1, 0)))
        fop_conj = FermionOperator(((5, 1), (1, 1), (3, 0), (1, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_op -
                                        sparse_op_conj.getH()) < 1E-9

    def test_singles_ham_hermitian(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_singles_ham_hermitian_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_cisd_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=2)

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which='SA')

        assert numpy.abs(eig_val[0] - self.molecule.cisd_energy) < 1E-9

    def test_cisd_energy_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
            excitation_level=2)

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which='SA')

        assert numpy.abs(eig_val[0] - self.molecule.cisd_energy) < 1E-9

    def test_cisd_matches_fci_energy_two_electron_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_cisd = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            8,
            2,
            spin_preserving=True,
            excitation_level=2)

        sparse_ham_fci = get_sparse_operator(
            hamiltonian_fop,
            n_qubits=8)

        eig_val_cisd, _ = scipy.sparse.linalg.eigsh(sparse_ham_cisd,
                                                    k=1,
                                                    which='SA')
        eig_val_fci, _ = scipy.sparse.linalg.eigsh(sparse_ham_fci,
                                                   k=1,
                                                   which='SA')

        assert numpy.abs(eig_val_cisd[0] - eig_val_fci[0]) < 1E-9

    def test_weird_determinant_matches_fci_energy_two_electron_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_cisd = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            8,
            2,
            spin_preserving=True,
            excitation_level=2,
            reference_determinant=numpy.asarray(
                [False, False, True, True, False, False, False, False]))

        sparse_ham_fci = get_sparse_operator(
            hamiltonian_fop,
            n_qubits=8)

        eig_val_cisd, _ = scipy.sparse.linalg.eigsh(sparse_ham_cisd,
                                                    k=1,
                                                    which='SA')
        eig_val_fci, _ = scipy.sparse.linalg.eigsh(sparse_ham_fci,
                                                   k=1,
                                                   which='SA')

        assert numpy.abs(eig_val_cisd[0] - eig_val_fci[0]) < 1E-9

    def test_number_restricted_spectra_match_molecule(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham_number_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        sparse_ham = get_sparse_operator(hamiltonian_fop,
                                         self.molecule.n_qubits)

        sparse_ham_restricted_number_preserving = jw_number_restrict_operator(
            sparse_ham,
            n_electrons=self.molecule.n_electrons,
            n_qubits=self.molecule.n_qubits)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_preserving)

        spectral_deviation = numpy.amax(numpy.absolute(
            spectrum_from_new_sparse_method - spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)

    def test_number_restricted_spectra_match_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_number_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            8,
            4,
            spin_preserving=False)

        sparse_ham = get_sparse_operator(hamiltonian_fop,
                                         8)

        sparse_ham_restricted_number_preserving = jw_number_restrict_operator(
            sparse_ham,
            n_electrons=4,
            n_qubits=8)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_preserving)

        spectral_deviation = numpy.amax(numpy.absolute(
            spectrum_from_new_sparse_method - spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)

    def test_number_sz_restricted_spectra_match_molecule(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham_number_sz_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        sparse_ham = get_sparse_operator(hamiltonian_fop,
                                         self.molecule.n_qubits)

        sparse_ham_restricted_number_sz_preserving = jw_sz_restrict_operator(
            sparse_ham,
            0,
            n_electrons=self.molecule.n_electrons,
            n_qubits=self.molecule.n_qubits)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_sz_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_sz_preserving)

        spectral_deviation = numpy.amax(numpy.absolute(
            spectrum_from_new_sparse_method - spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)

    def test_number_sz_restricted_spectra_match_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_number_sz_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            8,
            4,
            spin_preserving=True)

        sparse_ham = get_sparse_operator(hamiltonian_fop,
                                         8)

        sparse_ham_restricted_number_sz_preserving = jw_sz_restrict_operator(
            sparse_ham,
            0,
            n_electrons=4,
            n_qubits=8)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_sz_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_sz_preserving)

        spectral_deviation = numpy.amax(numpy.absolute(
            spectrum_from_new_sparse_method - spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)
