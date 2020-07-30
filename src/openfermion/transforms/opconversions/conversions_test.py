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
'''Tests for conversions.py'''
import unittest
import pytest
import numpy
import sympy

from openfermion.chem import MolecularData
from openfermion.config import EQ_TOLERANCE
from openfermion.ops.operators import (QuadOperator, BosonOperator,
                                       FermionOperator, MajoranaOperator,
                                       QubitOperator)
from openfermion.ops.representations import (InteractionOperatorError,
                                             QuadraticHamiltonianError)
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms.opconversions import (
    get_quad_operator, get_boson_operator, get_majorana_operator,
    get_fermion_operator, get_diagonal_coulomb_hamiltonian,
    get_interaction_operator, get_quadratic_hamiltonian, check_no_sympy,
    get_molecular_data)
from openfermion.transforms.opconversions.term_reordering import normal_ordered
from openfermion.testing.testing_utils import random_quadratic_hamiltonian
from openfermion.transforms.opconversions.conversions import (
    _fermion_operator_to_majorana_operator, _fermion_term_to_majorana_operator)


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
        expected = QuadOperator('q0') - 1j * QuadOperator('p0')
        expected /= numpy.sqrt(2 * self.hbar)
        self.assertTrue(q == expected)

    def test_annihilation(self):
        b = BosonOperator('0')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') + 1j * QuadOperator('p0')
        expected /= numpy.sqrt(2 * self.hbar)
        self.assertTrue(q == expected)

    def test_two_mode(self):
        b = BosonOperator('0^ 2')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') - 1j * QuadOperator('p0')
        expected *= (QuadOperator('q2') + 1j * QuadOperator('p2'))
        expected /= 2 * self.hbar
        self.assertTrue(q == expected)

    def test_two_term(self):
        b = BosonOperator('0^ 0') + BosonOperator('0 0^')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = (QuadOperator('q0') - 1j*QuadOperator('p0')) \
            * (QuadOperator('q0') + 1j*QuadOperator('p0')) \
            + (QuadOperator('q0') + 1j*QuadOperator('p0')) \
            * (QuadOperator('q0') - 1j*QuadOperator('p0'))
        expected /= 2 * self.hbar
        self.assertTrue(q == expected)

    def test_q_squared(self):
        b = self.hbar * (BosonOperator('0^ 0^') + BosonOperator('0 0') +
                         BosonOperator('') + 2 * BosonOperator('0^ 0')) / 2
        q = normal_ordered(get_quad_operator(b, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q0 q0')
        self.assertTrue(q == expected)

    def test_p_squared(self):
        b = self.hbar * (-BosonOperator('1^ 1^') - BosonOperator('1 1') +
                         BosonOperator('') + 2 * BosonOperator('1^ 1')) / 2
        q = normal_ordered(get_quad_operator(b, hbar=self.hbar), hbar=self.hbar)
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
        expected *= numpy.sqrt(self.hbar / 2)
        self.assertTrue(b == expected)

    def test_p(self):
        q = QuadOperator('p2')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = BosonOperator('2') - BosonOperator('2^')
        expected *= -1j * numpy.sqrt(self.hbar / 2)
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
    expected_op = (-2j * (FermionOperator(((0, 0), (1, 0))) - FermionOperator(
        ((0, 0), (1, 1))) + FermionOperator(((0, 1), (1, 0))) - FermionOperator(
            ((0, 1), (1, 1)))) - 2 * FermionOperator(
                ((0, 0), (1, 1), (1, 0))) + 2 * FermionOperator(
                    ((0, 1), (1, 1), (1, 0))) + FermionOperator(
                        (0, 0)) - FermionOperator((0, 1)))
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



class RaisesSympyExceptionTest(unittest.TestCase):

    def test_raises_sympy_expression(self):
        operator = FermionOperator('0^', sympy.Symbol('x'))
        with self.assertRaises(TypeError):
            check_no_sympy(operator)


class GetInteractionOperatorTest(unittest.TestCase):

    def test_get_molecular_operator(self):
        coefficient = 3.
        operators = ((2, 1), (3, 0), (0, 0), (3, 1))
        op = FermionOperator(operators, coefficient)

        molecular_operator = get_interaction_operator(op)
        fermion_operator = get_fermion_operator(molecular_operator)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(normal_ordered(op) == fermion_operator)

        op = FermionOperator('1^ 1')
        op *= 0.5 * EQ_TOLERANCE
        molecular_operator = get_interaction_operator(op)
        self.assertEqual(molecular_operator.constant, 0)
        self.assertTrue(
            numpy.allclose(molecular_operator.one_body_tensor,
                           numpy.zeros((2, 2))))

    def test_get_interaction_operator_bad_input(self):
        with self.assertRaises(TypeError):
            get_interaction_operator('3')

    def test_get_interaction_operator_below_threshold(self):
        op = get_interaction_operator(FermionOperator('1^ 1', 0.0))
        self.assertEqual(op.constant, 0)
        self.assertTrue(numpy.allclose(op.one_body_tensor, numpy.zeros((1, 1))))
        self.assertTrue(
            numpy.allclose(op.two_body_tensor, numpy.zeros((1, 1, 1, 1))))

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
        coefficient = 3.
        operators = ((2, 1), (3, 0), (0, 0), (3, 1))
        op = FermionOperator(operators, coefficient)

        molecular_operator = get_interaction_operator(op)
        molecule = get_molecular_data(molecular_operator,
                                      geometry=[['H', [0, 0, 0]]],
                                      basis='aug-cc-pvtz',
                                      multiplicity=2,
                                      n_electrons=1)
        self.assertTrue(isinstance(molecule, MolecularData))

        molecule = get_molecular_data(molecular_operator,
                                      geometry=[['H', [0, 0, 0]]],
                                      basis='aug-cc-pvtz',
                                      multiplicity=2,
                                      n_electrons=1,
                                      reduce_spin=False)
        self.assertTrue(isinstance(molecule, MolecularData))


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
        self.assertTrue(normal_ordered(self.hermitian_op) == fermion_operator)

        # Non-particle-number-conserving chemical potential
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op,
                                                 chemical_potential=3.)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(normal_ordered(self.hermitian_op) == fermion_operator)

        # Particle-number-conserving
        quadratic_op = get_quadratic_hamiltonian(self.hermitian_op_pc)
        fermion_operator = get_fermion_operator(quadratic_op)
        fermion_operator = normal_ordered(fermion_operator)
        self.assertTrue(
            normal_ordered(self.hermitian_op_pc) == fermion_operator)

        fop = FermionOperator('1^ 1')
        fop *= 0.5E-8
        quad_op = get_quadratic_hamiltonian(fop)
        self.assertEqual(quad_op.constant, 0)

    def test_get_quadratic_hamiltonian_hermitian_bad_term(self):
        """Test an operator with non-quadratic terms."""
        with self.assertRaises(QuadraticHamiltonianError):
            get_quadratic_hamiltonian(self.hermitian_op_bad_term)

    def test_get_quadratic_hamiltonian_threshold(self):
        """Test an operator with non-quadratic terms."""
        quad_op = get_quadratic_hamiltonian(FermionOperator('1^ 1', 0))
        self.assertEqual(quad_op.constant, 0)

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
        op7 = FermionOperator('0^ 1^ 2 3')
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
        with self.assertRaises(ValueError):
            _ = get_diagonal_coulomb_hamiltonian(op7)

    def test_threshold(self):
        op = get_diagonal_coulomb_hamiltonian(FermionOperator('1^ 1', 0))
        self.assertEqual(op.constant, 0)

        fop = FermionOperator('1^ 1')
        fop *= 0.5 * EQ_TOLERANCE
        op = get_diagonal_coulomb_hamiltonian(fop)
        self.assertEqual(op.constant, 0)
