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
import numpy

from openfermion.chem import MolecularData
from openfermion.config import EQ_TOLERANCE
from openfermion.ops.operators import (FermionOperator, QubitOperator)
from openfermion.hamiltonians import fermi_hubbard
from openfermion.ops.representations import (InteractionOperatorError,
                                             QuadraticHamiltonianError)
from openfermion.testing.testing_utils import random_quadratic_hamiltonian
from openfermion.transforms.opconversions import get_fermion_operator
from openfermion.transforms.opconversions.term_reordering import normal_ordered

from openfermion.transforms.repconversions.conversions import (
    get_diagonal_coulomb_hamiltonian,
    get_molecular_data,
    get_interaction_operator,
    get_quadratic_hamiltonian,
)


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