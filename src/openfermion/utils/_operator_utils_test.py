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

"""Tests for operator_utils."""
from __future__ import absolute_import

import numpy
import os
import unittest

from openfermion.config import *
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.ops import *
from openfermion.transforms import (bravyi_kitaev, jordan_wigner,
                                    get_fermion_operator,
                                    get_interaction_operator)
from openfermion.utils import Grid
from openfermion.utils._operator_utils import *
from openfermion.utils._testing_utils import random_interaction_operator
from scipy.sparse import csc_matrix


class OperatorUtilsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(
            self.fermion_operator)

    def test_n_qubits_single_fermion_term(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_term))

    def test_n_qubits_fermion_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_operator))

    def test_n_qubits_qubit_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.qubit_operator))

    def test_n_qubits_interaction_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.interaction_operator))

    def test_n_qubits_bad_type(self):
        with self.assertRaises(TypeError):
            count_qubits('twelve')

    def test_eigenspectrum(self):
        fermion_eigenspectrum = eigenspectrum(self.fermion_operator)
        qubit_eigenspectrum = eigenspectrum(self.qubit_operator)
        interaction_eigenspectrum = eigenspectrum(self.interaction_operator)
        for i in range(2 ** self.n_qubits):
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   qubit_eigenspectrum[i])
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   interaction_eigenspectrum[i])

    def test_is_identity_unit_fermionoperator(self):
        self.assertTrue(is_identity(FermionOperator(())))

    def test_is_identity_double_of_unit_fermionoperator(self):
        self.assertTrue(is_identity(2. * FermionOperator(())))

    def test_is_identity_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator(())))

    def test_is_identity_double_of_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator((), 2.)))

    def test_not_is_identity_single_term_fermionoperator(self):
        self.assertFalse(is_identity(FermionOperator('1^')))

    def test_not_is_identity_single_term_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator('X1')))

    def test_not_is_identity_zero_fermionoperator(self):
        self.assertFalse(is_identity(FermionOperator()))

    def test_not_is_identity_zero_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator()))

    def test_is_identity_bad_type(self):
        with self.assertRaises(TypeError):
            is_identity('eleven')

    def test_reorder(self):
        shift_by_one = lambda x, y: (x + 1) % y
        operator = FermionOperator('1^ 2^ 3 4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms,
                         {((2, 1), (3, 1), (4, 0), (0, 0)): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms,
                         {((0, 1), (1, 1), (2, 0), (3, 0)): -3.17})

    def test_up_then_down(self):
        operator = FermionOperator('1^ 2^ 3 4', -3.17)
        reordered = reorder(operator, up_then_down)
        reordered = reorder(reordered, up_then_down, reverse=True)

        self.assertEqual(reordered.terms, operator.terms)
        self.assertEqual(up_then_down(6, 8), 3)
        self.assertEqual(up_then_down(3, 8), 5)


class HermitianConjugatedTest(unittest.TestCase):

    def test_hermitian_conjugated_qubit_op(self):
        """Test conjugating QubitOperators."""
        op = QubitOperator()
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertTrue(op_hc.isclose(correct_op))

        op = QubitOperator('X0 Y1', 2.)
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertTrue(op_hc.isclose(correct_op))

        op = QubitOperator('X0 Y1', 2.j)
        op_hc = hermitian_conjugated(op)
        correct_op = QubitOperator('X0 Y1', -2.j)
        self.assertTrue(op_hc.isclose(correct_op))

        op = QubitOperator('X0 Y1', 2.) + QubitOperator('Z4 X5 Y7', 3.j)
        op_hc = hermitian_conjugated(op)
        correct_op = (QubitOperator('X0 Y1', 2.) +
                      QubitOperator('Z4 X5 Y7', -3.j))
        self.assertTrue(op_hc.isclose(correct_op))

    def test_hermitian_conjugated_qubit_op_consistency(self):
        """Some consistency checks for conjugating QubitOperators."""
        ferm_op = (FermionOperator('1^ 2') + FermionOperator('2 3 4') +
                   FermionOperator('2^ 7 9 11^'))

        # Check that hermitian conjugation commutes with transforms
        self.assertTrue(jordan_wigner(hermitian_conjugated(ferm_op)).isclose(
            hermitian_conjugated(jordan_wigner(ferm_op))))
        self.assertTrue(bravyi_kitaev(hermitian_conjugated(ferm_op)).isclose(
            hermitian_conjugated(bravyi_kitaev(ferm_op))))

    def test_hermitian_conjugate_empty(self):
        op = FermionOperator()
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(FermionOperator()))

    def test_hermitian_conjugate_simple(self):
        op = FermionOperator('1^')
        op_hc = FermionOperator('1')
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugate_complex_const(self):
        op = FermionOperator('1^ 3', 3j)
        op_hc = -3j * FermionOperator('3^ 1')
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugate_notordered(self):
        op = FermionOperator('1 3^ 3 3^', 3j)
        op_hc = -3j * FermionOperator('3 3^ 3 1^')
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugate_semihermitian(self):
        op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
              FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
        op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
                 FermionOperator('3^ 1', -2j) +
                 FermionOperator('2^ 2', -0.1j))
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugated_empty(self):
        op = FermionOperator()
        self.assertTrue(op.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_simple(self):
        op = FermionOperator('0')
        op_hc = FermionOperator('0^')
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_complex_const(self):
        op = FermionOperator('2^ 2', 3j)
        op_hc = FermionOperator('2^ 2', -3j)
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_multiterm(self):
        op = FermionOperator('1^ 2') + FermionOperator('2 3 4')
        op_hc = FermionOperator('2^ 1') + FermionOperator('4^ 3^ 2^')
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_semihermitian(self):
        op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
              FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
        op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
                 FermionOperator('3^ 1', -2j) +
                 FermionOperator('2^ 2', -0.1j))
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = is_hermitian('a')


class IsHermitianTest(unittest.TestCase):
    
    def test_fermion_operator_zero(self):
        op = FermionOperator()
        self.assertTrue(is_hermitian(op))

    def test_fermion_operator_identity(self):
        op = FermionOperator(())
        self.assertTrue(is_hermitian(op))

    def test_fermion_operator_nonhermitian(self):
        op = FermionOperator('0^ 1 2^ 3')
        self.assertFalse(is_hermitian(op))

    def test_fermion_operator_hermitian(self):
        op = FermionOperator('0^ 1 2^ 3')
        op += FermionOperator('3^ 2 1^ 0')
        self.assertTrue(is_hermitian(op))

    def test_qubit_operator_zero(self):
        op = QubitOperator()
        self.assertTrue(is_hermitian(op))

    def test_qubit_operator_identity(self):
        op = QubitOperator(())
        self.assertTrue(is_hermitian(op))

    def test_qubit_operator_nonhermitian(self):
        op = QubitOperator('X0 Y2 Z5', 1.+2.j)
        self.assertFalse(is_hermitian(op))

    def test_qubit_operator_hermitian(self):
        op = QubitOperator('X0 Y2 Z5', 1.+2.j)
        op += QubitOperator('X0 Y2 Z5', 1.-2.j)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_zero(self):
        op = numpy.zeros((4, 4))
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_identity(self):
        op = numpy.eye(4)
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_nonhermitian(self):
        op = numpy.arange(16).reshape((4, 4))
        self.assertFalse(is_hermitian(op))
        op = csc_matrix(op)
        self.assertFalse(is_hermitian(op))

    def test_sparse_matrix_and_numpy_array_hermitian(self):
        op = numpy.arange(16, dtype=complex).reshape((4, 4))
        op += 1.j * op
        op += op.T.conj()
        self.assertTrue(is_hermitian(op))
        op = csc_matrix(op)
        self.assertTrue(is_hermitian(op))

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = is_hermitian('a')


class SaveLoadOperatorTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.file_name = "test_file"

    def tearDown(self):
        file_path = os.path.join(DATA_DIRECTORY, self.file_name + '.data')
        if os.path.isfile(file_path):
            os.remove(file_path)

    def test_save_and_load_fermion_operators(self):
        save_operator(self.fermion_operator, self.file_name)
        loaded_fermion_operator = load_operator(self.file_name)
        self.assertEqual(self.fermion_operator.terms,
                         loaded_fermion_operator.terms,
                         msg=str(self.fermion_operator -
                                 loaded_fermion_operator))

    def test_save_and_load_qubit_operators(self):
        save_operator(self.qubit_operator, self.file_name)
        loaded_qubit_operator = load_operator(self.file_name)
        self.assertEqual(self.qubit_operator.terms,
                         loaded_qubit_operator.terms)

    def test_save_no_filename_operator_utils_error(self):
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator)

    def test_basic_save(self):
        save_operator(self.fermion_operator, self.file_name)

    def test_save_interaction_operator_not_implemented(self):
        constant = 100.0
        one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits), float)
        one_body[1, 1] = 10.0
        two_body[1, 2, 3, 4] = 12.0
        interaction_operator = InteractionOperator(
            constant, one_body, two_body)
        with self.assertRaises(NotImplementedError):
            save_operator(interaction_operator, self.file_name)

    def test_save_on_top_of_existing_operator_utils_error(self):
        save_operator(self.fermion_operator, self.file_name)
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator, self.file_name)

    def test_save_on_top_of_existing_operator_error_with_explicit_flag(self):
        save_operator(self.fermion_operator, self.file_name)
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator, self.file_name,
                          allow_overwrite=False)

    def test_overwrite_flag_save_on_top_of_existing_operator(self):
        save_operator(self.fermion_operator, self.file_name)
        save_operator(self.fermion_operator, self.file_name,
                      allow_overwrite=True)
        fermion_operator = load_operator(self.file_name)

        self.assertTrue(fermion_operator.isclose(self.fermion_operator))

    def test_load_bad_type(self):
        with self.assertRaises(TypeError):
            load_operator('bad_type_operator')

    def test_save_bad_type(self):
        with self.assertRaises(TypeError):
            save_operator('ping', 'somewhere')


class FourierTransformTest(unittest.TestCase):

    def test_fourier_transform(self):
        grid = Grid(dimensions=1, scale=1.5, length=3)
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(
                grid, geometry, spinless, True)
            h_dual_basis = plane_wave_hamiltonian(
                grid, geometry, spinless, False)
            h_plane_wave_t = fourier_transform(h_plane_wave, grid, spinless)
            self.assertTrue(normal_ordered(h_plane_wave_t).isclose(
                normal_ordered(h_dual_basis)))

    def test_inverse_fourier_transform_1d(self):
        grid = Grid(dimensions=1, scale=1.5, length=4)
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(
                grid, geometry, spinless, True)
            h_dual_basis = plane_wave_hamiltonian(
                grid, geometry, spinless, False)
            h_dual_basis_t = inverse_fourier_transform(
                h_dual_basis, grid, spinless)
            self.assertTrue(normal_ordered(h_dual_basis_t).isclose(
                normal_ordered(h_plane_wave)))

    def test_inverse_fourier_transform_2d(self):
        grid = Grid(dimensions=2, scale=1.5, length=3)
        spinless = True
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless, True)
        h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless, False)
        h_dual_basis_t = inverse_fourier_transform(
            h_dual_basis, grid, spinless)
        self.assertTrue(normal_ordered(h_dual_basis_t).isclose(
            normal_ordered(h_plane_wave)))
