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

import itertools
import os
import unittest

import numpy
from scipy.sparse import csc_matrix
from openfermion.config import EQ_TOLERANCE
from openfermion.hamiltonians import fermi_hubbard, plane_wave_hamiltonian
from openfermion.ops import *
from openfermion.transforms import (bravyi_kitaev, jordan_wigner,
                                    get_fermion_operator,
                                    get_interaction_operator,
                                    get_sparse_operator)
from openfermion.utils import (Grid, is_hermitian,
                               normal_ordered, number_operator,
                               random_interaction_operator)

from openfermion.utils._operator_utils import *
from openfermion.utils._testing_utils import random_interaction_operator


class OperatorUtilsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.majorana_operator = MajoranaOperator((1, 4, 9))
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(
            self.fermion_operator)

    def test_n_qubits_majorana_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.majorana_operator))

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

        with self.assertRaises(TypeError):
            _ = eigenspectrum(BosonOperator())

        with self.assertRaises(TypeError):
            _ = eigenspectrum(QuadOperator())

    def test_is_identity_unit_fermionoperator(self):
        self.assertTrue(is_identity(FermionOperator(())))

    def test_is_identity_double_of_unit_fermionoperator(self):
        self.assertTrue(is_identity(2. * FermionOperator(())))

    def test_is_identity_unit_bosonoperator(self):
        self.assertTrue(is_identity(BosonOperator(())))

    def test_is_identity_double_of_unit_bosonoperator(self):
        self.assertTrue(is_identity(2. * BosonOperator(())))

    def test_is_identity_unit_quadoperator(self):
        self.assertTrue(is_identity(QuadOperator(())))

    def test_is_identity_double_of_unit_quadoperator(self):
        self.assertTrue(is_identity(2. * QuadOperator(())))

    def test_is_identity_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator(())))

    def test_is_identity_double_of_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator((), 2.)))

    def test_not_is_identity_single_term_fermionoperator(self):
        self.assertFalse(is_identity(FermionOperator('1^')))

    def test_not_is_identity_single_term_bosonoperator(self):
        self.assertFalse(is_identity(BosonOperator('1^')))

    def test_not_is_identity_single_term_quadoperator(self):
        self.assertFalse(is_identity(QuadOperator('q1')))

    def test_not_is_identity_single_term_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator('X1')))

    def test_not_is_identity_zero_bosonoperator(self):
        self.assertFalse(is_identity(BosonOperator()))

    def test_not_is_identity_zero_quadoperator(self):
        self.assertFalse(is_identity(QuadOperator()))

    def test_not_is_identity_zero_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator()))

    def test_is_identity_bad_type(self):
        with self.assertRaises(TypeError):
            _ = is_identity('eleven')

    def test_reorder(self):
        def shift_by_one(x, y):
            return (x + 1) % y
        operator = FermionOperator('1^ 2^ 3 4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms,
                         {((2, 1), (3, 1), (4, 0), (0, 0)): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms,
                         {((0, 1), (1, 1), (2, 0), (3, 0)): -3.17})

    def test_reorder_boson(self):
        shift_by_one = lambda x, y: (x + 1) % y
        operator = BosonOperator('1^ 2^ 3 4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms,
                         {((0, 0), (2, 1), (3, 1), (4, 0)): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms,
                         {((0, 1), (1, 1), (2, 0), (3, 0)): -3.17})

    def test_reorder_quad(self):
        shift_by_one = lambda x, y: (x + 1) % y
        operator = QuadOperator('q1 q2 p3 p4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms,
                         {((0, 'p'), (2, 'q'), (3, 'q'), (4, 'p')): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms,
                         {((0, 'q'), (1, 'q'), (2, 'p'), (3, 'p')): -3.17})

    def test_up_then_down(self):
        for LadderOp in (FermionOperator, BosonOperator):
            operator = LadderOp('1^ 2^ 3 4', -3.17)
            reordered = reorder(operator, up_then_down)
            reordered = reorder(reordered, up_then_down, reverse=True)

            self.assertEqual(reordered.terms, operator.terms)
            self.assertEqual(up_then_down(6, 8), 3)
            self.assertEqual(up_then_down(3, 8), 5)


class ChemistOrderingTest(unittest.TestCase):

    def test_convert_forward_back(self):
        n_qubits = 6
        random_operator = get_fermion_operator(
            random_interaction_operator(n_qubits))
        chemist_operator = chemist_ordered(random_operator)
        normalized_chemist = normal_ordered(chemist_operator)
        difference = normalized_chemist - normal_ordered(random_operator)
        self.assertAlmostEqual(0., difference.induced_norm())

    def test_exception(self):
        n_qubits = 6
        random_operator = get_fermion_operator(
            random_interaction_operator(n_qubits))
        bad_term = ((2, 1), (3, 1))
        random_operator += FermionOperator(bad_term)
        with self.assertRaises(OperatorSpecificationError):
            chemist_ordered(random_operator)

    def test_form(self):
        n_qubits = 6
        random_operator = get_fermion_operator(
            random_interaction_operator(n_qubits))
        chemist_operator = chemist_ordered(random_operator)
        for term, _ in chemist_operator.terms.items():
            if len(term) == 2 or not len(term):
                pass
            else:
                self.assertTrue(term[0][1])
                self.assertTrue(term[2][1])
                self.assertFalse(term[1][1])
                self.assertFalse(term[3][1])
                self.assertTrue(term[0][0] > term[2][0])
                self.assertTrue(term[1][0] > term[3][0])


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


class HermitianConjugatedTest(unittest.TestCase):

    def test_hermitian_conjugated_qubit_op(self):
        """Test conjugating QubitOperators."""
        op = QubitOperator()
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertEqual(op_hc, correct_op)

        op = QubitOperator('X0 Y1', 2.)
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertEqual(op_hc, correct_op)

        op = QubitOperator('X0 Y1', 2.j)
        op_hc = hermitian_conjugated(op)
        correct_op = QubitOperator('X0 Y1', -2.j)
        self.assertEqual(op_hc, correct_op)

        op = QubitOperator('X0 Y1', 2.) + QubitOperator('Z4 X5 Y7', 3.j)
        op_hc = hermitian_conjugated(op)
        correct_op = (QubitOperator('X0 Y1', 2.) +
                      QubitOperator('Z4 X5 Y7', -3.j))
        self.assertEqual(op_hc, correct_op)

    def test_hermitian_conjugated_qubit_op_consistency(self):
        """Some consistency checks for conjugating QubitOperators."""
        ferm_op = (FermionOperator('1^ 2') + FermionOperator('2 3 4') +
                   FermionOperator('2^ 7 9 11^'))

        # Check that hermitian conjugation commutes with transforms
        self.assertEqual(jordan_wigner(hermitian_conjugated(ferm_op)),
                         hermitian_conjugated(jordan_wigner(ferm_op)))
        self.assertEqual(bravyi_kitaev(hermitian_conjugated(ferm_op)),
                         hermitian_conjugated(bravyi_kitaev(ferm_op)))

    def test_hermitian_conjugated_quad_op(self):
        """Test conjugating QuadOperator."""
        op = QuadOperator()
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertTrue(op_hc == correct_op)

        op = QuadOperator('q0 p1', 2.)
        op_hc = hermitian_conjugated(op)
        correct_op = op
        self.assertTrue(op_hc == correct_op)

        op = QuadOperator('q0 p1', 2.j)
        op_hc = hermitian_conjugated(op)
        correct_op = QuadOperator('q0 p1', -2.j)
        self.assertTrue(op_hc == correct_op)

        op = QuadOperator('q0 p1', 2.) + QuadOperator('q4 q5 p7', 3.j)
        op_hc = hermitian_conjugated(op)
        correct_op = (QuadOperator('q0 p1', 2.) +
                      QuadOperator('q4 q5 p7', -3.j))
        self.assertTrue(op_hc == correct_op)

        op = QuadOperator('q0 p0 q1', 2.) + QuadOperator('q1 p1 p2', 3.j)
        op_hc = hermitian_conjugated(op)
        correct_op = (QuadOperator('p0 q0 q1', 2.) +
                      QuadOperator('p1 q1 p2', -3.j))
        self.assertTrue(op_hc == correct_op)

    def test_hermitian_conjugate_empty(self):
        op = FermionOperator()
        op = hermitian_conjugated(op)
        self.assertEqual(op, FermionOperator())

        op = BosonOperator()
        op = hermitian_conjugated(op)
        self.assertEqual(op, BosonOperator())

    def test_hermitian_conjugate_simple(self):
        op = FermionOperator('1^')
        op_hc = FermionOperator('1')
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

        op = BosonOperator('1^')
        op_hc = BosonOperator('1')
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugate_complex_const(self):
        op = FermionOperator('1^ 3', 3j)
        op_hc = -3j * FermionOperator('3^ 1')
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

        op = BosonOperator('1^ 3', 3j)
        op_hc = -3j * BosonOperator('3^ 1')
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugate_notordered(self):
        op = FermionOperator('1 3^ 3 3^', 3j)
        op_hc = -3j * FermionOperator('3 3^ 3 1^')
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

        op = BosonOperator('1 3^ 3 3^', 3j)
        op_hc = -3j * BosonOperator('3 3^ 3 1^')
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugate_semihermitian(self):
        op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
              FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
        op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
                 FermionOperator('3^ 1', -2j) +
                 FermionOperator('2^ 2', -0.1j))
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

        op = (BosonOperator() + 2j * BosonOperator('1^ 3') +
              BosonOperator('3^ 1') * -2j + BosonOperator('2^ 2', 0.1j))
        op_hc = (BosonOperator() + BosonOperator('1^ 3', 2j) +
                 BosonOperator('3^ 1', -2j) +
                 BosonOperator('2^ 2', -0.1j))
        op = hermitian_conjugated(op)
        self.assertEqual(op, op_hc)

    def test_hermitian_conjugated_empty(self):
        op = FermionOperator()
        self.assertEqual(op, hermitian_conjugated(op))
        op = BosonOperator()
        self.assertEqual(op, hermitian_conjugated(op))

    def test_hermitian_conjugated_simple(self):
        op = FermionOperator('0')
        op_hc = FermionOperator('0^')
        self.assertEqual(op_hc, hermitian_conjugated(op))

        op = BosonOperator('0')
        op_hc = BosonOperator('0^')
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_complex_const(self):
        op = FermionOperator('2^ 2', 3j)
        op_hc = FermionOperator('2^ 2', -3j)
        self.assertEqual(op_hc, hermitian_conjugated(op))

        op = BosonOperator('2^ 2', 3j)
        op_hc = BosonOperator('2^ 2', -3j)
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_multiterm(self):
        op = FermionOperator('1^ 2') + FermionOperator('2 3 4')
        op_hc = FermionOperator('2^ 1') + FermionOperator('4^ 3^ 2^')
        self.assertEqual(op_hc, hermitian_conjugated(op))

        op = BosonOperator('1^ 2') + BosonOperator('2 3 4')
        op_hc = BosonOperator('2^ 1') + BosonOperator('4^ 3^ 2^')
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_semihermitian(self):
        op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
              FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
        op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
                 FermionOperator('3^ 1', -2j) +
                 FermionOperator('2^ 2', -0.1j))
        self.assertEqual(op_hc, hermitian_conjugated(op))

        op = (BosonOperator() + 2j * BosonOperator('1^ 3') +
              BosonOperator('3^ 1') * -2j + BosonOperator('2^ 2', 0.1j))
        op_hc = (BosonOperator() + BosonOperator('1^ 3', 2j) +
                 BosonOperator('3^ 1', -2j) +
                 BosonOperator('2^ 2', -0.1j))
        self.assertEqual(op_hc, hermitian_conjugated(op))

    def test_hermitian_conjugated_interaction_operator(self):
        for n_orbitals, _ in itertools.product((1, 2, 5), range(5)):
            operator = random_interaction_operator(n_orbitals)
            qubit_operator = jordan_wigner(operator)
            conjugate_operator = hermitian_conjugated(operator)
            conjugate_qubit_operator = jordan_wigner(conjugate_operator)
            assert hermitian_conjugated(qubit_operator) == \
                conjugate_qubit_operator

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = is_hermitian('a')

        with self.assertRaises(TypeError):
            _ = hermitian_conjugated(1)


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

        op = fermi_hubbard(2, 2, 1., 1.)
        self.assertTrue(is_hermitian(op))

    def test_boson_operator_zero(self):
        op = BosonOperator()
        self.assertTrue(is_hermitian(op))

    def test_boson_operator_identity(self):
        op = BosonOperator(())
        self.assertTrue(is_hermitian(op))

    def test_boson_operator_nonhermitian(self):
        op = BosonOperator('0^ 1 2^ 3')
        self.assertFalse(is_hermitian(op))

    def test_boson_operator_hermitian(self):
        op = BosonOperator('0^ 1 2^ 3')
        op += BosonOperator('3^ 2 1^ 0')
        self.assertTrue(is_hermitian(op))

    def test_quad_operator_zero(self):
        op = QuadOperator()
        self.assertTrue(is_hermitian(op))

    def test_quad_operator_identity(self):
        op = QuadOperator(())
        self.assertTrue(is_hermitian(op))

    def test_quad_operator_nonhermitian(self):
        op = QuadOperator('q0 p1 q1')
        self.assertFalse(is_hermitian(op))

    def test_quad_operator_hermitian(self):
        op = QuadOperator('q0 p1 q2 p3')
        self.assertTrue(is_hermitian(op))

        op = QuadOperator('q0 p0 q1 p1')
        op += QuadOperator('p0 q0 p1 q1')
        self.assertTrue(is_hermitian(op))

        # TODO: insert bose_hubbard here
        # op = fermi_hubbard(2, 2, 1., 1.)
        # self.assertTrue(is_hermitian(op))

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
        self.boson_term = BosonOperator('1^ 2^ 3 4', -3.17)
        self.boson_operator = self.boson_term + hermitian_conjugated(
            self.boson_term)
        self.quad_term = QuadOperator('q0 p0 q1 p0 p0', -3.17)
        self.quad_operator = self.quad_term + hermitian_conjugated(
            self.quad_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.file_name = "test_file"

    def tearDown(self):
        file_path = os.path.join(DATA_DIRECTORY, self.file_name + '.data')
        if os.path.isfile(file_path):
            os.remove(file_path)

    def test_save_and_load_fermion_operators(self):
        save_operator(self.fermion_operator, self.file_name)
        loaded_fermion_operator = load_operator(self.file_name)
        self.assertEqual(self.fermion_operator,
                         loaded_fermion_operator,
                         msg=str(self.fermion_operator -
                                 loaded_fermion_operator))

    def test_save_and_load_fermion_operators_readably(self):
        save_operator(self.fermion_operator, self.file_name,
                      plain_text=True)
        loaded_fermion_operator = load_operator(self.file_name,
                                                plain_text=True)
        self.assertTrue(self.fermion_operator == loaded_fermion_operator)

    def test_save_and_load_boson_operators(self):
        save_operator(self.boson_operator, self.file_name)
        loaded_boson_operator = load_operator(self.file_name)
        self.assertEqual(self.boson_operator.terms,
                         loaded_boson_operator.terms,
                         msg=str(self.boson_operator -
                                 loaded_boson_operator))

    def test_save_and_load_boson_operators_readably(self):
        save_operator(self.boson_operator, self.file_name,
                      plain_text=True)
        loaded_boson_operator = load_operator(self.file_name,
                                              plain_text=True)
        self.assertTrue(self.boson_operator == loaded_boson_operator)

    def test_save_and_load_quad_operators(self):
        save_operator(self.quad_operator, self.file_name)
        loaded_quad_operator = load_operator(self.file_name)
        self.assertEqual(self.quad_operator.terms,
                         loaded_quad_operator.terms)

    def test_save_and_load_quad_operators_readably(self):
        save_operator(self.quad_operator, self.file_name,
                      plain_text=True)
        loaded_quad_operator = load_operator(self.file_name,
                                             plain_text=True)
        self.assertTrue(self.quad_operator == loaded_quad_operator)

    def test_save_and_load_qubit_operators(self):
        save_operator(self.qubit_operator, self.file_name)
        loaded_qubit_operator = load_operator(self.file_name)
        self.assertTrue(self.qubit_operator == loaded_qubit_operator)

    def test_save_and_load_qubit_operators_readably(self):
        save_operator(self.qubit_operator, self.file_name, plain_text=True)
        loaded_qubit_operator = load_operator(self.file_name,
                                              plain_text=True)
        self.assertEqual(self.qubit_operator,
                         loaded_qubit_operator)

    def test_save_readably(self):
        save_operator(self.fermion_operator, self.file_name, plain_text=True)
        file_path = os.path.join(DATA_DIRECTORY, self.file_name + '.data')
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), "\n".join([
                "FermionOperator:",
                "-3.17 [1^ 2^ 3 4] +",
                "-3.17 [4^ 3^ 2 1]"
            ]))

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

        self.assertEqual(fermion_operator, self.fermion_operator)

    def test_load_bad_type(self):
        with self.assertRaises(TypeError):
            _ = load_operator('bad_type_operator')

    def test_save_bad_type(self):
        with self.assertRaises(TypeError):
            save_operator('ping', 'somewhere')


class FourierTransformTest(unittest.TestCase):

    def test_fourier_transform(self):
        for length in [2, 3]:
            grid = Grid(dimensions=1, scale=1.5, length=length)
            spinless_set = [True, False]
            geometry = [('H', (0.1,)), ('H', (0.5,))]
            for spinless in spinless_set:
                h_plane_wave = plane_wave_hamiltonian(
                    grid, geometry, spinless, True)
                h_dual_basis = plane_wave_hamiltonian(
                    grid, geometry, spinless, False)
                h_plane_wave_t = fourier_transform(h_plane_wave, grid,
                                                   spinless)

                self.assertEqual(normal_ordered(h_plane_wave_t),
                                 normal_ordered(h_dual_basis))

                # Verify that all 3 are Hermitian
                plane_wave_operator = get_sparse_operator(h_plane_wave)
                dual_operator = get_sparse_operator(h_dual_basis)
                plane_wave_t_operator = get_sparse_operator(h_plane_wave_t)
                self.assertTrue(is_hermitian(plane_wave_operator))
                self.assertTrue(is_hermitian(dual_operator))
                self.assertTrue(is_hermitian(plane_wave_t_operator))

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
            self.assertEqual(normal_ordered(h_dual_basis_t),
                             normal_ordered(h_plane_wave))

    def test_inverse_fourier_transform_2d(self):
        grid = Grid(dimensions=2, scale=1.5, length=3)
        spinless = True
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless, True)
        h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless, False)
        h_dual_basis_t = inverse_fourier_transform(
            h_dual_basis, grid, spinless)
        self.assertEqual(normal_ordered(h_dual_basis_t),
                         normal_ordered(h_plane_wave))


class TestNormalOrdering(unittest.TestCase):

    def test_boson_single_term(self):
        op = BosonOperator('4 3 2 1') + BosonOperator('3 2')
        self.assertTrue(op == normal_ordered(op))

    def test_boson_two_term(self):
        op_b = BosonOperator(((2, 0), (4, 0), (2, 1)), 88.)
        normal_ordered_b = normal_ordered(op_b)
        expected = (BosonOperator(((4, 0),), 88.) +
                    BosonOperator(((2, 1), (4, 0), (2, 0)), 88.))
        self.assertTrue(normal_ordered_b == expected)

    def test_boson_number(self):
        number_op2 = BosonOperator(((2, 1), (2, 0)))
        self.assertTrue(number_op2 == normal_ordered(number_op2))

    def test_boson_number_reversed(self):
        n_term_rev2 = BosonOperator(((2, 0), (2, 1)))
        number_op2 = number_operator(3, 2, parity=1)
        expected = BosonOperator(()) + number_op2
        self.assertTrue(normal_ordered(n_term_rev2) == expected)

    def test_boson_offsite(self):
        op = BosonOperator(((3, 1), (2, 0)))
        self.assertTrue(op == normal_ordered(op))

    def test_boson_offsite_reversed(self):
        op = BosonOperator(((3, 0), (2, 1)))
        expected = BosonOperator(((2, 1), (3, 0)))
        self.assertTrue(expected == normal_ordered(op))

    def test_boson_multi(self):
        op = BosonOperator(((2, 0), (1, 1), (2, 1)))
        expected = (BosonOperator(((2, 1), (1, 1), (2, 0))) +
                    BosonOperator(((1, 1),)))
        self.assertTrue(expected == normal_ordered(op))

    def test_boson_triple(self):
        op_132 = BosonOperator(((1, 1), (3, 0), (2, 0)))
        op_123 = BosonOperator(((1, 1), (2, 0), (3, 0)))
        op_321 = BosonOperator(((3, 0), (2, 0), (1, 1)))

        self.assertTrue(op_132 == normal_ordered(op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_fermion_single_term(self):
        op = FermionOperator('4 3 2 1') + FermionOperator('3 2')
        self.assertTrue(op == normal_ordered(op))

    def test_fermion_two_term(self):
        op_b = FermionOperator(((2, 0), (4, 0), (2, 1)), -88.)
        normal_ordered_b = normal_ordered(op_b)
        expected = (FermionOperator(((4, 0),), 88.) +
                    FermionOperator(((2, 1), (4, 0), (2, 0)), 88.))
        self.assertTrue(normal_ordered_b == expected)

    def test_fermion_number(self):
        number_op2 = FermionOperator(((2, 1), (2, 0)))
        self.assertTrue(number_op2 == normal_ordered(number_op2))

    def test_fermion_number_reversed(self):
        n_term_rev2 = FermionOperator(((2, 0), (2, 1)))
        number_op2 = number_operator(3, 2)
        expected = FermionOperator(()) - number_op2
        self.assertTrue(normal_ordered(n_term_rev2) == expected)

    def test_fermion_offsite(self):
        op = FermionOperator(((3, 1), (2, 0)))
        self.assertTrue(op == normal_ordered(op))

    def test_fermion_offsite_reversed(self):
        op = FermionOperator(((3, 0), (2, 1)))
        expected = -FermionOperator(((2, 1), (3, 0)))
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_double_create(self):
        op = FermionOperator(((2, 0), (3, 1), (3, 1)))
        expected = FermionOperator((), 0.0)
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_double_create_separated(self):
        op = FermionOperator(((3, 1), (2, 0), (3, 1)))
        expected = FermionOperator((), 0.0)
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_multi(self):
        op = FermionOperator(((2, 0), (1, 1), (2, 1)))
        expected = (-FermionOperator(((2, 1), (1, 1), (2, 0))) -
                    FermionOperator(((1, 1),)))
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_triple(self):
        op_132 = FermionOperator(((1, 1), (3, 0), (2, 0)))
        op_123 = FermionOperator(((1, 1), (2, 0), (3, 0)))
        op_321 = FermionOperator(((3, 0), (2, 0), (1, 1)))

        self.assertTrue(op_132 == normal_ordered(-op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_quad_single_term(self):
        op = QuadOperator('p4 p3 p2 p1') + QuadOperator('p3 p2')
        self.assertTrue(op == normal_ordered(op))

        op = QuadOperator('q0 p0') - QuadOperator('p0 q0')
        expected = QuadOperator('', 2.j)
        self.assertTrue(expected == normal_ordered(op, hbar=2.))

    def test_quad_two_term(self):
        op_b = QuadOperator('p0 q0 p3', 88.)
        normal_ordered_b = normal_ordered(op_b, hbar=2)
        expected = QuadOperator('p3', -88.*2j) + QuadOperator('q0 p0 p3', 88.0)
        self.assertTrue(normal_ordered_b == expected)

    def test_quad_offsite(self):
        op = QuadOperator(((3, 'p'), (2, 'q')))
        self.assertTrue(op == normal_ordered(op))

    def test_quad_offsite_reversed(self):
        op = QuadOperator(((3, 'q'), (2, 'p')))
        expected = QuadOperator(((2, 'p'), (3, 'q')))
        self.assertTrue(expected == normal_ordered(op))

    def test_quad_triple(self):
        op_132 = QuadOperator(((1, 'p'), (3, 'q'), (2, 'q')))
        op_123 = QuadOperator(((1, 'p'), (2, 'q'), (3, 'q')))
        op_321 = QuadOperator(((3, 'q'), (2, 'q'), (1, 'p')))

        self.assertTrue(op_132 == normal_ordered(op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_interaction_operator(self):
        for n_orbitals, real, _ in itertools.product(
                (1, 2, 5), (True, False), range(5)):
            operator = random_interaction_operator(n_orbitals, real=real)
            normal_ordered_operator = normal_ordered(operator)
            expected_qubit_operator = jordan_wigner(operator)
            actual_qubit_operator = jordan_wigner(
                    normal_ordered_operator)
            assert expected_qubit_operator == actual_qubit_operator
            two_body_tensor = normal_ordered_operator.two_body_tensor
            n_orbitals = len(two_body_tensor)
            ones = numpy.ones((n_orbitals,) * 2)
            triu = numpy.triu(ones, 1)
            shape = (n_orbitals ** 2, 1)
            mask = (triu.reshape(shape) * ones.reshape(shape[::-1]) +
                    ones.reshape(shape) * triu.reshape(shape[::-1])
                    ).reshape((n_orbitals,) * 4)
            assert numpy.allclose(mask * two_body_tensor,
                    numpy.zeros((n_orbitals,) * 4))
            for term in normal_ordered_operator:
                order = len(term) // 2
                left_term, right_term = term[:order], term[order:]
                assert all(i[1] == 1 for i in left_term)
                assert all(i[1] == 0 for i in right_term)
                assert left_term == tuple(sorted(left_term, reverse=True))
                assert right_term == tuple(sorted(right_term, reverse=True))

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = normal_ordered(1)


class GroupTensorProductBasisTest(unittest.TestCase):

    def test_demo_qubit_operator(self):
        for seed in [None, 0, 10000]:
            op = QubitOperator('X0 Y1', 2.) + QubitOperator('X1 Y2', 3.j)
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected = {((0, 'X'), (1, 'Y')): QubitOperator('X0 Y1', 2.),
                        ((1, 'X'), (2, 'Y')): QubitOperator('X1 Y2', 3.j)}
            self.assertEqual(sub_operators, expected)

            op = QubitOperator('X0 Y1', 2.) + QubitOperator('Y1 Y2', 3.j)
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected = {((0, 'X'), (1, 'Y'), (2, 'Y')): op}
            self.assertEqual(sub_operators, expected)

            op = QubitOperator('', 4.) + QubitOperator('X1', 2.j)
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected = {((1, 'X'),): op}
            self.assertEqual(sub_operators, expected)

            op = (QubitOperator('X0 X1', 0.1) + QubitOperator('X1 X2', 2.j)
                  + QubitOperator('Y2 Z3', 3.) + QubitOperator('X3 Z4', 5.))
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected1 = {
                ((0, 'X'), (1, 'X'), (2, 'X'),
                 (3, 'X'), (4, 'Z')): (QubitOperator('X0 X1', 0.1)
                                       + QubitOperator('X1 X2', 2.j)
                                       + QubitOperator('X3 Z4', 5.)),
                ((2, 'Y'), (3, 'Z')): QubitOperator('Y2 Z3', 3.)
            }
            expected2 = {
                ((0, 'X'), (1, 'X'),
                 (2, 'Y'), (3, 'Z')): (QubitOperator('X0 X1', 0.1)
                                       + QubitOperator('Y2 Z3', 3.)),
                ((1, 'X'), (2, 'X'),
                 (3, 'X'), (4, 'Z')): (QubitOperator('X1 X2', 2.j)
                                       + QubitOperator('X3 Z4', 5.))
            }
            self.assertTrue(sub_operators == expected1 or
                            sub_operators == expected2)

    def test_empty_qubit_operator(self):
        sub_operators = group_into_tensor_product_basis_sets(QubitOperator())
        self.assertTrue(sub_operators == {})

    def test_fermion_operator_bad_type(self):
        with self.assertRaises(TypeError):
            _ = group_into_tensor_product_basis_sets(FermionOperator())

    def test_boson_operator_bad_type(self):
        with self.assertRaises(TypeError):
            _ = group_into_tensor_product_basis_sets(BosonOperator())

    def test_none_bad_type(self):
        with self.assertRaises(TypeError):
            _ = group_into_tensor_product_basis_sets(None)


class IsContextualTest(unittest.TestCase):

    def setUp(self):
        self.x1 = QubitOperator('X1',1.)
        self.x2 = QubitOperator('X2',1.)
        self.x3 = QubitOperator('X3',1.)
        self.x4 = QubitOperator('X4',1.)
        self.z1 = QubitOperator('Z1',1.)
        self.z2 = QubitOperator('Z2',1.)
        self.x1x2 = QubitOperator('X1 X2',1.)
        self.y1y2 = QubitOperator('Y1 Y2',1.)

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
