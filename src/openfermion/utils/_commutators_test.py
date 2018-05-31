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

"""Tests for _commutators.py."""
import unittest

from openfermion.ops import FermionOperator, QubitOperator, \
                            BosonOperator, QuadOperator
from openfermion.transforms import jordan_wigner
from openfermion.utils import hermitian_conjugated, normal_ordered
from openfermion.utils._commutators import *
from openfermion.utils._sparse_tools import pauli_matrix_map


class CommutatorTest(unittest.TestCase):

    def setUp(self):
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

    def test_commutes_identity(self):
        com = commutator(FermionOperator.identity(),
                         FermionOperator('2^ 3', 2.3))
        self.assertEqual(com, FermionOperator.zero())

        com = commutator(BosonOperator.identity(),
                         BosonOperator('2^ 3', 2.3))
        self.assertTrue(com == BosonOperator.zero())

        com = commutator(QuadOperator.identity(),
                         QuadOperator('q2 p3', 2.3))
        self.assertTrue(com == QuadOperator.zero())

    def test_commutes_no_intersection(self):
        com = commutator(FermionOperator('2^ 3'), FermionOperator('4^ 5^ 3'))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator.zero())

        com = commutator(BosonOperator('2^ 3'), BosonOperator('4^ 5^ 3'))
        com = normal_ordered(com)
        self.assertTrue(com == BosonOperator.zero())

        com = commutator(QuadOperator('q2 p3'), QuadOperator('q4 q5 p3'))
        com = normal_ordered(com)
        self.assertTrue(com == QuadOperator.zero())

    def test_commutes_number_operators(self):
        com = commutator(FermionOperator('4^ 3^ 4 3'), FermionOperator('2^ 2'))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator.zero())

        com = commutator(BosonOperator('4^ 3^ 4 3'), BosonOperator('2^ 2'))
        com = normal_ordered(com)
        self.assertTrue(com == BosonOperator.zero())

    def test_commutator_hopping_operators(self):
        com = commutator(3 * FermionOperator('1^ 2'), FermionOperator('2^ 3'))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator('1^ 3', 3))

        com = commutator(3 * BosonOperator('1^ 2'), BosonOperator('2^ 3'))
        com = normal_ordered(com)
        self.assertTrue(com == BosonOperator('1^ 3', 3))

    def test_commutator_hopping_with_single_number(self):
        com = commutator(FermionOperator('1^ 2', 1j), FermionOperator('1^ 1'))
        com = normal_ordered(com)
        self.assertEqual(com, -FermionOperator('1^ 2') * 1j)

    def test_commutator_hopping_with_double_number_one_intersection(self):
        com = commutator(FermionOperator('1^ 3'), FermionOperator('3^ 2^ 3 2'))
        com = normal_ordered(com)
        self.assertEqual(com, -FermionOperator('2^ 1^ 3 2'))

    def test_commutator_hopping_with_double_number_two_intersections(self):
        com = commutator(FermionOperator('2^ 3'), FermionOperator('3^ 2^ 3 2'))
        com = normal_ordered(com)
        self.assertEqual(com, FermionOperator.zero())

    def test_commutator(self):
        operator_a = FermionOperator('')
        self.assertEqual(FermionOperator.zero(),
                         commutator(operator_a, self.fermion_operator))
        operator_b = QubitOperator('X1 Y2')
        self.assertEqual(commutator(self.qubit_operator, operator_b),
                         (self.qubit_operator * operator_b -
                          operator_b * self.qubit_operator))

    def test_canonical_boson_commutation_relations(self):
        op_1 = BosonOperator('3')
        op_1_dag = BosonOperator('3^')
        op_2 = BosonOperator('4')
        op_2_dag = BosonOperator('4^')
        zero = BosonOperator()
        one = BosonOperator('')

        self.assertTrue(one == normal_ordered(
            commutator(op_1, op_1_dag)))
        self.assertTrue(zero == normal_ordered(
            commutator(op_1, op_2)))
        self.assertTrue(zero == normal_ordered(
            commutator(op_1, op_2_dag)))
        self.assertTrue(zero == normal_ordered(
            commutator(op_1_dag, op_2)))
        self.assertTrue(zero == normal_ordered(
            commutator(op_1_dag, op_2_dag)))
        self.assertTrue(one == normal_ordered(
            commutator(op_2, op_2_dag)))

    def test_canonical_quad_commutation_relations(self):
        q1 = QuadOperator('q3')
        p1 = QuadOperator('p3')
        q2 = QuadOperator('q4')
        p2 = QuadOperator('p4')
        zero = QuadOperator()
        one = QuadOperator('')
        hbar = 2.

        self.assertTrue(1j*hbar*one == normal_ordered(
            commutator(q1, p1), hbar))
        self.assertTrue(zero == normal_ordered(
            commutator(q1, q2), hbar))
        self.assertTrue(zero == normal_ordered(
            commutator(q1, p2), hbar))
        self.assertTrue(zero == normal_ordered(
            commutator(p1, q2), hbar))
        self.assertTrue(zero == normal_ordered(
            commutator(p1, p2), hbar))
        self.assertTrue(1j*hbar*one == normal_ordered(
            commutator(q2, p2), hbar))

    def test_ndarray_input(self):
        """Test when the inputs are numpy arrays."""
        X = pauli_matrix_map['X'].toarray()
        Y = pauli_matrix_map['Y'].toarray()
        Z = pauli_matrix_map['Z'].toarray()
        self.assertTrue(numpy.allclose(commutator(X, Y), 2.j * Z))

    def test_commutator_operator_a_bad_type(self):
        with self.assertRaises(TypeError):
            commutator(1, self.fermion_operator)

    def test_commutator_operator_b_bad_type(self):
        with self.assertRaises(TypeError):
            commutator(self.qubit_operator, "hello")

    def test_commutator_not_same_type(self):
        with self.assertRaises(TypeError):
            commutator(self.fermion_operator, self.qubit_operator)


class AnticommutatorTest(unittest.TestCase):

    def test_canonical_anticommutation_relations(self):
        op_1 = FermionOperator('3')
        op_1_dag = FermionOperator('3^')
        op_2 = FermionOperator('4')
        op_2_dag = FermionOperator('4^')
        zero = FermionOperator.zero()
        one = FermionOperator.identity()

        self.assertEqual(one,
                         normal_ordered(anticommutator(op_1, op_1_dag)))
        self.assertEqual(zero,
                         normal_ordered(anticommutator(op_1, op_2)))
        self.assertEqual(zero,
                         normal_ordered(anticommutator(op_1, op_2_dag)))
        self.assertEqual(zero,
                         normal_ordered(anticommutator(op_1_dag, op_2)))
        self.assertEqual(zero,
                         normal_ordered(anticommutator(op_1_dag, op_2_dag)))
        self.assertEqual(one,
                         normal_ordered(anticommutator(op_2, op_2_dag)))

    def test_ndarray_input(self):
        """Test when the inputs are numpy arrays."""
        X = pauli_matrix_map['X'].toarray()
        Y = pauli_matrix_map['Y'].toarray()
        zero = numpy.zeros((2, 2))
        self.assertTrue(numpy.allclose(anticommutator(X, Y), zero))

    def test_anticommutator_not_same_type(self):
        with self.assertRaises(TypeError):
            anticommutator(FermionOperator(), QubitOperator())


class DoubleCommutatorTest(unittest.TestCase):

    def test_double_commutator_no_intersection_with_union_of_second_two(self):
        com = double_commutator(FermionOperator('4^ 3^ 6 5'),
                                FermionOperator('2^ 1 0'),
                                FermionOperator('0^'))
        self.assertEqual(com, FermionOperator.zero())

    def test_double_commutator_more_info_not_hopping(self):
        com = double_commutator(
            FermionOperator('3^ 2'),
            FermionOperator('2^ 3') + FermionOperator('3^ 2'),
            FermionOperator('4^ 2^ 4 2'), indices2=set([2, 3]),
            indices3=set([2, 4]), is_hopping_operator2=True,
            is_hopping_operator3=False)
        self.assertEqual(com, (FermionOperator('4^ 2^ 4 2') -
                               FermionOperator('4^ 3^ 4 3')))

    def test_double_commtator_more_info_both_hopping(self):
        com = double_commutator(
            FermionOperator('4^ 3^ 4 3'),
            FermionOperator('1^ 2', 2.1) + FermionOperator('2^ 1', 2.1),
            FermionOperator('1^ 3', -1.3) + FermionOperator('3^ 1', -1.3),
            indices2=set([1, 2]), indices3=set([1, 3]),
            is_hopping_operator2=True, is_hopping_operator3=True)
        self.assertEqual(com, (FermionOperator('4^ 3^ 4 2', 2.73) +
                               FermionOperator('4^ 2^ 4 3', 2.73)))


class TriviallyDoubleCommutesDualBasisUsingTermInfoTest(unittest.TestCase):

    def test_number_operators_trivially_commute(self):
        self.assertTrue(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=False, is_hopping_operator_beta=False,
            is_hopping_operator_alpha_prime=False, jellium_only=True))

    def test_left_hopping_operator_no_trivial_commutation(self):
        self.assertFalse(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False, jellium_only=True))

    def test_right_hopping_operator_no_trivial_commutation(self):
        self.assertFalse(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=False,
            is_hopping_operator_alpha_prime=True, jellium_only=True))

    def test_alpha_is_hopping_operator_others_number_trivial_commutation(self):
        self.assertTrue(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=False,
            is_hopping_operator_alpha_prime=False, jellium_only=True))

    def test_no_intersection_in_first_commutator_trivially_commutes(self):
        self.assertTrue(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([1, 2]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False, jellium_only=True))

    def test_double_intersection_in_first_commutator_trivially_commutes(self):
        self.assertTrue(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([3, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([4, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False, jellium_only=True))

    def test_single_intersection_in_first_commutator_nontrivial(self):
        self.assertFalse(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([3, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([4, 5]),
            is_hopping_operator_alpha=False, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False, jellium_only=True))

    def test_no_intersection_between_first_and_other_terms_is_trivial(self):
        self.assertTrue(trivially_double_commutes_dual_basis_using_term_info(
            indices_alpha=set([3, 2]), indices_beta=set([1, 4]),
            indices_alpha_prime=set([4, 5]),
            is_hopping_operator_alpha=False, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False, jellium_only=True))


class TriviallyCommutesDualBasisTest(unittest.TestCase):

    def test_trivially_commutes_no_intersection(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('3^ 2^ 3 2'), FermionOperator('4^ 1')))

    def test_no_trivial_commute_with_intersection(self):
        self.assertFalse(trivially_commutes_dual_basis(
            FermionOperator('2^ 1'), FermionOperator('5^ 2^ 5 2')))

    def test_trivially_commutes_both_single_number_operators(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('3^ 3'), FermionOperator('3^ 3')))

    def test_trivially_commutes_nonintersecting_single_number_operators(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('2^ 2'), FermionOperator('3^ 3')))

    def test_trivially_commutes_both_double_number_operators(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 1^ 3 1')))

    def test_trivially_commutes_one_double_number_operators(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 3')))

    def test_no_trivial_commute_right_hopping_operator(self):
        self.assertFalse(trivially_commutes_dual_basis(
            FermionOperator('3^ 1^ 3 1'), FermionOperator('3^ 2')))

    def test_no_trivial_commute_left_hopping_operator(self):
        self.assertFalse(trivially_commutes_dual_basis(
            FermionOperator('3^ 2'), FermionOperator('3^ 3')))

    def test_trivially_commutes_both_hopping_create_same_mode(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('3^ 2'), FermionOperator('3^ 1')))

    def test_trivially_commutes_both_hopping_annihilate_same_mode(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('4^ 1'), FermionOperator('3^ 1')))

    def test_trivially_commutes_both_hopping_and_number_on_same_modes(self):
        self.assertTrue(trivially_commutes_dual_basis(
            FermionOperator('4^ 1'), FermionOperator('4^ 1^ 4 1')))


class TriviallyDoubleCommutesDualBasisTest(unittest.TestCase):

    def test_trivially_double_commutes_no_intersection(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('3^ 4'),
            FermionOperator('3^ 2^ 3 2'), FermionOperator('4^ 1')))

    def test_no_trivial_double_commute_with_intersection(self):
        self.assertFalse(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 2'),
            FermionOperator('2^ 1'), FermionOperator('5^ 2^ 5 2')))

    def test_trivially_double_commutes_both_single_number_operators(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 3'), FermionOperator('3^ 3')))

    def test_trivially_double_commutes_nonintersecting_single_number_ops(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('3^ 2'),
            FermionOperator('2^ 2'), FermionOperator('3^ 3')))

    def test_trivially_double_commutes_both_double_number_operators(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 1^ 3 1')))

    def test_trivially_double_commutes_one_double_number_operators(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 3')))

    def test_no_trivial_double_commute_right_hopping_operator(self):
        self.assertFalse(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 1^ 3 1'), FermionOperator('3^ 2')))

    def test_no_trivial_double_commute_left_hopping_operator(self):
        self.assertFalse(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 2'), FermionOperator('3^ 3')))

    def test_trivially_double_commutes_both_hopping_create_same_mode(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('3^ 3'),
            FermionOperator('3^ 2'), FermionOperator('3^ 1')))

    def test_trivially_double_commutes_both_hopping_annihilate_same_mode(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('1^ 1'),
            FermionOperator('4^ 1'), FermionOperator('3^ 1')))

    def test_trivially_double_commutes_hopping_and_number_on_same_modes(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('4^ 3'),
            FermionOperator('4^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_trivially_double_commutes_no_intersection_a_with_bc(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_trivially_double_commutes_double_create_in_a_and_b(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_trivially_double_commutes_double_annihilate_in_a_and_c(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_no_trivial_double_commute_double_annihilate_with_create(self):
        self.assertFalse(trivially_double_commutes_dual_basis(
            FermionOperator('5^ 2'),
            FermionOperator('2^ 1'), FermionOperator('4^ 2')))

    def test_trivially_double_commutes_excess_create(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('5^ 2'),
            FermionOperator('5^ 5'), FermionOperator('5^ 1')))

    def test_trivially_double_commutes_excess_annihilate(self):
        self.assertTrue(trivially_double_commutes_dual_basis(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 2'), FermionOperator('2^ 2')))
