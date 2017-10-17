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

"""Tests for _dual_basis_trotter_error.py."""
import unittest

from openfermion.ops import FermionOperator
from openfermion.hamiltonians import jellium_model, wigner_seitz_length_scale
from openfermion.utils._dual_basis_trotter_error import *
from openfermion.utils import Grid


class DoubleCommutatorTest(unittest.TestCase):

    def test_double_commutator_no_intersection_with_union_of_second_two(self):
        com = double_commutator(FermionOperator('4^ 3^ 6 5'),
                                FermionOperator('2^ 1 0'),
                                FermionOperator('0^'))
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_double_commutator_more_info_not_hopping(self):
        com = double_commutator(
            FermionOperator('3^ 2'),
            FermionOperator('2^ 3') + FermionOperator('3^ 2'),
            FermionOperator('4^ 2^ 4 2'), indices2=set([2, 3]),
            indices3=set([2, 4]), is_hopping_operator2=True,
            is_hopping_operator3=False)
        self.assertTrue(com.isclose(FermionOperator('4^ 2^ 4 2') -
                                    FermionOperator('4^ 3^ 4 3')))

    def test_double_commtator_more_info_both_hopping(self):
        com = double_commutator(
            FermionOperator('4^ 3^ 4 3'),
            FermionOperator('1^ 2', 2.1) + FermionOperator('2^ 1', 2.1),
            FermionOperator('1^ 3', -1.3) + FermionOperator('3^ 1', -1.3),
            indices2=set([1, 2]), indices3=set([1, 3]),
            is_hopping_operator2=True, is_hopping_operator3=True)
        self.assertTrue(com.isclose(FermionOperator('4^ 3^ 4 2', 2.73) +
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


class ErrorOperatorTest(unittest.TestCase):

    def test_error_operator(self):
        FO = FermionOperator

        terms = []
        for i in range(4):
            terms.append(FO(((i, 1), (i, 0)), 0.018505508252))
            terms.append(FO(((i, 1), ((i + 1) % 4, 0)), -0.0123370055014))
            terms.append(FO(((i, 1), ((i + 2) % 4, 0)), 0.00616850275068))
            terms.append(FO(((i, 1), ((i + 3) % 4, 0)), -0.0123370055014))
            terms.append(normal_ordered(FO(((i, 1), ((i + 1) % 4, 1),
                                            (i, 0), ((i + 1) % 4, 0)),
                                           3.18309886184)))
            if i // 2:
                terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

        self.assertAlmostEqual(
            dual_basis_error_operator(terms, jellium_only=True).terms[
                ((3, 1), (2, 1), (1, 1), (2, 0), (1, 0), (0, 0))],
            -0.562500000003)


class ErrorBoundTest(unittest.TestCase):

    def setUp(self):
        FO = FermionOperator

        self.terms = []
        for i in range(4):
            self.terms.append(FO(((i, 1), (i, 0)), 0.018505508252))
            self.terms.append(FO(((i, 1), ((i + 1) % 4, 0)), -0.0123370055014))
            self.terms.append(FO(((i, 1), ((i + 2) % 4, 0)), 0.00616850275068))
            self.terms.append(FO(((i, 1), ((i + 3) % 4, 0)), -0.0123370055014))
            self.terms.append(normal_ordered(FO(((i, 1), ((i + 1) % 4, 1),
                                                 (i, 0), ((i + 1) % 4, 0)),
                                                3.18309886184)))
            if i // 2:
                self.terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

    def test_error_bound(self):
        self.assertAlmostEqual(dual_basis_error_bound(
            self.terms, jellium_only=True), 6.92941899358)

    def test_error_bound_using_info_1d(self):
        # Generate the Hamiltonian.
        hamiltonian = dual_basis_jellium_hamiltonian(grid_length=4,
                                                     dimension=1)

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_dual_basis_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result
        self.assertAlmostEqual(dual_basis_error_bound(
            terms, indices, is_hopping), 7.4239378440953283)

    def test_error_bound_using_info_2d_verbose(self):
        # Generate the Hamiltonian.
        hamiltonian = dual_basis_jellium_hamiltonian(grid_length=3,
                                                     dimension=2)

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_dual_basis_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result
        self.assertAlmostEqual(0.052213321121580794, dual_basis_error_bound(
            terms, indices, is_hopping, jellium_only=True, verbose=True))


class OrderedDualBasisTermsMoreInfoTest(unittest.TestCase):

    def test_sum_of_ordered_terms_equals_full_hamiltonian(self):
        grid_length = 4
        dimension = 2
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension

        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction

        # Generate the Hamiltonian.
        hamiltonian = dual_basis_jellium_hamiltonian(
            grid_length, dimension, wigner_seitz_radius, n_particles)

        terms = simulation_ordered_grouped_dual_basis_terms_with_info(
            hamiltonian)[0]
        terms_total = sum(terms, FermionOperator.zero())

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        grid = Grid(dimension, grid_length, length_scale)
        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)
        hamiltonian = normal_ordered(hamiltonian)
        self.assertTrue(terms_total.isclose(hamiltonian))

    def test_correct_indices_terms_with_info(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension

        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction

        # Generate the Hamiltonian.
        hamiltonian = dual_basis_jellium_hamiltonian(
            grid_length, dimension, wigner_seitz_radius, n_particles)

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_dual_basis_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        for i in range(len(terms)):
            term = list(terms[i].terms)
            term_indices = set()
            for single_term in term:
                term_indices = term_indices.union(
                    [single_term[j][0] for j in range(len(single_term))])
            self.assertEqual(term_indices, indices[i])

    def test_is_hopping_operator_terms_with_info(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension

        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction

        hamiltonian = dual_basis_jellium_hamiltonian(
            grid_length, dimension, wigner_seitz_radius, n_particles)

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_dual_basis_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        for i in range(len(terms)):
            single_term = list(terms[i].terms)[0]
            is_hopping_term = not (single_term[1][1] or
                                   single_term[0][0] == single_term[1][0])
            self.assertEqual(is_hopping_term, is_hopping[i])

    def test_total_length(self):
        grid_length = 8
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension

        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction

        hamiltonian = dual_basis_jellium_hamiltonian(
            grid_length, dimension, wigner_seitz_radius, n_particles)

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_dual_basis_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        self.assertEqual(len(terms), n_qubits * (n_qubits - 1))


class OrderedDualBasisTermsNoInfoTest(unittest.TestCase):

    def test_all_terms_in_dual_basis_jellium_hamiltonian(self):
        grid_length = 4
        dimension = 1

        # Generate the Hamiltonian.
        hamiltonian = dual_basis_jellium_hamiltonian(grid_length, dimension)

        terms = ordered_dual_basis_terms_no_info(hamiltonian)
        FO = FermionOperator

        expected_terms = []
        for i in range(grid_length ** dimension):
            expected_terms.append(FO(((i, 1), (i, 0)),
                                     0.018505508252))
            expected_terms.append(FO(((i, 1), ((i + 1) % 4, 0)),
                                     -0.0123370055014))
            expected_terms.append(FO(((i, 1), ((i + 2) % 4, 0)),
                                     0.00616850275068))
            expected_terms.append(FO(((i, 1), ((i + 3) % 4, 0)),
                                     -0.0123370055014))
            expected_terms.append(normal_ordered(
                FO(((i, 1), ((i + 1) % 4, 1), (i, 0), ((i + 1) % 4, 0)),
                   3.18309886184)))
            if i // 2:
                expected_terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

        for term in terms:
            found_in_other = False
            for term2 in expected_terms:
                if term.isclose(term2, rel_tol=1e-8):
                    self.assertFalse(found_in_other)
                    found_in_other = True
            self.assertTrue(found_in_other, msg=str(term))
        for term in expected_terms:
            found_in_other = False
            for term2 in terms:
                if term.isclose(term2, rel_tol=1e-8):
                    self.assertFalse(found_in_other)
                    found_in_other = True
            self.assertTrue(found_in_other, msg=str(term))

    def test_sum_of_ordered_terms_equals_full_hamiltonian(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension

        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction
        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        hamiltonian = dual_basis_jellium_hamiltonian(grid_length, dimension)
        terms = ordered_dual_basis_terms_no_info(hamiltonian)
        terms_total = sum(terms, FermionOperator.zero())

        grid = Grid(dimension, grid_length, length_scale)
        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)
        hamiltonian = normal_ordered(hamiltonian)
        self.assertTrue(terms_total.isclose(hamiltonian))


if __name__ == '__main__':
    unittest.main()
