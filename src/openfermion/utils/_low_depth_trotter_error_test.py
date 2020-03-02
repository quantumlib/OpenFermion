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

from openfermion.hamiltonians import (
    jellium_model, hypercube_grid_with_given_wigner_seitz_radius_and_filling,
    wigner_seitz_length_scale)
from openfermion.utils._low_depth_trotter_error import *
from openfermion.utils import Grid


class ErrorOperatorTest(unittest.TestCase):

    def test_error_operator(self):
        FO = FermionOperator

        terms = []
        for i in range(4):
            terms.append(FO(((i, 1), (i, 0)), 0.018505508252042547))
            terms.append(FO(((i, 1), ((i + 1) % 4, 0)), -0.012337005501361697))
            terms.append(FO(((i, 1), ((i + 2) % 4, 0)), 0.0061685027506808475))
            terms.append(FO(((i, 1), ((i + 3) % 4, 0)), -0.012337005501361697))
            terms.append(normal_ordered(FO(((i, 1), ((i + 1) % 4, 1),
                                            (i, 0), ((i + 1) % 4, 0)),
                                           3.1830988618379052)))
            if i // 2:
                terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.281692032865351)))

        self.assertAlmostEqual(
            low_depth_second_order_trotter_error_operator(
                terms, jellium_only=True).terms[
                    ((3, 1), (2, 1), (1, 1), (2, 0), (1, 0), (0, 0))],
            -0.562500000003)


class ErrorBoundTest(unittest.TestCase):

    def setUp(self):
        FO = FermionOperator

        self.terms = []
        for i in range(4):
            self.terms.append(FO(((i, 1), (i, 0)),
                                 0.018505508252042547))
            self.terms.append(FO(((i, 1), ((i + 1) % 4, 0)),
                                 -0.012337005501361697))
            self.terms.append(FO(((i, 1), ((i + 2) % 4, 0)),
                                 0.0061685027506808475))
            self.terms.append(FO(((i, 1), ((i + 3) % 4, 0)),
                                 -0.012337005501361697))
            self.terms.append(normal_ordered(FO(((i, 1), ((i + 1) % 4, 1),
                                                 (i, 0), ((i + 1) % 4, 0)),
                                                3.1830988618379052)))
            if i // 2:
                self.terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.281692032865351)))

    def test_error_bound(self):
        self.assertAlmostEqual(low_depth_second_order_trotter_error_bound(
            self.terms, jellium_only=True), 6.92941899358)

    def test_error_bound_using_info_1d(self):
        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension=1, grid_length=4, wigner_seitz_radius=10.)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result
        self.assertAlmostEqual(low_depth_second_order_trotter_error_bound(
            terms, indices, is_hopping), 7.4239378440953283)

    def test_error_bound_using_info_1d_with_input_ordering(self):
        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension=1, grid_length=4, wigner_seitz_radius=10.)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian, input_ordering=[0, 1, 2, 3])
        terms, indices, is_hopping = result
        self.assertAlmostEqual(low_depth_second_order_trotter_error_bound(
            terms, indices, is_hopping), 7.4239378440953283)

    def test_error_bound_using_info_2d_verbose(self):
        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension=2, grid_length=3, wigner_seitz_radius=10.)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result
        self.assertAlmostEqual(
            low_depth_second_order_trotter_error_bound(
                terms, indices, is_hopping, jellium_only=True, verbose=True),
            0.052213321121580794)


class OrderedDualBasisTermsMoreInfoTest(unittest.TestCase):

    def test_sum_of_ordered_terms_equals_full_hamiltonian(self):
        grid_length = 4
        dimension = 2
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension
        n_particles = n_qubits // inverse_filling_fraction

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        terms = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)[0]
        terms_total = sum(terms, FermionOperator.zero())

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        grid = Grid(dimension, grid_length, length_scale)
        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)
        hamiltonian = normal_ordered(hamiltonian)
        self.assertTrue(terms_total == hamiltonian)

    def test_sum_of_ordered_terms_equals_full_hamiltonian_rots_at_end(self):
        grid_length = 4
        dimension = 2
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension
        n_particles = n_qubits // inverse_filling_fraction

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        terms = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian, external_potential_at_end=True)[0]
        terms_total = sum(terms, FermionOperator.zero())

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        grid = Grid(dimension, grid_length, length_scale)
        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)
        hamiltonian = normal_ordered(hamiltonian)
        self.assertTrue(terms_total == hamiltonian)

    def test_correct_indices_terms_with_info(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        grid_length**dimension

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, indices, _ = result

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

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, _, is_hopping = result

        for i in range(len(terms)):
            single_term = list(terms[i].terms)[0]
            is_hopping_term = not (single_term[1][1] or
                                   single_term[0][0] == single_term[1][0])
            self.assertEqual(is_hopping_term, is_hopping[i])

    def test_correct_indices_terms_with_info_external_pot_at_end(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        grid_length**dimension

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian, external_potential_at_end=True)
        terms, indices, _ = result

        for i in range(len(terms)):
            term = list(terms[i].terms)
            term_indices = set()
            for single_term in term:
                term_indices = term_indices.union(
                    [single_term[j][0] for j in range(len(single_term))])
            self.assertEqual(term_indices, indices[i])

        # Last four terms are the rotations
        self.assertListEqual(indices[-4:], [set([i]) for i in range(4)])

    def test_is_hopping_operator_terms_with_info_external_pot_at_end(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian, external_potential_at_end=True)
        terms, _, is_hopping = result

        for i in range(len(terms)):
            single_term = list(terms[i].terms)[0]
            is_hopping_term = not (single_term[1][1] or
                                   single_term[0][0] == single_term[1][0])
            self.assertEqual(is_hopping_term, is_hopping[i])

        # Last four terms are the rotations
        self.assertFalse(sum(is_hopping[-4:]))

    def test_total_length(self):
        grid_length = 8
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2
        n_qubits = grid_length ** dimension

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius,
            1. / inverse_filling_fraction)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), n_qubits * (n_qubits - 1))


class OrderedDualBasisTermsNoInfoTest(unittest.TestCase):

    def test_all_terms_in_standardized_dual_basis_jellium_hamiltonian(self):
        grid_length = 4
        dimension = 1

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius=10.)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        terms = ordered_low_depth_terms_no_info(hamiltonian)
        FO = FermionOperator

        expected_terms = []
        for i in range(grid_length ** dimension):
            expected_terms.append(FO(((i, 1), (i, 0)),
                                     0.018505508252042547))
            expected_terms.append(FO(((i, 1), ((i + 1) % 4, 0)),
                                     -0.012337005501361697))
            expected_terms.append(FO(((i, 1), ((i + 2) % 4, 0)),
                                     0.0061685027506808475))
            expected_terms.append(FO(((i, 1), ((i + 3) % 4, 0)),
                                     -0.012337005501361697))
            expected_terms.append(normal_ordered(
                FO(((i, 1), ((i + 1) % 4, 1), (i, 0), ((i + 1) % 4, 0)),
                   3.1830988618379052)))
            if i // 2:
                expected_terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.281692032865351)))

        for term in terms:
            found_in_other = False
            for term2 in expected_terms:
                if term == term2:
                    self.assertFalse(found_in_other)
                    found_in_other = True
            self.assertTrue(found_in_other, msg=str(term))
        for term in expected_terms:
            found_in_other = False
            for term2 in terms:
                if term == term2:
                    self.assertFalse(found_in_other)
                    found_in_other = True
            self.assertTrue(found_in_other, msg=str(term))

    def test_sum_of_ordered_terms_equals_full_hamiltonian(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0

        # Generate the Hamiltonian.
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dimension, grid_length, wigner_seitz_radius)
        hamiltonian = normal_ordered(jellium_model(grid, spinless=True,
                                                   plane_wave=False))
        hamiltonian.compress()

        terms = ordered_low_depth_terms_no_info(hamiltonian)
        terms_total = sum(terms, FermionOperator.zero())

        self.assertTrue(terms_total == hamiltonian)
