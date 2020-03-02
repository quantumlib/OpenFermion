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

"""Tests for _hubbard_trotter_error.py."""
import unittest

from openfermion.ops import FermionOperator
from openfermion.hamiltonians import fermi_hubbard
from openfermion.utils._hubbard_trotter_error import *
from openfermion.utils._low_depth_trotter_error import (
    low_depth_second_order_trotter_error_bound,
    low_depth_second_order_trotter_error_operator)
from openfermion.utils._operator_utils import normal_ordered


class ErrorOperatorTest(unittest.TestCase):
    def test_error_operator(self):
        FO = FermionOperator

        terms = []
        for i in range(4):
            terms.append(FO(((i, 1), ((i + 1) % 4, 0)), -0.0123370055014))
            terms.append(FO(((i, 1), ((i - 1) % 4, 0)), -0.0123370055014))

            if i in [0, 2]:
                terms.append(normal_ordered(FO(((i + 1, 1), (i, 1),
                                                (i + 1, 0), (i, 0)),
                                               3.18309886184)))
            if i < 2:
                terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

        self.assertAlmostEqual(
            low_depth_second_order_trotter_error_operator(terms).terms[
                ((3, 1), (2, 1), (1, 1), (2, 0), (1, 0), (0, 0))], 0.75)


class ErrorBoundTest(unittest.TestCase):
    def test_error_bound_superset_hubbard(self):
        FO = FermionOperator
        terms = []

        for i in [0, 2]:
            terms.append(FO(((i, 1), (i + 1, 0)), -0.01) +
                         FO(((i + 1, 1), (i, 0)), -0.01))

        for i in [0, 1]:
            terms.append(FO(((i, 1), (i + 2, 0)), -0.03) +
                         FO(((i + 2, 1), (i, 0)), -0.03))
            terms.append(FO(((i + 2, 1), (i, 1),
                             (i + 2, 0), (i, 0)), 3.))

        indices = [set([0, 1]), set([2, 3]), set([0, 2]), set([0, 2]),
                   set([1, 3]), set([1, 3])]
        is_hopping_operator = [True, True, True, False, True, False]

        self.assertAlmostEqual(low_depth_second_order_trotter_error_bound(
            terms, indices, is_hopping_operator), 0.0608)

    def test_error_bound_using_info_even_side_length(self):
        # Generate the Hamiltonian.
        hamiltonian = normal_ordered(
            fermi_hubbard(4, 4, 0.5, 0.2, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        self.assertAlmostEqual(low_depth_second_order_trotter_error_bound(
            terms, indices, is_hopping), 13.59)

    def test_error_bound_using_info_odd_side_length_verbose(self):
        # Generate the Hamiltonian.
        hamiltonian = normal_ordered(
            fermi_hubbard(5, 5, -0.5, 0.3, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        self.assertAlmostEqual(32.025,
                               low_depth_second_order_trotter_error_bound(
                                   terms, indices, is_hopping, verbose=True))


class OrderedHubbardTermsMoreInfoTest(unittest.TestCase):
    def test_sum_of_ordered_terms_equals_full_side_length_2_hopping_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(2, 2, 1., 0.0, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        terms_total = sum(terms, FermionOperator.zero())

        self.assertTrue(terms_total == hamiltonian)

    def test_sum_of_ordered_terms_equals_full_hamiltonian_even_side_len(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(4, 4, 10.0, 0.3, periodic=False))
        hamiltonian.compress()

        terms = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)[0]
        terms_total = sum(terms, FermionOperator.zero())

        self.assertTrue(terms_total == hamiltonian)

    def test_sum_of_ordered_terms_equals_full_hamiltonian_odd_side_len(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(5, 5, 1.0, -0.3, periodic=False))
        hamiltonian.compress()

        terms = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)[0]
        terms_total = sum(terms, FermionOperator.zero())

        self.assertTrue(terms_total == hamiltonian)

    def test_correct_indices_terms_with_info(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(5, 5, 1., -1., periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
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
        hamiltonian = normal_ordered(
            fermi_hubbard(5, 5, 1., -1., periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, is_hopping = result

        for i in range(len(terms)):
            single_term = list(terms[i].terms)[0]
            is_hopping_term = not (single_term[1][1] or
                                   single_term[0][0] == single_term[1][0])
            self.assertEqual(is_hopping_term, is_hopping[i])

    def test_total_length_side_length_2_hopping_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(2, 2, 1., 0.0, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 8)

    def test_total_length_odd_side_length_hopping_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(3, 3, 1., 0.0, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 24)

    def test_total_length_even_side_length_hopping_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(4, 4, 1., 0.0, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 48)

    def test_total_length_side_length_2_onsite_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(2, 2, 0.0, 1., periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 4)

    def test_total_length_odd_side_length_onsite_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(3, 3, 0.0, 1., periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 9)

    def test_total_length_even_side_length_onsite_only(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(4, 4, 0., -0.3, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 16)

    def test_total_length_odd_side_length_full_hubbard(self):
        hamiltonian = normal_ordered(
            fermi_hubbard(5, 5, -1., -0.3, periodic=False))
        hamiltonian.compress()

        # Unpack result into terms, indices they act on, and whether they're
        # hopping operators.
        result = simulation_ordered_grouped_hubbard_terms_with_info(
            hamiltonian)
        terms, _, _ = result

        self.assertEqual(len(terms), 105)
