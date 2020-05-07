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

"""Tests for _diagonal_coulomb_trotter_error.py."""

import unittest

import numpy

from openfermion import (
    commutator,
    count_qubits,
    DiagonalCoulombHamiltonian,
    FermionOperator,
    Grid,
    normal_ordered)
from openfermion.hamiltonians import (
    dual_basis_jellium_model,
    fermi_hubbard,
    hypercube_grid_with_given_wigner_seitz_radius_and_filling,
    jellium_model)
from openfermion.utils._low_depth_trotter_error import (
    low_depth_second_order_trotter_error_bound,
    low_depth_second_order_trotter_error_operator,
    simulation_ordered_grouped_low_depth_terms_with_info)
from openfermion.utils._diagonal_coulomb_trotter_error import (
    diagonal_coulomb_potential_and_kinetic_terms_as_arrays,
    bit_mask_of_modes_acted_on_by_fermionic_terms,
    split_operator_trotter_error_operator_diagonal_two_body,
    fermionic_swap_trotter_error_operator_diagonal_two_body)


class BreakHamiltonianIntoPotentialKineticArraysTest(unittest.TestCase):

    def test_simple_hamiltonian(self):
        hamiltonian = (FermionOperator('3^ 1^ 3 1') +
                       FermionOperator('1^ 1') - FermionOperator('1^ 2') -
                       FermionOperator('2^ 1'))

        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                hamiltonian))

        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        self.assertEqual(potential, (FermionOperator('1^ 1') +
                                     FermionOperator('3^ 1^ 3 1')))
        self.assertEqual(kinetic, (-FermionOperator('1^ 2') -
                                   FermionOperator('2^ 1')))

    def test_jellium_hamiltonian_correctly_broken_up(self):
        grid = Grid(2, 3, 1.)

        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)

        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                hamiltonian))

        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        true_potential = dual_basis_jellium_model(grid, spinless=True,
                                                  kinetic=False)
        true_kinetic = dual_basis_jellium_model(grid, spinless=True,
                                                potential=False)
        for i in range(count_qubits(true_kinetic)):
            coeff = true_kinetic.terms.get(((i, 1), (i, 0)))
            if coeff:
                true_kinetic -= FermionOperator(((i, 1), (i, 0)), coeff)
                true_potential += FermionOperator(((i, 1), (i, 0)), coeff)

        self.assertEqual(potential, true_potential)
        self.assertEqual(kinetic, true_kinetic)

    def test_identity_recognized_as_potential_term(self):
        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                FermionOperator.identity()))

        self.assertListEqual(list(potential_terms),
                             [FermionOperator.identity()])
        self.assertListEqual(list(kinetic_terms), [])

    def test_zero_hamiltonian(self):
        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                FermionOperator.zero()))

        self.assertListEqual(list(potential_terms), [])
        self.assertListEqual(list(kinetic_terms), [])

    def test_diagonal_coulomb_hamiltonian_class(self):
        hamiltonian = DiagonalCoulombHamiltonian(
            numpy.array([[1, 1], [1, 1]], dtype=float),
            numpy.array([[0, 1], [1, 0]], dtype=float),
            constant=2.3)

        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                hamiltonian))

        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        expected_potential = (2.3 * FermionOperator.identity() +
                              FermionOperator('0^ 0') +
                              FermionOperator('1^ 1') -
                              FermionOperator('1^ 0^ 1 0', 2.0))
        expected_kinetic = FermionOperator('0^ 1') + FermionOperator('1^ 0')

        self.assertEqual(potential, expected_potential)
        self.assertEqual(kinetic, expected_kinetic)

    def test_type_error_on_bad_input_hamiltonian(self):
        with self.assertRaises(TypeError):
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays('oops')


class BitMaskModesActedOnByFermionTermsTest(unittest.TestCase):
    def test_mask_no_terms(self):
        mask = bit_mask_of_modes_acted_on_by_fermionic_terms([], n_qubits=2)
        self.assertTrue(numpy.array_equal(mask, numpy.array([[], []])))

    def test_identity_masks_no_modes(self):
        mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
            [FermionOperator.zero()], n_qubits=3)

        self.assertTrue(numpy.array_equal(mask, numpy.zeros((3, 1))))

    def test_mask_single_term(self):
        mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
            [FermionOperator('0^ 0')], n_qubits=2)
        self.assertTrue(numpy.array_equal(mask,
                                          numpy.array([[True], [False]])))

    def test_mask_hermitian_terms_results_in_duplicated_row(self):
        mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
            [FermionOperator('2^ 3'), FermionOperator('3^ 2')], n_qubits=5)

        expected_mask = numpy.array([[False, False],
                                     [False, False],
                                     [True, True],
                                     [True, True],
                                     [False, False]])

        self.assertTrue(numpy.array_equal(mask, expected_mask))

    def test_mask_n_qubits_too_small_for_term(self):
        with self.assertRaises(ValueError):
            bit_mask_of_modes_acted_on_by_fermionic_terms(
                [FermionOperator('1^ 1')], n_qubits=1)

    def test_mask_long_arbitrary_terms(self):
        operator1 = FermionOperator('6^ 5^ 4^ 3 2 1', 2.3 - 1.7j)
        operator2 = FermionOperator('7^ 5^ 1^ 0^', 0.)

        mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
            [operator1, operator2], n_qubits=8)

        expected_mask = numpy.array([[False, True],
                                    [True, True],
                                    [True, False],
                                    [True, False],
                                    [True, False],
                                    [True, True],
                                    [True, False],
                                    [False, True]])

        self.assertTrue(numpy.array_equal(mask, expected_mask))

    def test_bit_mask_n_qubits_not_specified(self):
        mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
            [FermionOperator('0^ 0') + FermionOperator('2^ 2')])

        self.assertTrue(numpy.array_equal(mask, numpy.array(
            [[True], [False], [True]])))


class FermionicSwapNetworkTrotterErrorTest(unittest.TestCase):

    def test_1D_jellium_trotter_error_matches_low_depth_trotter_error(self):
        hamiltonian = normal_ordered(jellium_model(
            hypercube_grid_with_given_wigner_seitz_radius_and_filling(
                1, 5, wigner_seitz_radius=10.,
                spinless=True), spinless=True, plane_wave=False))

        error_operator = (
            fermionic_swap_trotter_error_operator_diagonal_two_body(
                hamiltonian))
        error_operator.compress()

        # Unpack result into terms, indices they act on, and whether
        # they're hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        old_error_operator = low_depth_second_order_trotter_error_operator(
            terms, indices, is_hopping, jellium_only=True)

        old_error_operator -= error_operator
        self.assertEqual(old_error_operator, FermionOperator.zero())

    def test_hubbard_trotter_error_matches_low_depth_trotter_error(self):
        hamiltonian = normal_ordered(fermi_hubbard(3, 3, 1., 2.3))

        error_operator = (
            fermionic_swap_trotter_error_operator_diagonal_two_body(
                hamiltonian))
        error_operator.compress()

        # Unpack result into terms, indices they act on, and whether
        # they're hopping operators.
        result = simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian)
        terms, indices, is_hopping = result

        old_error_operator = low_depth_second_order_trotter_error_operator(
            terms, indices, is_hopping, jellium_only=True)

        old_error_operator -= error_operator
        self.assertEqual(old_error_operator, FermionOperator.zero())


class SplitOperatorTrotterErrorTest(unittest.TestCase):

    def test_split_operator_error_operator_TV_order_against_definition(self):
        hamiltonian = (normal_ordered(fermi_hubbard(3, 3, 1., 4.0)) -
                       2.3 * FermionOperator.identity())
        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                hamiltonian))
        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='T+V'))

        # T-then-V ordered double commutators: [T, [V, T]] + [V, [V, T]] / 2
        inner_commutator = normal_ordered(commutator(potential, kinetic))
        error_operator_definition = normal_ordered(
            commutator(kinetic, inner_commutator))
        error_operator_definition += normal_ordered(
            commutator(potential, inner_commutator)) / 2.0
        error_operator_definition /= 12.0

        self.assertEqual(error_operator, error_operator_definition)

    def test_split_operator_error_operator_VT_order_against_definition(self):
        hamiltonian = (normal_ordered(fermi_hubbard(3, 3, 1., 4.0)) -
                       2.3 * FermionOperator.identity())
        potential_terms, kinetic_terms = (
            diagonal_coulomb_potential_and_kinetic_terms_as_arrays(
                hamiltonian))
        potential = sum(potential_terms, FermionOperator.zero())
        kinetic = sum(kinetic_terms, FermionOperator.zero())

        error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='V+T'))

        # V-then-T ordered double commutators: [V, [T, V]] + [T, [T, V]] / 2
        inner_commutator = normal_ordered(commutator(kinetic, potential))
        error_operator_definition = normal_ordered(
            commutator(potential, inner_commutator))
        error_operator_definition += normal_ordered(
            commutator(kinetic, inner_commutator)) / 2.0
        error_operator_definition /= 12.0

        self.assertEqual(error_operator, error_operator_definition)

    def test_intermediate_interaction_hubbard_TV_order_has_larger_error(self):
        hamiltonian = normal_ordered(fermi_hubbard(4, 4, 1., 4.0))

        TV_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='T+V'))
        TV_error_bound = numpy.sum(numpy.absolute(
            list(TV_error_operator.terms.values())))

        VT_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='V+T'))
        VT_error_bound = numpy.sum(numpy.absolute(
            list(VT_error_operator.terms.values())))

        self.assertAlmostEqual(TV_error_bound, 1706.66666666666)
        self.assertAlmostEqual(VT_error_bound, 1365.33333333333)

    def test_strong_interaction_hubbard_VT_order_gives_larger_error(self):
        hamiltonian = normal_ordered(fermi_hubbard(4, 4, 1., 10.0))

        TV_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='T+V'))
        TV_error_bound = numpy.sum(numpy.absolute(
            list(TV_error_operator.terms.values())))

        VT_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='V+T'))
        VT_error_bound = numpy.sum(numpy.absolute(
            list(VT_error_operator.terms.values())))

        self.assertGreater(VT_error_bound, TV_error_bound)

    def test_jellium_wigner_seitz_10_VT_order_gives_larger_error(self):
        hamiltonian = normal_ordered(jellium_model(
            hypercube_grid_with_given_wigner_seitz_radius_and_filling(
                2, 3, wigner_seitz_radius=10.,
                spinless=True), spinless=True, plane_wave=False))

        TV_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='T+V'))
        TV_error_bound = numpy.sum(numpy.absolute(
            list(TV_error_operator.terms.values())))

        VT_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='V+T'))
        VT_error_bound = numpy.sum(numpy.absolute(
            list(VT_error_operator.terms.values())))

        self.assertGreater(VT_error_bound, TV_error_bound)

    def test_1d_jellium_wigner_seitz_10_VT_order_gives_larger_error(self):
        hamiltonian = normal_ordered(jellium_model(
            hypercube_grid_with_given_wigner_seitz_radius_and_filling(
                1, 5, wigner_seitz_radius=10.,
                spinless=True), spinless=True, plane_wave=False))

        TV_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='T+V'))
        TV_error_bound = numpy.sum(numpy.absolute(
            list(TV_error_operator.terms.values())))

        VT_error_operator = (
            split_operator_trotter_error_operator_diagonal_two_body(
                hamiltonian, order='V+T'))
        VT_error_bound = numpy.sum(numpy.absolute(
            list(VT_error_operator.terms.values())))

        self.assertGreater(VT_error_bound, TV_error_bound)
