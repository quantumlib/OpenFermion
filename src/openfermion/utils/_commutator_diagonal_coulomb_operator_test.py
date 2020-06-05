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

"""Tests for _commutator_diagonal_coulomb_operator.py"""

import unittest
import warnings

from openfermion import FermionOperator, Grid, jellium_model, normal_ordered
from openfermion.transforms import get_fermion_operator
from openfermion.utils import commutator
from openfermion.utils._commutator_diagonal_coulomb_operator import (
    commutator_ordered_diagonal_coulomb_with_two_body_operator)
from openfermion.utils._testing_utils import (
    random_diagonal_coulomb_hamiltonian)


class DiagonalHamiltonianCommutatorTest(unittest.TestCase):
    def test_commutator(self):
        operator_a = (FermionOperator('0^ 0', 0.3) +
                      FermionOperator('1^ 1', 0.1j) +
                      FermionOperator('1^ 0^ 1 0', -0.2) +
                      FermionOperator('1^ 3') + FermionOperator('3^ 0') +
                      FermionOperator('3^ 2', 0.017) -
                      FermionOperator('2^ 3', 1.99) +
                      FermionOperator('3^ 1^ 3 1', .09) +
                      FermionOperator('2^ 0^ 2 0', .126j) +
                      FermionOperator('4^ 2^ 4 2') +
                      FermionOperator('3^ 0^ 3 0'))

        operator_b = (FermionOperator('3^ 1', 0.7) +
                      FermionOperator('1^ 3', -9.) +
                      FermionOperator('1^ 0^ 3 0', 0.1) -
                      FermionOperator('3^ 0^ 1 0', 0.11) +
                      FermionOperator('3^ 2^ 3 2') +
                      FermionOperator('3^ 1^ 3 1', -1.37) +
                      FermionOperator('4^ 2^ 4 2') +
                      FermionOperator('4^ 1^ 4 1') +
                      FermionOperator('1^ 0^ 4 0', 16.7) +
                      FermionOperator('1^ 0^ 4 3', 1.67) +
                      FermionOperator('4^ 3^ 5 2', 1.789j) +
                      FermionOperator('6^ 5^ 4 1', -11.789j))

        reference = normal_ordered(commutator(operator_a, operator_b))
        result = commutator_ordered_diagonal_coulomb_with_two_body_operator(
            operator_a, operator_b)

        diff = result - reference
        self.assertTrue(diff.isclose(FermionOperator.zero()))

    def test_nonstandard_second_arg(self):
        operator_a = (FermionOperator('0^ 0', 0.3) +
                      FermionOperator('1^ 1', 0.1j) +
                      FermionOperator('2^ 0^ 2 0', -0.2) +
                      FermionOperator('2^ 1^ 2 1', -0.2j) +
                      FermionOperator('1^ 3') + FermionOperator('3^ 0') +
                      FermionOperator('4^ 4', -1.4j))

        operator_b = (FermionOperator('4^ 1^ 3 0', 0.1) -
                      FermionOperator('3^ 0^ 1 0', 0.11))

        reference = (FermionOperator('1^ 0^ 1 0', -0.11) +
                     FermionOperator('3^ 0^ 1 0', 0.011j) +
                     FermionOperator('3^ 0^ 3 0', 0.11) +
                     FermionOperator('3^ 2^ 0^ 2 1 0', -0.022j) +
                     FermionOperator('4^ 1^ 3 0', -0.03-0.13j) +
                     FermionOperator('4^ 2^ 1^ 3 2 0', -0.02+0.02j))

        res = commutator_ordered_diagonal_coulomb_with_two_body_operator(
            operator_a, operator_b)

        self.assertTrue(res.isclose(reference))

    def test_add_to_existing_result(self):
        prior_terms = FermionOperator('0^ 1')
        operator_a = FermionOperator('2^ 1')
        operator_b = FermionOperator('0^ 2')

        commutator_ordered_diagonal_coulomb_with_two_body_operator(
            operator_a, operator_b, prior_terms=prior_terms)

        self.assertTrue(prior_terms.isclose(FermionOperator.zero()))

    def test_integration_jellium_hamiltonian_with_negation(self):
        hamiltonian = normal_ordered(
            jellium_model(Grid(2, 3, 1.), plane_wave=False))

        part_a = FermionOperator.zero()
        part_b = FermionOperator.zero()

        add_to_a_or_b = 0  # add to a if 0; add to b if 1
        for term, coeff in hamiltonian.terms.items():
            # Partition terms in the Hamiltonian into part_a or part_b
            if add_to_a_or_b:
                part_a += FermionOperator(term, coeff)
            else:
                part_b += FermionOperator(term, coeff)
            add_to_a_or_b ^= 1

        reference = normal_ordered(commutator(part_a, part_b))
        result = commutator_ordered_diagonal_coulomb_with_two_body_operator(
            part_a, part_b)

        self.assertTrue(result.isclose(reference))

        negative = commutator_ordered_diagonal_coulomb_with_two_body_operator(
            part_b, part_a)
        result += negative

        self.assertTrue(result.isclose(FermionOperator.zero()))

    def test_no_warning_on_nonstandard_input_second_arg(self):
        with warnings.catch_warnings(record=True) as w:
            operator_a = FermionOperator('3^ 2^ 3 2')
            operator_b = FermionOperator('4^ 3^ 4 1')

            reference = FermionOperator('4^ 3^ 2^ 4 2 1')
            result = (
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    operator_a, operator_b))

            self.assertFalse(w)

            # Result should still be correct even though we hit the warning.
            self.assertTrue(result.isclose(reference))

    def test_warning_on_bad_input_first_arg(self):
        with warnings.catch_warnings(record=True) as w:
            operator_a = FermionOperator('4^ 3^ 2 1')
            operator_b = FermionOperator('3^ 2^ 3 2')

            reference = normal_ordered(commutator(operator_a, operator_b))
            result = (
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    operator_a, operator_b))

            self.assertTrue(len(w) == 1)
            self.assertIn('Defaulted to standard commutator evaluation',
                          str(w[-1].message))

            # Result should still be correct in this case.
            diff = result - reference
            self.assertTrue(diff.isclose(FermionOperator.zero()))

    def test_integration_random_diagonal_coulomb_hamiltonian(self):
        hamiltonian1 = normal_ordered(get_fermion_operator(
            random_diagonal_coulomb_hamiltonian(n_qubits=7)))
        hamiltonian2 = normal_ordered(get_fermion_operator(
            random_diagonal_coulomb_hamiltonian(n_qubits=7)))

        reference = normal_ordered(commutator(hamiltonian1, hamiltonian2))
        result = commutator_ordered_diagonal_coulomb_with_two_body_operator(
            hamiltonian1, hamiltonian2)

        self.assertTrue(result.isclose(reference))
