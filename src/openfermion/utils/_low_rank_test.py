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

import itertools
import numpy
import unittest

from openfermion.ops import FermionOperator
from openfermion.transforms import get_fermion_operator
from openfermion.utils import (chemist_ordered,
                               get_chemist_two_body_coefficients,
                               is_hermitian,
                               low_rank_two_body_decomposition,
                               normal_ordered,
                               random_interaction_operator)


class ChemistTwoBodyTest(unittest.TestCase):

    def test_operator_consistency(self):

        # Initialize a random InteractionOperator and FermionOperator.
        n_qubits = 4
        random_interaction = random_interaction_operator(n_qubits)
        random_fermion = get_fermion_operator(random_interaction)

        # Convert to chemist ordered tensor.
        io_constant, io_one_body_coefficients, io_chemist_tensor = \
            get_chemist_two_body_coefficients(random_interaction)
        fo_constant, fo_one_body_coefficients, fo_chemist_tensor = \
            get_chemist_two_body_coefficients(random_fermion)

        # Ensure consistency between FermionOperator and InteractionOperator.
        self.assertAlmostEqual(io_constant, fo_constant)

        one_body_difference = numpy.sum(numpy.absolute(
            io_one_body_coefficients - fo_one_body_coefficients))
        self.assertAlmostEqual(0., one_body_difference)

        two_body_difference = numpy.sum(numpy.absolute(
            io_chemist_tensor - fo_chemist_tensor))
        self.assertAlmostEqual(0., two_body_difference)

        # Convert output to FermionOperator.
        output_operator = FermionOperator()
        output_operator += FermionOperator((), fo_constant)

        # Convert one-body.
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = fo_one_body_coefficients[p, q]
            output_operator += FermionOperator(term, coefficient)

        # Convert two-body.
        for p, q, r, s in itertools.product(range(n_qubits), repeat=4):
            term = ((p, 1), (q, 0), (r, 1), (s, 0))
            coefficient = fo_chemist_tensor[p, q, r, s]
            output_operator += FermionOperator(term, coefficient)

        # Check that difference is small.
        difference = normal_ordered(random_fermion - output_operator)
        self.assertAlmostEqual(0., difference.induced_norm())

    def test_exception(self):

        # Initialize a bad FermionOperator.
        n_qubits = 4
        random_interaction = random_interaction_operator(n_qubits)
        random_fermion = get_fermion_operator(random_interaction)
        bad_term = ((1, 1), (2, 1))
        random_fermion += FermionOperator(bad_term)

        # Check for exception.
        with self.assertRaises(TypeError):
            fo_constant, fo_one_body_coefficients, fo_chemist_tensor = \
                get_chemist_two_body_coefficients(random_fermion)


class LowRankTest(unittest.TestCase):

    def test_operator_consistency(self):

        # Initialize a random two-body FermionOperator.
        n_qubits = 4
        random_operator = get_fermion_operator(
            random_interaction_operator(n_qubits))

        # Convert to chemist tensor.
        constant, one_body_coefficients, chemist_tensor = \
            get_chemist_two_body_coefficients(random_operator)

        # Build back operator constant and one-body components.
        decomposed_operator = FermionOperator((), constant)
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = one_body_coefficients[p, q]
            decomposed_operator += FermionOperator(term, coefficient)

        # Perform decomposition.
        eigenvalues, one_body_squares = low_rank_two_body_decomposition(
            chemist_tensor)

        # Build back two-body component.
        for l in range(n_qubits ** 2):
            one_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_squares[l, p, q]
                one_body_operator += FermionOperator(term, coefficient)
            decomposed_operator += eigenvalues[l] * (one_body_operator ** 2)

        # Test for consistency.
        difference = normal_ordered(decomposed_operator - random_operator)
        self.assertAlmostEqual(0., difference.induced_norm())
