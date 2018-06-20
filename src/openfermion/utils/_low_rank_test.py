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

import numpy
import unittest

from openfermion.ops import FermionOperator
from openfermion.transforms import get_fermion_operator
from openfermion.utils import (chemist_ordered,
                               is_hermitian,
                               low_rank_two_body_decomposition,
                               normal_ordered,
                               random_interaction_operator)


class LowRankTest(unittest.TestCase):

    def test_matrix_consistency(self):

        # Initialize an operator that is just a two-body operator.
        n_qubits = 4
        random_operator = chemist_ordered(get_fermion_operator(
            random_interaction_operator(n_qubits)))
        for term, coefficient in random_operator.terms.items():
            if len(term) != 4:
                del random_operator.terms[term]

        # Initialize (pq|rs) array.
        interaction_array = numpy.zeros((n_qubits ** 2, n_qubits ** 2), float)

        # Populate interaction array.
        for p in range(n_qubits):
            for q in range(n_qubits):
                for r in range(n_qubits):
                    for s in range(n_qubits):
                        x = p + n_qubits * q
                        y = r + n_qubits * s
                        interaction_array[x, y] = random_operator.terms.get(
                            ((p, 1), (q, 0), (r, 1), (s, 0)), 0.)
        interaction_array = (interaction_array
                             + numpy.transpose(interaction_array)) / 2.

        # Perform low rank decomposition and build operator back.
        one_body_squares = low_rank_two_body_decomposition(random_operator)
        l_max = one_body_squares.shape[0]
        test_matrix = numpy.zeros((n_qubits ** 2, n_qubits ** 2), complex)
        for l in range(l_max):
            vector = one_body_squares[l].reshape((n_qubits ** 2, 1))
            test_matrix += numpy.dot(vector, vector.transpose())
        difference = numpy.sum(numpy.absolute(test_matrix - interaction_array))
        self.assertAlmostEqual(0., difference)


    def test_fermion_operator_consistency(self):

        # Initialize an operator that is just a two-body operator.
        n_qubits = 3
        random_operator = chemist_ordered(get_fermion_operator(
            random_interaction_operator(n_qubits)))
        for term, coefficient in random_operator.terms.items():
            if len(term) != 4:
                del random_operator.terms[term]

        # Perform low rank decomposition and build operator back.
        one_body_squares = low_rank_two_body_decomposition(random_operator)
        l_max, n_qubits = one_body_squares.shape[:2]
        decomposed_operator = FermionOperator()
        for l in range(l_max):
            one_body_operator = FermionOperator()
            for p in range(n_qubits):
                for q in range(n_qubits):
                    term = ((p, 1), (q, 0))
                    one_body_operator += FermionOperator(
                        term, one_body_squares[l, p, q])
            decomposed_operator += one_body_operator ** 2

        # Test for consistency.
        difference = normal_ordered(decomposed_operator - random_operator)
        self.assertAlmostEqual(0., difference.induced_norm())
