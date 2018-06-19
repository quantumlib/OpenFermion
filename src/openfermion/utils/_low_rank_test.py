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
                               low_rank_two_body_decomposition,
                               random_interaction_operator)


class LowRankTest(unittest.TestCase):

    def test_consistency(self):

        # Initialize an operator that is just a two-body operator.
        n_qubits = 6
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
                    one_body_operator = FermionOperator(
                        term, one_body_squares[l, p, q])
            decomposed_operator += one_body_operator ** 2

        # Test for consistency.
        difference = decomposed_operator - random_operator
        self.assertAlmostEqual(0., difference.induced_norm())
