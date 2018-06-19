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

"""This module provides tools for simulating rank deficient operators."""

import numpy
import numpy.linalg

from openfermion.utils import chemist_ordered


def low_rank_two_body_decomposition(fermion_operator,
                                    truncation_threshold=None):
    """Convert two-body operator into sum of squared one-body operators.

    This function decomposes
    :math: \sum_{pqrs} h_{pqrs} a^\dagger_p a_q a^\dagger_r a_s
    = \sum_{l} (\sum_{pq} g_{lpq} a^\dagger_p a_q)^2
    l is truncated to take max value L so that
    :math: \sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 < x

    Args:
        fermion_operator (FermionOperator): The fermion operator to decompose.
            This operator needs to be a two-body number conserviing operator.
            It might have one-body terms as well but they will be ignored.
        truncation_threshold (Float): the value of x in the expression above.
            If None, then no truncation will occur.

    Returns:
        one_body_squares (ndarray of floats): L x N x N array of floats
            corresponding to the g_{pql} values.
    """
    # Obtain chemist ordered FermionOperator.
    ordered_operator = chemist_ordered(fermion_operator)

    # Initialize (pq|rs) array.
    n_qubits = count_qubits(ordered_operator)
    interaction_array = numpy.zeros((
    for p in range(

