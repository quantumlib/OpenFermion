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

from openfermion.utils import chemist_ordered, count_qubits


def low_rank_two_body_decomposition(fermion_operator,
                                    truncation_threshold=None):
    """Convert two-body operator into sum of squared one-body operators.

    This function decomposes
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a_q a^\dagger_r a_s
    = \sum_{l} \lambda_l (\sum_{pq} g_{lpq} a^\dagger_p a_q)^2`
    l is truncated to take max value L so that
    :math:`\sum_{l=0}^{L-1} |\lambda_l| < x`

    Args:
        fermion_operator (FermionOperator): The fermion operator to decompose.
            This operator needs to be a two-body number conserviing operator.
            It might have one-body terms as well but they will be ignored.
        truncation_threshold (Float): the value of x in the expression above.
            If None, then L = N ** 2 and no truncation will occur.

    Returns:
        one_body_squares (ndarray of floats): L x N x N array of floats
            corresponding to the value of :math:`\sqrt{\lambda_l} * g_{pql}`.
    """
    # Obtain chemist ordered FermionOperator.
    ordered_operator = chemist_ordered(fermion_operator)

    # Initialize (pq|rs) array.
    n_qubits = count_qubits(ordered_operator)
    interaction_array = numpy.zeros((n_qubits ** 2, n_qubits ** 2), float)

    # Populate interaction array.
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    x = p + n_qubits * q
                    y = r + n_qubits * s
                    interaction_array(x, y) = ordered_operator.terms.get(
                        ((p, 1), (q, 0), (r, 1), (s, 0)), 0.)

    # Perform the SVD.
    left_vectors, singular_values, right_vectors = numpy.linalg.svd(
        interaction_array)

    # Determine where to truncate.
    if truncation_threshold is None:
        max_index = n_qubits ** 2
    else:
        cumulative_sum = numpy.cumsum(singular_values)
        truncation_error = cumulative_sum[-1] - cumulative_sum
        max_index = numpy.argmax(truncation_error < truncation_threshold) + 1

    # Return one-body squares.
    one_body_squares = numpy.zeros((max_index, n_qubits, n_qubits), float)
    for l in range(max_index):
        left_vector = left_vectors[l]
        for p in range(n_qubits):
            for q in range(n_qubits):
                linear_index = p + n_qubits * q
                one_body_squares[l, p, q] = (numpy.sqrt(singular_values[l]) *
                                             left_vector[linear_index])
    return one_body_squares
