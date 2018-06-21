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

import itertools
import numpy
import numpy.linalg

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import FermionOperator, InteractionOperator
from openfermion.utils import count_qubits, normal_ordered


def get_chemist_ordered_two_body_coefficients(two_body_operator):
    """Convert a FermionOperator or InteractionOperator to low rank tensor.

    In the chemistry convention the Hamiltonian should be expressed as
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    Furthermore, we will require that all symmetries are unpacked as
    :math:`p^\dagger q^\dagger r s =
    (1/4)(-p^\dagger r q^\dagger s + q^\dagger r p^\dagger s
          + p^\dagger s q^\dagger r - q^\dagger s p^\dagger r
          + \delta_{qr} p^\dagger s - \delta_{pr} q^\dagger s
          - \delta_{qs} p^\dagger r + \delta_{ps} q^\dagger r)`

    Args:
        two_body_operator(FermionOperator or InteractionOperator): the term
            to decompose. This operator needs to be a two-body number
            conserving operator. It might also have one-body terms which
            also needs to be number conserving. There might be a constant.

    Returns:
        chemist_ordered_two_body_coefficients (ndarray): an N x N x N x N
            numpy array giving the :math:`h_{pqrs}` tensor in chemist notation
            so that all symmetries are already unpacked.
        one_body_coefficients (ndarray): an N x N array of floats giving
            coefficients of the :math:`a^\dagger_p a_q` terms.
        constant: (float): A constant shift.

    Raises:
        TypeError: Input must be two-body number conserving
            FermionOperator or InteractionOperator.
    """
    # Initialize.
    n_qubits = count_qubits(two_body_operator)
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits), complex)
    chemist_ordered_two_body_coefficients = numpy.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits), complex)

    # Determine input type and ensure normal order.
    if (isinstance(two_body_operator, FermionOperator) and
            two_body_operator.is_two_body_number_conserving()):
        ordered_operator = normal_ordered(fermion_operator)
        is_fermion_operator = True
    elif isinstance(two_body_operator, InteractionOperator):
        is_fermion_operator = False
    else:
        raise TypeError('Input must be two-body number conserving ' +
                        'FermionOperator or InteractionOperator.')

    # Extract explicit one-body terms.
    for p,q in itertools.product(range(n_qubits), 2):
        if is_fermion_operator:
            term = ((p, 1), (q, 0))
            coefficient = ordered_operator.terms.get(term, 0.)
        else:
            coefficient = two_body_operator.one_body_tensor[p, q]
        one_body_coefficients[p, q] += coefficient

    # Loop through and populate two-body coefficient array.
    for p,q,r,s in itertools.product(range(n_qubits), 4):
        if is_fermion_operator:
            term = ((p, 1), (q, 1), (r, 0), (s, 0))
            coefficient = ordered_operator.terms.get(term, 0) / 4.
        else:
            coefficient = two_body_operator.one_body_tensor[p, q]

        # Set two-body elements unpacking symmetries as described in docs.
        chemist_ordered_two_body_coefficients[p, r, q, s] -= coefficient
        chemist_ordered_two_body_coefficients[q, r, p, s] += coefficient
        chemist_ordered_two_body_coefficients[p, s, q, r] += coefficient
        chemist_ordered_two_body_coefficients[q, s, p, r] -= coefficient

        # Account for any one-body terms that might pop out.
        if q == r:
            one_body_coefficients[p, s] += coefficient
        if p == r:
            one_body_coefficients[q, s] -= coefficient
        if q == s:
            one_body_coefficients[p, r] -= coefficient
        if p == s:
            one_body_coefficients[q, r] += coefficient

    return chemist_ordered_two_body_coefficients, one_body_coefficients


def low_rank_two_body_decomposition(chemist_ordered_two_body_coefficients,
                                    truncation_threshold=None):
    """Convert two-body operator into sum of squared one-body operators.

    This function decomposes
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    as :math:`\sum_{l} \lambda_l (\sum_{pq} g_{lpq} a^\dagger_p a_q)^2`
    l is truncated to take max value L so that
    :math:`\sum_{l=0}^{L-1} |\lambda_l| < x`

    Args:
        chemist_ordered_two_body_coefficients (ndarray): an N x N x N x N
            numpy array giving the :math:`h_{pqrs}` tensor in chemist notation
            so that all symmetries are already unpacked.
        truncation_threshold (optional Float): the value of x in the expression
            above. If None, then L = N ** 2 and no truncation will occur.

    Returns:
        eigenvalues (ndarray of floats): length L array giving the g_{lpq}.
        one_body_squares (ndarray of floats): L x N x N array of floats
            corresponding to the value of :math:`\sqrt{\lambda_l} * g_{pql}`.

    Raises:
        TypeError: Invalid two-body coefficient tensor specification.
    """
    # Initialize N^2 by N^2 interaction array.
    n_qubits = chemist_ordered_two_body_coefficients.shape[0]
    interaction_array = numpy.reshape(chemist_ordered_two_body_coefficients,
                                      (n_qubits ** 2, n_qubits ** 2))

    # Make sure interaction array is symmetric.
    asymmetry = numpy.amax(numpy.absolute(
        interaction_array - numpy.transpose(interaction_array)))
    if asymmetry > EQ_TOLERANCE:
        raise TypeError('Invalid two-body coefficient tensor specification.')

    # Diagonalize.
    eigenvalues, eigenvectors = numpy.linalg.eigh(interaction_array)

    # Sort eigenvalues and eigenvectors in descending order by magnitude.
    indices = numpy.argsort(numpy.absolute(eigenvalues))[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Optionally truncate.
    if truncation_threshold is None:
        max_index = n_qubits ** 2
    else:
        cumulative_sum = numpy.cumsum(singular_values)
        truncation_error = cumulative_sum[-1] - cumulative_sum
        max_index = numpy.argmax(truncation_error < truncation_threshold)

    # Return one-body squares.
    one_body_squares = numpy.zeros((max_index, n_qubits, n_qubits))
    for l in range(max_index):
        one_body_squares[l] = numpy.reshape(eigenvectors[:, l],
                                            (n_qubits, n_qubits))

    return eigenvalues[:max_index], one_body_squares

# To delete.
#def low_rank_two_body_decomposition(fermion_operator,
#                                    truncation_threshold=None):
#    """Convert two-body operator into sum of squared one-body operators.
#
#    This function decomposes
#    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a_q a^\dagger_r a_s
#    = \sum_{l} \lambda_l (\sum_{pq} g_{lpq} a^\dagger_p a_q)^2`
#    l is truncated to take max value L so that
#    :math:`\sum_{l=0}^{L-1} |\lambda_l| < x`
#
#    Args:
#        fermion_operator (FermionOperator): The fermion operator to decompose.
#            This operator needs to be a two-body number conserviing operator.
#            It might have one-body terms as well but they will be ignored.
#        truncation_threshold (Float): the value of x in the expression above.
#            If None, then L = N ** 2 and no truncation will occur.
#
#    Returns:
#        singular_values (ndarray of floats): length L array of floats giving
#            the singular values g_{lpq}.
#        one_body_squares (ndarray of floats): L x N x N array of floats
#            corresponding to the value of :math:`\sqrt{\lambda_l} * g_{pql}`.
#        one_body_coefficients (ndarray of floats): N x N array of floats
#            corresponding to left over one-body terms (not squared).
#    """
#    # Obtain chemist ordered FermionOperator.
#    ordered_operator = chemist_ordered(fermion_operator)
#
#    # Initialize (pq|rs) array.
#    n_qubits = count_qubits(ordered_operator)
#    one_body_coefficients = numpy.zeros((n_qubits, n_qubits), complex)
#    interaction_array = numpy.zeros((n_qubits ** 2, n_qubits ** 2), complex)
#
#    # Populate interaction array.
#    for p in range(n_qubits):
#        for q in range(n_qubits):
#            for r in range(n_qubits):
#                for s in range(n_qubits):
#                    x = p + n_qubits * q
#                    y = r + n_qubits * s
#                    interaction_array[x, y] = ordered_operator.terms.get(
#                        ((p, 1), (q, 0), (r, 1), (s, 0)), 0.)
#    interaction_array = (interaction_array +
#                         numpy.conjugate(numpy.transpose(interaction_array)))
#    interaction_array /= 2.
#
#    # Diagonalize.
#    eigenvalues, eigenvectors = numpy.linalg.eigh(interaction_array)
#    negative_eigenvalues = (eigenvalues < 0)
#    eigenvalues = numpy.absolute(eigenvalues)
#    indices = numpy.argsort(-eigenvalues)
#    eigenvalues = eigenvalues[indices]
#    eigenvectors = eigenvectors[:, indices]
#    negative_eigenvalues = negative_eigenvalues[indices]
#
#    # Determine where to truncate.
#    if truncation_threshold is None:
#        max_index = n_qubits ** 2
#    else:
#        cumulative_sum = numpy.cumsum(singular_values)
#        truncation_error = cumulative_sum[-1] - cumulative_sum
#        max_index = numpy.argmax(truncation_error < truncation_threshold) + 1
#
#    # Return one-body squares.
#    one_body_squares = numpy.zeros((max_index, n_qubits, n_qubits), complex)
#    for l in range(max_index):
#        eigenvector = numpy.sqrt(eigenvalues[l]) * eigenvectors[:, l]
#        if negative_eigenvalues[l]:
#            eigenvector = 1.j * eigenvector
#        for p in range(n_qubits):
#            for q in range(n_qubits):
#                linear_index = p + n_qubits * q
#                one_body_squares[l, p, q] = complex(eigenvector[linear_index])
#
#    return one_body_squares
