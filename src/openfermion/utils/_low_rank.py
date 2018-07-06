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


def get_chemist_two_body_coefficients(two_body_operator):
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
            also needs to be number conserving. There could also
            be a constant in the InteractionOperator or FermionOperator.

    Returns:
        constant: (float): A constant shift.
        one_body_coefficients (ndarray): an N x N array of floats giving
            coefficients of the :math:`a^\dagger_p a_q` terms.
        chemist_two_body_coefficients (ndarray): an N x N x N x N numpy array
            giving the :math:`h_{pqrs}` tensor in chemist notation so that all
            symmetries are already unpacked.

    Raises:
        TypeError: Input must be two-body number conserving
            FermionOperator or InteractionOperator.
    """
    # Initialize.
    n_qubits = count_qubits(two_body_operator)
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits), complex)
    chemist_two_body_coefficients = numpy.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits), complex)

    # Determine input type and ensure normal order.
    if (isinstance(two_body_operator, FermionOperator) and
            two_body_operator.is_two_body_number_conserving()):
        ordered_operator = normal_ordered(two_body_operator)
        constant = ordered_operator.terms.get((), 0.)
        is_fermion_operator = True
    elif isinstance(two_body_operator, InteractionOperator):
        constant = two_body_operator.constant
        is_fermion_operator = False
    else:
        raise TypeError('Input must be two-body number conserving ' +
                        'FermionOperator or InteractionOperator.')

    # Extract explicit one-body terms.
    for p, q in itertools.product(range(n_qubits), repeat=2):
        if is_fermion_operator:
            term = ((p, 1), (q, 0))
            coefficient = ordered_operator.terms.get(term, 0.)
        else:
            coefficient = two_body_operator.one_body_tensor[p, q]
        one_body_coefficients[p, q] += coefficient

    # Loop through and populate two-body coefficient array.
    for p, q, r, s in itertools.product(range(n_qubits), repeat=4):
        if p == q or r == s:
            continue
        elif is_fermion_operator:
            term = ((p, 1), (q, 1), (r, 0), (s, 0))
            coefficient = ordered_operator.terms.get(term, 0) / 4.
        else:
            coefficient = two_body_operator.two_body_tensor[p, q, r, s] / 4.

        # Set two-body elements unpacking symmetries as described in docs.
        chemist_two_body_coefficients[p, r, q, s] -= coefficient
        chemist_two_body_coefficients[q, r, p, s] += coefficient
        chemist_two_body_coefficients[p, s, q, r] += coefficient
        chemist_two_body_coefficients[q, s, p, r] -= coefficient

        # Account for any one-body terms that might pop out.
        if q == r:
            one_body_coefficients[p, s] += coefficient
        if p == r:
            one_body_coefficients[q, s] -= coefficient
        if q == s:
            one_body_coefficients[p, r] -= coefficient
        if p == s:
            one_body_coefficients[q, r] += coefficient

    return (constant, one_body_coefficients,
            chemist_two_body_coefficients)


def low_rank_two_body_decomposition(chemist_two_body_coefficients,
                                    truncation_threshold=None,
                                    final_rank=None):
    """Convert two-body operator into sum of squared one-body operators.

    This function decomposes
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    as :math:`\sum_{l} \lambda_l (\sum_{pq} g_{lpq} a^\dagger_p a_q)^2`
    l is truncated to take max value L so that
    :math:`\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x`

    Args:
        chemist_two_body_coefficients (ndarray): an N x N x N x N
            numpy array giving the :math:`h_{pqrs}` tensor in chemist notation
            so that all symmetries are already unpacked. At this point we will
            also require that the matrix is strictly real.
        truncation_threshold (optional Float): the value of x in the expression
            above. If None, then L = N ** 2 and no truncation will occur.
        final_rank (optional int): if provided, this specifies the value of
            L at which to truncate.

    Returns:
        eigenvalues (ndarray of floats): length L array
            giving the :math:`\lambda_l`.
        one_body_squares (ndarray of floats): L x N x N array of floats
            corresponding to the value of :math:`g_{pql}`.
        truncation_value (optional float): after truncation, this is the value
            :math:`\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x`

    Raises:
        TypeError: Invalid two-body coefficient tensor specification.
        ValueError: Cannot provide both final_rank and truncation_value.
    """
    # Initialize N^2 by N^2 interaction array.
    n_qubits = chemist_two_body_coefficients.shape[0]
    full_rank = n_qubits ** 2
    interaction_array = numpy.reshape(chemist_two_body_coefficients,
                                      (full_rank, full_rank))

    # Make sure interaction array is symmetric and real.
    asymmetry = numpy.sum(numpy.absolute(
        interaction_array - interaction_array.transpose()))
    imaginary_norm = numpy.sum(numpy.absolute(interaction_array.imag))
    if asymmetry > EQ_TOLERANCE or imaginary_norm > EQ_TOLERANCE:
        raise TypeError('Invalid two-body coefficient tensor specification.')

    # Diagonalize.
    eigenvalues, eigenvectors = numpy.linalg.eigh(interaction_array)

    # Get one-body squares and compute weights.
    term_weights = numpy.zeros(full_rank)
    one_body_squares = numpy.zeros((full_rank, n_qubits, n_qubits), complex)
    for l in range(full_rank):
        one_body_squares[l] = numpy.reshape(eigenvectors[:, l],
                                            (n_qubits, n_qubits))
        term_weights[l] = abs(eigenvalues[l]) * numpy.sum(
            numpy.absolute(one_body_squares[l])) ** 2

    # Sort by weight.
    indices = numpy.argsort(term_weights)[::-1]
    eigenvalues = eigenvalues[indices]
    term_weights = term_weights[indices]
    one_body_squares = one_body_squares[indices]

    # Determine upper-bound on truncation errors that would occur.
    cumulative_error_sum = numpy.cumsum(term_weights)
    truncation_errors = cumulative_error_sum[-1] - cumulative_error_sum

    # Optionally truncate rank and return.
    if truncation_threshold is None and final_rank is None:
        max_rank = full_rank
    elif truncation_threshold is None:
        max_rank = final_rank
    elif final_rank is None:
        max_rank = 1 + numpy.argmax(truncation_errors <= truncation_threshold)
    else:
        raise ValueError(
            'Cannot provide both final_rank and truncation_value.')
    truncation_value = truncation_errors[max_rank - 1]
    return (eigenvalues[:max_rank],
            one_body_squares[:max_rank],
            truncation_value)


def prepare_one_body_squared_evolution(one_body_matrix):
    """Get Givens angles and DiagonalHamiltonian to simulate squared one-body.

    The goal here will be to prepare to simulate evolution under
    :math:`(\sum_{pq} h_{pq} a^\dagger_p a_q)^2` by decomposing as
    :math:`R e^{-i \sum_{pq} V_{pq} n_p n_q} R^\dagger' where
    :math:`R` is a basis transformation matrix.

    TODO: Add option for truncation based on one-body eigenvalues.

    Args:
        one_body_matrix (ndarray of floats): an N by N array storing the
            coefficients of a one-body operator to be squared. For instance,
            in the above the elements of this matrix are :math:`h_{pq}`.

    Returns:
        density_density_matrix(ndarray of floats) an N by N array storing
            the diagonal two-body coefficeints :math:`V_{pq}` above.
        basis_transformation_matrix (ndarray of floats) an N by N array
            storing the values of the basis transformation.
    """
    # Diagonalize the one-body matrix.
    eigenvalues, eigenvectors = numpy.linalg.eig(one_body_matrix)
    basis_transformation_matrix = numpy.conjugate(eigenvectors.transpose())

    # Obtain the diagonal two-body matrix.
    density_density_matrix = numpy.outer(eigenvalues, eigenvalues)

    # Return.
    return density_density_matrix, basis_transformation_matrix
