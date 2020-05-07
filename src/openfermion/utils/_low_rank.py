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
import scipy.linalg

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import FermionOperator, InteractionOperator
from openfermion.utils import count_qubits, is_hermitian, normal_ordered


def get_chemist_two_body_coefficients(two_body_coefficients, spin_basis=True):
    r"""Convert two-body operator coefficients to low rank tensor.

    The input is a two-body fermionic Hamiltonian expressed as
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s`

    We will convert this to the chemistry convention expressing it as
    :math:`\sum_{pqrs} g_{pqrs} a^\dagger_p a_q a^\dagger_r a_s`
    but without the spin degree of freedom.

    In the process of performing this conversion, constants and one-body
    terms come out, which will be returned as well.

    Args:
        two_body_coefficients (ndarray): an N x N x N x N
            numpy array giving the :math:`h_{pqrs}` tensor.
        spin_basis (bool): True if the two-body terms are passed in spin
            orbital basis. False if already in spatial orbital basis.

    Returns:
        one_body_correction (ndarray): an N x N array of floats giving
            coefficients of the :math:`a^\dagger_p a_q` terms that come out.
        chemist_two_body_coefficients (ndarray): an N x N x N x N numpy array
            giving the :math:`g_{pqrs}` tensor in chemist notation.

    Raises:
        TypeError: Input must be two-body number conserving
            FermionOperator or InteractionOperator.
    """
    # Initialize.
    n_orbitals = two_body_coefficients.shape[0]
    chemist_two_body_coefficients = numpy.transpose(
        two_body_coefficients, [0, 3, 1, 2])

    # If the specification was in spin-orbitals, chop down to spatial orbitals
    # assuming a spin-symmetric interaction.
    if spin_basis:
        n_orbitals = n_orbitals // 2
        alpha_indices = list(range(0, n_orbitals * 2, 2))
        beta_indices = list(range(1, n_orbitals * 2, 2))
        chemist_two_body_coefficients = chemist_two_body_coefficients[
            numpy.ix_(alpha_indices, alpha_indices,
                      beta_indices, beta_indices)]

    # Determine a one body correction in the spin basis from spatial basis.
    one_body_correction = numpy.zeros(
        (2 * n_orbitals, 2 * n_orbitals), complex)
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        for sigma, tau in itertools.product(range(2), repeat=2):
            if (q == r) and (sigma == tau):
                one_body_correction[2 * p + sigma, 2 * s + tau] -= (
                    chemist_two_body_coefficients[p, q, r, s])

    # Return.
    return one_body_correction, chemist_two_body_coefficients


def low_rank_two_body_decomposition(two_body_coefficients,
                                    truncation_threshold=1e-8,
                                    final_rank=None,
                                    spin_basis=True):
    r"""Convert two-body operator into sum of squared one-body operators.

    As in arXiv:1808.02625, this function decomposes
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s` as
    :math:`\sum_{l} \lambda_l (\sum_{pq} g_{lpq} a^\dagger_p a_q)^2`
    l is truncated to take max value L so that
    :math:`\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x`

    Args:
        two_body_coefficients (ndarray): an N x N x N x N
            numpy array giving the :math:`h_{pqrs}` tensor.
            This tensor must be 8-fold symmetric (real integrals).
        truncation_threshold (optional Float): the value of x, above.
        final_rank (optional int): if provided, this specifies the value of
            L at which to truncate. This overrides truncation_threshold.
        spin_basis (bool): True if the two-body terms are passed in spin
            orbital basis.  False if already in spatial orbital basis.

    Returns:
        eigenvalues (ndarray of floats): length L array
            giving the :math:`\lambda_l`.
        one_body_squares (ndarray of floats): L x N x N array of floats
            corresponding to the value of :math:`g_{pql}`.
        one_body_correction (ndarray): One-body correction terms that result
            from reordering to chemist ordering, in spin-orbital basis.
        truncation_value (float): after truncation, this is the value
            :math:`\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x`

    Raises:
        TypeError: Invalid two-body coefficient tensor specification.
    """
    # Initialize N^2 by N^2 interaction array.
    one_body_correction, chemist_two_body_coefficients = (
        get_chemist_two_body_coefficients(two_body_coefficients, spin_basis))
    n_orbitals = chemist_two_body_coefficients.shape[0]
    full_rank = n_orbitals ** 2
    interaction_array = numpy.reshape(chemist_two_body_coefficients,
                                      (full_rank, full_rank))

    # Make sure interaction array is symmetric and real.
    asymmetry = numpy.sum(numpy.absolute(
        interaction_array - interaction_array.transpose()))
    imaginary_norm = numpy.sum(numpy.absolute(interaction_array.imag))
    if asymmetry > EQ_TOLERANCE or imaginary_norm > EQ_TOLERANCE:
        raise TypeError('Invalid two-body coefficient tensor specification.')

    # Decompose with exact diagonalization.
    eigenvalues, eigenvectors = numpy.linalg.eigh(interaction_array)

    # Get one-body squares and compute weights.
    term_weights = numpy.zeros(full_rank)
    one_body_squares = numpy.zeros((full_rank,
                                    2 * n_orbitals, 2 * n_orbitals), complex)

    # Reshape and add spin back in.
    for l in range(full_rank):
        one_body_squares[l] = numpy.kron(
            numpy.reshape(
                eigenvectors[:, l], (n_orbitals, n_orbitals)), numpy.eye(2))
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
    if final_rank is None:
        max_rank = 1 + numpy.argmax(
            truncation_errors <= truncation_threshold)
    else:
        max_rank = final_rank
    truncation_value = truncation_errors[max_rank - 1]
    return (eigenvalues[:max_rank],
            one_body_squares[:max_rank],
            one_body_correction,
            truncation_value)


def prepare_one_body_squared_evolution(one_body_matrix, spin_basis=True):
    r"""Get Givens angles and DiagonalHamiltonian to simulate squared one-body.

    The goal here will be to prepare to simulate evolution under
    :math:`(\sum_{pq} h_{pq} a^\dagger_p a_q)^2` by decomposing as
    :math:`R e^{-i \sum_{pq} V_{pq} n_p n_q} R^\dagger' where
    :math:`R` is a basis transformation matrix.

    TODO: Add option for truncation based on one-body eigenvalues.

    Args:
        one_body_matrix (ndarray of floats): an N by N array storing the
            coefficients of a one-body operator to be squared. For instance,
            in the above the elements of this matrix are :math:`h_{pq}`.
        spin_basis (bool): Whether the matrix is passed in the
            spin orbital basis.

    Returns:
        density_density_matrix(ndarray of floats) an N by N array storing
            the diagonal two-body coefficeints :math:`V_{pq}` above.
        basis_transformation_matrix (ndarray of floats) an N by N array
            storing the values of the basis transformation.

    Raises:
        ValueError: one_body_matrix is not Hermitian.
    """
    # If the specification was in spin-orbitals, chop back down to spatial orbs
    # assuming a spin-symmetric interaction
    if spin_basis:
        n_modes = one_body_matrix.shape[0]
        alpha_indices = list(range(0, n_modes, 2))
        one_body_matrix = one_body_matrix[
                numpy.ix_(alpha_indices, alpha_indices)]

    # Diagonalize the one-body matrix.
    if is_hermitian(one_body_matrix):
        eigenvalues, eigenvectors = numpy.linalg.eigh(one_body_matrix)
    else:
        raise ValueError('one_body_matrix is not Hermitian.')
    basis_transformation_matrix = numpy.conjugate(eigenvectors.transpose())

    # If the specification was in spin-orbitals, expand back
    if spin_basis:
        basis_transformation_matrix = numpy.kron(
                basis_transformation_matrix, numpy.eye(2))
        eigenvalues = numpy.kron(eigenvalues, numpy.ones(2))

    # Obtain the diagonal two-body matrix.
    density_density_matrix = numpy.outer(eigenvalues, eigenvalues)

    # Return.
    return density_density_matrix, basis_transformation_matrix
