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

"""This module contains functions for compiling circuits to prepare
Slater determinants and fermionic Gaussian states."""
from __future__ import absolute_import

import numpy

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import QuadraticHamiltonian
from openfermion.ops._quadratic_hamiltonian import (
        fermionic_gaussian_decomposition, givens_matrix_elements,
        givens_rotate)


def gaussian_state_preparation_circuit(
        quadratic_hamiltonian, occupied_orbitals=None):
    """Obtain the description of a circuit which prepares a fermionic Gaussian
    state.

    Fermionic Gaussian states can be regarded as eigenstates of quadratic
    Hamiltonians. If the Hamiltonian conserves particle number, then these are
    just Slater determinants. See arXiv:1711.05395 for a detailed description
    of how this procedure works.

    The circuit description is returned as a sequence of elementary
    operations; operations that can be performed in parallel are grouped
    together. Each elementary operation is either

    - the string 'pht', indicating the particle-hole transformation
      on the last fermionic mode, which is the operator :math:`\mathcal{B}`
      such that

      .. math::

          \\begin{align}
              \mathcal{B} a_N \mathcal{B}^\dagger &= a_N^\dagger,\\\\
              \mathcal{B} a_j \mathcal{B}^\dagger &= a_j, \\quad
                  j = 1, \ldots, N-1,
          \end{align}

      or

    - a tuple :math:`(i, j, \\theta, \\varphi)`, indicating the operation

      .. math::
          \exp[i \\varphi a_j^\dagger a_j]
          \exp[\\theta (a_i^\dagger a_j - a_j^\dagger a_i)],

      a Givens rotation of modes :math:`i` and :math:`j` by angles
      :math:`\\theta` and :math:`\\varphi`.

    Args:
        quadratic_hamiltonian(QuadraticHamiltonian):
            The Hamiltonian whose eigenstate is desired.
        occupied_orbitals(list):
            A list of integers representing the indices of the occupied
            orbitals in the desired Gaussian state. If this is None
            (the default), then it is assumed that the ground state is
            desired, i.e., the orbitals with negative energies are filled.

    Returns
    -------
        circuit_description (list[tuple]):
            A list of operations describing the circuit. Each operation
            is a tuple of objects describing elementary operations that
            can be performed in parallel. Each elementary operation
            is either the string 'pht', indicating a particle-hole
            transformation on the last fermionic mode, or a tuple of
            the form :math:`(i, j, \\theta, \\varphi)`,
            indicating a Givens rotation
            of modes :math:`i` and :math:`j` by angles :math:`\\theta`
            and :math:`\\varphi`.
        start_orbitals (list):
            The occupied orbitals to start with. This describes the
            initial state that the circuit should be applied to: it should
            be a Slater determinant (in the computational basis) with these
            orbitals filled.
    """
    if not isinstance(quadratic_hamiltonian, QuadraticHamiltonian):
        raise ValueError('Input must be an instance of QuadraticHamiltonian.')

    if quadratic_hamiltonian.conserves_particle_number:
        # The Hamiltonian conserves particle number, so we don't need
        # to use the most general procedure.
        hermitian_matrix = quadratic_hamiltonian.combined_hermitian_part
        energies, diagonalizing_unitary = numpy.linalg.eigh(hermitian_matrix)

        if occupied_orbitals is None:
            # The ground state is desired, so we fill the orbitals that have
            # negative energy
            num_negative_energies = numpy.count_nonzero(
                energies < -EQ_TOLERANCE)
            occupied_orbitals = range(num_negative_energies)

        # Get the unitary rows which represent the Slater determinant
        slater_determinant_matrix = diagonalizing_unitary.T[occupied_orbitals]

        # Get the circuit description
        circuit_description = slater_determinant_preparation_circuit(
            slater_determinant_matrix)
        start_orbitals = range(len(occupied_orbitals))
    else:
        # The Hamiltonian does not conserve particle number, so we
        # need to use the most general procedure.
        diagonalizing_unitary = (
            quadratic_hamiltonian.diagonalizing_bogoliubov_transform())

        # Get the unitary rows which represent the Gaussian unitary
        gaussian_unitary_matrix = diagonalizing_unitary[
            quadratic_hamiltonian.n_qubits:]

        # Get the circuit description
        decomposition, left_decomposition, diagonal, left_diagonal = (
            fermionic_gaussian_decomposition(gaussian_unitary_matrix))
        if occupied_orbitals is None:
            # The ground state is desired, so the circuit should be applied
            # to the vaccuum state
            start_orbitals = []
            circuit_description = list(reversed(decomposition))
        else:
            start_orbitals = occupied_orbitals
            # The circuit won't be applied to the ground state, so we need to
            # use left_decomposition
            circuit_description = list(reversed(
                decomposition + left_decomposition))

    return circuit_description, start_orbitals


def slater_determinant_preparation_circuit(slater_determinant_matrix):
    """Obtain the description of a circuit which prepares a Slater determinant.

    The input is an :math:`N_f \\times N` matrix :math:`Q` with orthonormal
    rows. Such a matrix describes the Slater determinant

    .. math::

        b^\dagger_1 \cdots b^\dagger_{N_f} \lvert \\text{vac} \\rangle,

    where

    .. math::

        b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k.

    The output is the description of a circuit which prepares this
    Slater determinant, up to a global phase.
    The starting state which the circuit should be applied to
    is a Slater determinant (in the computational basis) with
    the first :math:`N_f` orbitals filled.

    Args:
        slater_determinant_matrix: The matrix :math:`Q` which describes the
            Slater determinant to be prepared.
    Returns:
        circuit_description:
            A list of operations describing the circuit. Each operation
            is a tuple of elementary operations that can be performed in
            parallel. Each elementary operation is a tuple of the form
            :math:`(i, j, \\theta, \\varphi)`, indicating a Givens rotation
            of modes :math:`i` and :math:`j` by angles :math:`\\theta`
            and :math:`\\varphi`.
    """
    decomposition, left_unitary, diagonal = givens_decomposition(
        slater_determinant_matrix)
    circuit_description = list(reversed(decomposition))
    return circuit_description


def givens_decomposition(unitary_rows):
    """Decompose a matrix into a sequence of Givens rotations.

    The input is an :math:`m \\times n` matrix :math:`Q` with :math:`m \leq n`.
    The rows of :math:`Q` are orthonormal.
    :math:`Q` can be decomposed as follows:

    .. math::

        V Q U^\dagger = D

    where :math:`V` and :math:`U` are unitary matrices, and :math:`D`
    is an :math:`m \\times n` matrix with the
    first :math:`m` columns forming a diagonal matrix and the rest of the
    columns being zero. Furthermore, we can decompose :math:`U` as

    .. math::

        U = G_k ... G_1

    where :math:`G_1, \\ldots, G_k` are complex Givens rotations.
    A Givens rotation is a rotation within the two-dimensional subspace
    spanned by two coordinate axes. Within the two relevant coordinate
    axes, a Givens rotation has the form

    .. math::

        \\begin{pmatrix}
            \\cos(\\theta) & -e^{i \\varphi} \\sin(\\theta) \\\\
            \\sin(\\theta) &     e^{i \\varphi} \\cos(\\theta)
        \\end{pmatrix}.

    Args:
        unitary_rows: A numpy array or matrix with orthonormal rows,
            representing the matrix Q.

    Returns
    -------
        givens_rotations (list[tuple]):
            A list of tuples of objects describing Givens
            rotations. The list looks like [(G_1, ), (G_2, G_3), ... ].
            The Givens rotations within a tuple can be implemented in parallel.
            The description of a Givens rotation is itself a tuple of the
            form :math:`(i, j, \\theta, \\varphi)`, which represents a
            Givens rotation of coordinates
            :math:`i` and :math:`j` by angles :math:`\\theta` and
            :math:`\\varphi`.
        left_decomposition (list[tuple]):
            The decomposition of :math:`V^\dagger D`.
        diagonal (ndarray):
            A list of the nonzero entries of :math:`D`.
        left_diagonal (ndarray):
            A list of the nonzero entries left from the decomposition
            of :math:`V^T D^*`.
    """
    current_matrix = numpy.copy(unitary_rows)
    m, n = current_matrix.shape

    # Check that m <= n
    if m > n:
        raise ValueError('The input m x n matrix must have m <= n')

    # Compute left_unitary using Givens rotations
    left_unitary = numpy.eye(m, dtype=complex)
    for k in reversed(range(n - m + 1, n)):
        # Zero out entries in column k
        for l in range(m - n + k):
            # Zero out entry in row l if needed
            if abs(current_matrix[l, k]) > EQ_TOLERANCE:
                givens_rotation = givens_matrix_elements(
                    current_matrix[l, k], current_matrix[l + 1, k])
                # Apply Givens rotation
                givens_rotate(current_matrix, givens_rotation, l, l + 1)
                givens_rotate(left_unitary, givens_rotation, l, l + 1)

    # Compute the decomposition of current_matrix into Givens rotations
    givens_rotations = []
    # If m = n (the matrix is square) then we don't need to perform any
    # Givens rotations!
    if m != n:
        # Get the maximum number of simultaneous rotations that
        # will be performed
        max_simul_rotations = min(m, n - m)
        # There are n - 1 iterations (the circuit depth is n - 1)
        for k in range(n - 1):
            # Get the (row, column) indices of elements to zero out in
            # parallel.
            if k < max_simul_rotations - 1:
                # There are k + 1 elements to zero out
                start_row = 0
                end_row = k + 1
                start_column = n - m - k
                end_column = start_column + 2 * (k + 1)
            elif k > n - 1 - max_simul_rotations:
                # There are n - 1 - k elements to zero out
                start_row = m - (n - 1 - k)
                end_row = m
                start_column = m - (n - 1 - k) + 1
                end_column = start_column + 2 * (n - 1 - k)
            else:
                # There are max_simul_rotations elements to zero out
                if max_simul_rotations == m:
                    start_row = 0
                    end_row = m
                    start_column = n - m - k
                    end_column = start_column + 2 * m
                else:
                    start_row = k + 1 - max_simul_rotations
                    end_row = k + 1
                    start_column = k + 1 - max_simul_rotations + 1
                    end_column = start_column + 2 * max_simul_rotations

            row_indices = range(start_row, end_row)
            column_indices = range(start_column, end_column, 2)
            indices_to_zero_out = zip(row_indices, column_indices)

            parallel_rotations = []
            for i, j in indices_to_zero_out:
                # Compute the Givens rotation to zero out the (i, j) element,
                # if needed
                right_element = current_matrix[i, j].conj()
                if abs(right_element) > EQ_TOLERANCE:
                    # We actually need to perform a Givens rotation
                    left_element = current_matrix[i, j - 1].conj()
                    givens_rotation = givens_matrix_elements(
                        left_element, right_element, which='right')

                    # Add the parameters to the list
                    theta = numpy.arcsin(numpy.real(givens_rotation[1, 0]))
                    phi = numpy.angle(givens_rotation[1, 1])
                    parallel_rotations.append((j - 1, j, theta, phi))

                    # Update the matrix
                    givens_rotate(current_matrix, givens_rotation,
                                  j - 1, j, which='col')

            # If the current list of parallel operations is not empty,
            # append it to the list,
            if parallel_rotations:
                givens_rotations.append(tuple(parallel_rotations))

    # Get the diagonal entries
    diagonal = current_matrix.diagonal()

    return givens_rotations, left_unitary, diagonal
