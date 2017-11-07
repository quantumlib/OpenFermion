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
from scipy.linalg import schur
from scipy.sparse import csc_matrix, eye

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import QuadraticHamiltonian
from openfermion.utils._sparse_tools import (jw_hartree_fock_state,
                                             kronecker_operators,
                                             pauli_matrix_map)


def ground_state_preparation_circuit(quadratic_hamiltonian):
    """Obtain a description of a circuit which prepares the ground state
    of a quadratic Hamiltonian.

    Args:
        quadratic_hamiltonian(QuadraticHamiltonian):
            The Hamiltonian whose ground state is desired.

    Returns:
        circuit_description(list[tuple]):
            A list of operations describing the circuit. Each operation
            is a tuple of objects describing elementary operations that
            can be performed in parallel. Each elementary operation
            is either the string 'pht', indicating a particle-hole
            transformation on the last fermionic mode, or a tuple of
            the form (i, j, theta, phi), indicating a Givens rotation
            of modes i and j by angles theta and phi.
        n_electrons(int):
            The number of electrons to start with. This describes the
            initial state that the circuit should be applied to: it should
            be a Slater determinant (in the computational basis) with the
            first n_electrons orbitals filled.
    """
    if not isinstance(quadratic_hamiltonian, QuadraticHamiltonian):
        raise ValueError('Input must be an instance of QuadraticHamiltonian.')

    if quadratic_hamiltonian.conserves_particle_number():
        # The Hamiltonian conserves particle number, so we don't need
        # to use the most general procedure.
        hermitian_matrix = quadratic_hamiltonian.combined_hermitian_part()
        # Get the unitary rows which represent the ground state
        # Slater determinant
        energies, diagonalizing_unitary = numpy.linalg.eigh(
                hermitian_matrix)
        # We get the ground state by filling the orbitals that have
        # negative energy
        num_negative_energies = numpy.count_nonzero(
                energies < -EQ_TOLERANCE)
        slater_determinant_matrix = diagonalizing_unitary.T[
                :num_negative_energies]
        # Get the circuit description
        left_unitary, decomposition, diagonal = givens_decomposition(
                slater_determinant_matrix)
        circuit_description = list(reversed(decomposition))
        n_electrons = num_negative_energies
    else:
        # The Hamiltonian does not conserve particle number, so we
        # need to use the most general procedure.
        majorana_matrix, majorana_constant = (
                quadratic_hamiltonian.majorana_form())
        # Get the unitary rows which represent the ground state
        # Slater determinant
        diagonalizing_unitary = diagonalizing_fermionic_unitary(
                majorana_matrix)
        slater_determinant_matrix = diagonalizing_unitary[
                quadratic_hamiltonian.n_qubits:]
        # Get the circuit description
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(
                    slater_determinant_matrix))
        circuit_description = list(reversed(decomposition))
        n_electrons = 0
    return circuit_description, n_electrons


def jw_get_quadratic_hamiltonian_ground_state(quadratic_hamiltonian):
    """Get the ground state of a quadratic hamiltonian as a sparse array
    using the Jordan-Wigner encoding."""
    n_qubits = quadratic_hamiltonian.n_qubits

    # Obtain the circuit that prepares the ground state
    circuit_description, n_electrons = ground_state_preparation_circuit(
            quadratic_hamiltonian)

    # Initialize the starting state
    state = jw_hartree_fock_state(n_electrons, n_qubits)

    # Apply the circuit
    particle_hole_transformation = (
            jw_sparse_particle_hole_transformation_last_mode(n_qubits))
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            if op == 'pht':
                state = particle_hole_transformation.dot(state)
            else:
                i, j, theta, phi = op
                state = jw_sparse_givens_rotation(
                            i, j, theta, phi, n_qubits).dot(state)

    return state


def fermionic_gaussian_decomposition(unitary_rows):
    """Decompose a matrix into a sequence of Givens rotations and
    particle-hole transformations on the last fermionic mode.

    The input is an n x (2 * n) matrix W with orthonormal rows.
    Furthermore, W has the block form::

        W = [ W_1  |  W_2 ]

    where W_1 and W_2 satisfy::

        W_1 * W_1^\dagger + W_2 * W_2^\dagger = I
        W_1 * W_2^T + W_2 * W_1^T = 0

    W can be decomposed as::

        V * W * U^\dagger = [ 0  |  D ]

    where V and U are unitary matrices and D is a diagonal unitary matrix.
    Furthermore, we can decompose U as a sequence of Givens rotations
    and particle-hole transformations on the last fermionic mode.
    This particle-hole transformation maps a^\dagger_n to a^n and vice
    versa, while leaving the other ladder operators invariant.

    The decomposition of U is returned as a list of tuples of objects
    describing rotations and particle-hole transformations. The list looks
    something like [('pht', ), (G_1, ), ('pht', G_2), ... ].
    The objects within a tuple are either the string 'pht', which indicates
    a particle-hole transformation on the last fermionic mode, or a tuple
    of the form (i, j, theta, phi), which indicates a Givens rotation
    of rows i and j by angles theta and phi.

    Args:
        unitary_rows(ndarray): A matrix with orthonormal rows and
            additional structure described above.

    Returns:
        left_unitary(ndarray): An n x n matrix representing V.
        decomposition(list[tuple]): The decomposition of U.
        diagonal(ndarray): A list of the nonzero entries of D.
    """
    current_matrix = numpy.copy(unitary_rows)
    n, p = current_matrix.shape

    # Check that p = 2 * n
    if p != 2 * n:
        raise ValueError('The input matrix must have twice as many columns '
                         'as rows.')

    # Check that left and right parts of unitary_rows satisfy the constraints
    # necessary for the transformed fermionic operators to satisfy
    # the fermionic anticommutation relations
    left_part = unitary_rows[:, :n]
    right_part = unitary_rows[:, n:]
    constraint_matrix_1 = (left_part.dot(left_part.T.conj()) +
                           right_part.dot(right_part.T.conj()))
    constraint_matrix_2 = (left_part.dot(right_part.T) +
                           right_part.dot(left_part.T))
    discrepancy_1 = numpy.amax(abs(constraint_matrix_1 - numpy.eye(n)))
    discrepancy_2 = numpy.amax(abs(constraint_matrix_2))
    if discrepancy_1 > EQ_TOLERANCE or discrepancy_2 > EQ_TOLERANCE:
        raise ValueError('The input matrix does not satisfy the constraints '
                         'necessary for a proper transformation of the '
                         'fermionic ladder operators.')

    # Compute left_unitary using Givens rotations
    left_unitary = numpy.eye(n, dtype=complex)
    for k in range(n - 1):
        # Zero out entries in column k
        for l in range(n - 1 - k):
            # Zero out entry in row l if needed
            if abs(current_matrix[l, k]) > EQ_TOLERANCE:
                givens_rotation = givens_matrix_elements(
                        current_matrix[l, k], current_matrix[l + 1, k])
                # Apply Givens rotation
                givens_rotate(current_matrix, givens_rotation, l, l + 1)
                givens_rotate(left_unitary, givens_rotation, l, l + 1)

    # Initialize list to store decomposition of current_matrix
    decomposition = list()
    # There are 2 * n - 1 iterations (that is the circuit depth)
    for k in range(2 * n - 1):
        # Initialize the list of parallel operations to perform
        # in this iteration
        parallel_ops = list()

        # Perform a particle-hole transformation if necessary
        if k % 2 == 0 and abs(current_matrix[k // 2, n - 1]) > EQ_TOLERANCE:
            parallel_ops.append('pht')
            swap_columns(current_matrix, n - 1, 2 * n - 1)

        # Get the (row, column) indices of elements to zero out in parallel.
        if k < n:
            end_row = k
            end_column = n - 1 - k
        else:
            end_row = n - 1
            end_column = k - (n - 1)
        column_indices = range(end_column, n - 1, 2)
        row_indices = range(end_row, end_row - len(column_indices), -1)
        indices_to_zero_out = zip(row_indices, column_indices)

        for i, j in indices_to_zero_out:
            # Compute the Givens rotation to zero out the (i, j) element,
            # if needed
            left_element = current_matrix[i, j].conj()
            if abs(left_element) > EQ_TOLERANCE:
                # We actually need to perform a Givens rotation
                right_element = current_matrix[i, j + 1].conj()
                givens_rotation = givens_matrix_elements(left_element,
                                                         right_element)

                # Add the parameters to the list
                theta = numpy.arccos(numpy.real(givens_rotation[0, 0]))
                phi = numpy.angle(givens_rotation[1, 1])
                parallel_ops.append((j, j + 1, theta, phi))

                # Update the matrix
                double_givens_rotate(current_matrix, givens_rotation,
                                     j, j + 1, which='col')

        # Append the current list of parallel rotations to the list
        decomposition.append(tuple(parallel_ops))

    # Get the diagonal entries
    diagonal = current_matrix[range(n), range(n, 2 * n)]
    return left_unitary, decomposition, diagonal


def givens_decomposition(unitary_rows):
    """Decompose a matrix into a sequence of Givens rotations.

    The input is an m x n matrix Q with m <= n. The rows of Q are orthonormal.
    Q can be decomposed as follows:

        V * Q * U^\dagger = D

    where V and U are unitary matrices, and D is an m x n matrix with the
    first m columns forming a diagonal matrix and the rest of the columns
    being zero. Furthermore, we can decompose U as

        U = G_k * ... * G_1

    where G_1, ..., G_k are complex Givens rotations, which are invertible
    n x n matrices. We describe a complex Givens rotation by the column
    indices (i, j) that it acts on, plus two angles (theta, phi) that
    characterize the corresponding 2x2 unitary matrix

        [ cos(theta)    -e^{i phi} sin(theta) ]
        [ sin(theta)     e^{i phi} cos(theta) ]

    Args:
        unitary_rows: A numpy array or matrix with orthonormal rows,
            representing the matrix Q.

    Returns:
        left_unitary: An m x m numpy array representing the matrix V.

        givens_rotations: A list of tuples of objects describing Givens
            rotations. The list looks like [(G_1, ), (G_2, G_3), ... ].
            The Givens rotations within a tuple can be implemented in parallel.
            The description of a Givens rotation is itself a tuple of the
            form (i, j, theta, phi), which represents a Givens rotation of
            rows i and j by angles theta and phi.

        diagonal: A list of the nonzero entries of D.
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
    givens_rotations = list()
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

            parallel_rotations = list()
            for i, j in indices_to_zero_out:
                # Compute the Givens rotation to zero out the (i, j) element,
                # if needed
                right_element = current_matrix[i, j].conj()
                if abs(right_element) > EQ_TOLERANCE:
                    # We actually need to perform a Givens rotation
                    left_element = current_matrix[i, j - 1].conj()
                    givens_rotation = givens_matrix_elements(left_element,
                                                             right_element)
                    # Need to swap the rows to zero out right_element
                    # rather than left_element
                    swap_rows(givens_rotation, 0, 1)

                    # Add the parameters to the list
                    theta = numpy.arccos(numpy.real(givens_rotation[0, 0]))
                    phi = numpy.angle(givens_rotation[1, 1])
                    parallel_rotations.append((j - 1, j, theta, phi))

                    # Update the matrix
                    givens_rotate(current_matrix, givens_rotation,
                                  j - 1, j, which='col')

            # Append the current list of parallel rotations to the list
            givens_rotations.append(tuple(parallel_rotations))

    # Get the diagonal entries
    diagonal = current_matrix.diagonal()

    return left_unitary, givens_rotations, diagonal


def diagonalizing_fermionic_unitary(antisymmetric_matrix):
    """Compute the unitary that diagonalizes a quadratic Hamiltonian.

    The input matrix represents a quadratic Hamiltonian in the Majorana basis.
    The output matrix is a unitary that represents a transformation (mixing)
    of the fermionic ladder operators. We use the convention that the
    creation operators are listed before the annihilation operators.
    The returned unitary has additional structure which ensures
    that the transformed ladder operators also satisfy the fermionic
    anticommutation relations.

    Args:
        antisymmetric_matrix(ndarray): A (2 * n_qubits) x (2 * n_qubits)
            antisymmetric matrix representing a quadratic Hamiltonian in the
            Majorana basis.
    Returns:
        diagonalizing_unitary(ndarray): A (2 * n_qubits) x (2 * n_qubits)
            unitary matrix representing a transformation of the fermionic
            ladder operators.
    """
    m, n = antisymmetric_matrix.shape
    n_qubits = n // 2

    # Get the orthogonal transformation that puts antisymmetric_matrix
    # into canonical form
    canonical, orthogonal = antisymmetric_canonical_form(antisymmetric_matrix)

    # Create the matrix that converts between fermionic ladder and
    # Majorana bases
    normalized_identity = numpy.eye(n_qubits, dtype=complex) / numpy.sqrt(2.)
    majorana_basis_change = numpy.eye(
            2 * n_qubits, dtype=complex) / numpy.sqrt(2.)
    majorana_basis_change[n_qubits:, n_qubits:] *= -1.j
    majorana_basis_change[:n_qubits, n_qubits:] = normalized_identity
    majorana_basis_change[n_qubits:, :n_qubits] = 1.j * normalized_identity

    # Compute the unitary and return
    diagonalizing_unitary = majorana_basis_change.T.conj().dot(
            orthogonal.dot(majorana_basis_change))

    return diagonalizing_unitary


def antisymmetric_canonical_form(antisymmetric_matrix):
    """Compute the canonical form of an antisymmetric matrix.

    The input is a real, antisymmetric n x n matrix A, where n is even.
    Its canonical form is::

        A = R^T C R

    where R is a real, orthogonal matrix and C has the form::

        [  0     D ]
        [ -D     0 ]

    where D is a diagonal matrix with nonnegative entries.

    Args:
        antisymmetric_matrix(ndarray): An antisymmetric matrix with even
            dimension.

    Returns:
        canonical(ndarray): The canonical form C of antisymmetric_matrix
        orthogonal(ndarray): The orthogonal transformation R.
    """
    m, n = antisymmetric_matrix.shape

    if m != n or n % 2 != 0:
        raise ValueError('The input matrix must be square with even '
                         'dimension.')

    # Check that input matrix is antisymmetric
    matrix_plus_transpose = antisymmetric_matrix + antisymmetric_matrix.T
    maxval = numpy.max(numpy.abs(matrix_plus_transpose))
    if maxval > EQ_TOLERANCE:
        raise ValueError('The input matrix must be antisymmetric.')

    # Compute Schur decomposition
    canonical, orthogonal = schur(antisymmetric_matrix, output='real')

    # The returned form is block diagonal; we need to permute rows and columns
    # to put it into the form we want
    num_blocks = n // 2
    for i in range(1, num_blocks, 2):
        swap_rows(canonical, i, num_blocks + i - 1)
        swap_columns(canonical, i, num_blocks + i - 1)
        swap_columns(orthogonal, i, num_blocks + i - 1)
        if num_blocks % 2 != 0:
            swap_rows(canonical, num_blocks - 1, num_blocks + i)
            swap_columns(canonical, num_blocks - 1, num_blocks + i)
            swap_columns(orthogonal, num_blocks - 1, num_blocks + i)

    # Now we permute so that the upper right block is non-negative
    for i in range(num_blocks):
        if canonical[i, num_blocks + i] < -EQ_TOLERANCE:
            swap_rows(canonical, i, num_blocks + i)
            swap_columns(canonical, i, num_blocks + i)
            swap_columns(orthogonal, i, num_blocks + i)

    return canonical, orthogonal.T


def givens_matrix_elements(a, b):
    """Compute the matrix elements of the Givens rotation that zeroes out one
    of two row entries.

    Returns a matrix G such that

        G * [a  b]^T= [0  r]^T

    where r is a complex number.

    Args:
        a: A complex number representing the upper row entry
        b: A complex number representing the lower row entry
    Returns:
        G: A 2 x 2 numpy array representing the matrix G. The numbers in the
            first column of G are real.
    """
    # Handle case that a is zero
    if abs(a) < EQ_TOLERANCE:
        cosine = 1.
        sine = 0.
        phase = 1.
    # Handle case that b is zero and a is nonzero
    elif abs(b) < EQ_TOLERANCE:
        cosine = 0.
        sine = 1.
        phase = 1.
    # Handle case that a and b are both nonzero
    else:
        denominator = numpy.sqrt(abs(a) ** 2 + abs(b) ** 2)
        cosine = abs(b) / denominator
        sine = abs(a) / denominator
        sign_b = b / abs(b)
        sign_a = a / abs(a)
        phase = sign_a * sign_b.conjugate()
        # If phase is a real number, convert it to a float
        if numpy.isreal(phase):
            phase = numpy.real(phase)

    # Construct matrix and return
    givens_rotation = numpy.array([[cosine, -phase * sine],
                                   [sine, phase * cosine]])
    return givens_rotation


def givens_rotate(operator, givens_rotation, i, j, which='row'):
    """Apply a Givens rotation to coordinates i and j of an operator."""
    if which == 'row':
        # Rotate rows i and j
        row_i = operator[i].copy()
        row_j = operator[j].copy()
        operator[i] = (givens_rotation[0, 0] * row_i +
                       givens_rotation[0, 1] * row_j)
        operator[j] = (givens_rotation[1, 0] * row_i +
                       givens_rotation[1, 1] * row_j)
    elif which == 'col':
        # Rotate columns i and j
        col_i = operator[:, i].copy()
        col_j = operator[:, j].copy()
        operator[:, i] = (givens_rotation[0, 0] * col_i +
                          givens_rotation[0, 1].conj() * col_j)
        operator[:, j] = (givens_rotation[1, 0] * col_i +
                          givens_rotation[1, 1].conj() * col_j)
    else:
        raise ValueError('"which" must be equal to "row" or "col".')


def double_givens_rotate(operator, givens_rotation, i, j, which='row'):
    """Apply a double Givens rotation.

    Applies a Givens rotation to coordinates i and j and the the conjugate
    Givens rotation to coordinates n + i and n + j, where
    n = dim(operator) / 2. dim(operator) must be even.
    """
    m, p = operator.shape

    if which == 'row':
        if m % 2 != 0:
            raise ValueError('To apply a double Givens rotation on rows, '
                             'the number of rows must be even.')
        n = m // 2
        # Rotate rows i and j
        givens_rotate(operator[:n], givens_rotation, i, j, which='row')
        # Rotate rows n + i and n + j
        givens_rotate(operator[n:], givens_rotation.conj(), i, j, which='row')
    elif which == 'col':
        if p % 2 != 0:
            raise ValueError('To apply a double Givens rotation on columns, '
                             'the number of columns must be even.')
        n = p // 2
        # Rotate columns i and j
        givens_rotate(operator[:, :n], givens_rotation, i, j, which='col')
        # Rotate cols n + i and n + j
        givens_rotate(operator[:, n:], givens_rotation.conj(), i, j,
                      which='col')
    else:
        raise ValueError('"which" must be equal to "row" or "col".')


def jw_sparse_givens_rotation(i, j, theta, phi, n_qubits):
    """Return the matrix (acting on a full wavefunction) that performs a
    Givens rotation of modes i and j in the Jordan-Wigner encoding."""
    if j != i + 1:
        raise ValueError('Only adjacent modes can be rotated.')
    if j > n_qubits - 1:
        raise ValueError('Too few qubits requested.')

    cosine = numpy.cos(theta)
    sine = numpy.sin(theta)
    phase = numpy.exp(1.j * phi)

    # Create the two-qubit rotation matrix
    rotation_matrix = csc_matrix(
            ([1., phase * cosine, -phase * sine, sine, cosine, phase],
             ((0, 1, 1, 2, 2, 3), (0, 1, 2, 1, 2, 3))),
            shape=(4, 4))

    # Initialize identity operators
    left_eye = eye(2 ** i, format='csc')
    right_eye = eye(2 ** (n_qubits - 1 - j), format='csc')

    # Construct the matrix and return
    givens_matrix = kronecker_operators([left_eye, rotation_matrix, right_eye])

    return givens_matrix


def jw_sparse_particle_hole_transformation_last_mode(n_qubits):
    """Return the matrix (acting on a full wavefunction) that performs a
    particle-hole transformation on the last mode in the Jordan-Wigner
    encoding.
    """
    left_eye = eye(2 ** (n_qubits - 1), format='csc')
    return kronecker_operators([left_eye, pauli_matrix_map['X']])


def swap_rows(M, i, j):
    """Swap rows i and j of matrix M."""
    row_i = M[i, :].copy()
    row_j = M[j, :].copy()
    M[i, :], M[j, :] = row_j, row_i


def swap_columns(M, i, j):
    """Swap columns i and j of matrix M."""
    column_i = M[:, i].copy()
    column_j = M[:, j].copy()
    M[:, i], M[:, j] = column_j, column_i
