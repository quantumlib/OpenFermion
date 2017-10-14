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

"""Givens rotation decomposition of matrices."""
from __future__ import absolute_import

import numpy

from openfermion.config import EQ_TOLERANCE


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
            rotations. The list looks something like
            [(G_1, ), (G_2, G_3), ... ]. The Givens rotations within a tuple
            can be implemented in parallel. The description of a Givens
            rotation is itself a tuple of the form (i, j, theta, phi), which
            represents a Givens rotation of rows i and j by angles theta
            and phi.
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
            # Zero out entry in row l
            givens_rotation = givens_matrix_elements(current_matrix[l, k],
                                                     current_matrix[l + 1, k])
            expanded_givens_rotation = expand_two_by_two(givens_rotation,
                                                         l, l + 1, m)
            current_matrix = expanded_givens_rotation.dot(current_matrix)
            left_unitary = expanded_givens_rotation.dot(left_unitary)

    # Compute the decomposition of unitary_rows into Givens rotations
    givens_rotations = list()
    if m != n:
        # There are n - 1 iterations (the circuit depth is n - 1)
        for k in range(n - 1):
            # Get the (row, column) indices of elements to zero out in
            # parallel.
            # Get the maximum number of simultaneous rotations that
            # will be performed
            max_simul_rotations = min(m, n - m)
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
                # Compute the Givens rotation to zero out the (i, j) element
                left_element = current_matrix[i, j - 1].conj()
                right_element = current_matrix[i, j].conj()
                givens_rotation = givens_matrix_elements(left_element,
                                                         right_element)
                # Need to switch the rows to zero out right_element
                # rather than left_element
                givens_rotation = givens_rotation[(1, 0), :]

                # Add the parameters to the list
                theta = numpy.arccos(numpy.real(givens_rotation[0, 0]))
                phi = numpy.angle(givens_rotation[1, 1])
                parallel_rotations.append((j - 1, j, theta, phi))

                # Update the matrix
                expanded_givens_rotation = expand_two_by_two(givens_rotation,
                                                             j - 1, j, n)
                current_matrix = current_matrix.dot(
                        expanded_givens_rotation.T.conj())

            # Append the current list of parallel rotations to the list
            givens_rotations.append(tuple(parallel_rotations))

    # Get the diagonal entries
    diagonal = current_matrix.diagonal()

    return left_unitary, givens_rotations, diagonal


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
        c = 1.
        s = 0.
        phase = 1.
    # Handle case that b is zero and a is nonzero
    elif abs(b) < EQ_TOLERANCE:
        c = 0.
        s = 1.
        sign_a = a / abs(a)
        phase = sign_a
    # Handle case that a and b are both nonzero
    else:
        denominator = numpy.sqrt(abs(a) ** 2 + abs(b) ** 2)

        c = abs(b) / denominator
        s = abs(a) / denominator
        sign_b = b / abs(b)
        sign_a = a / abs(a)
        phase = sign_a * sign_b.conjugate()

    # Construct matrix and return
    givens_rotation = numpy.array([[c, -phase * s],
                                  [s, phase * c]], dtype=complex)
    return givens_rotation


def expand_two_by_two(M, i, j, n):
    """Expand the 2 x 2 matrix M to an n x n matrix acting on coordinates
    i and j.
    """
    expanded_M = numpy.eye(n, dtype=complex)
    expanded_M[([i], [j]), (i, j)] = M
    return expanded_M
