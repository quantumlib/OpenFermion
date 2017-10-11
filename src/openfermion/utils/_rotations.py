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

"""Rotations."""
from __future__ import absolute_import

import numpy

from openfermion.config import *

def givens_decomposition(unitary_rows):
    """Decompose a matrix into a sequence of Givens rotations.

    The input is an m x n matrix Q with m <= n. The rows of Q are orthonormal.
    Q can be decomposed as follows:

        V * Q * U^\dagger = D

    where V and U are unitary matrices, and D is an m x n matrix with the
    first m columns forming a diagonal matrix and the rest of the columns being zero.
    Furthermore, we can decompose U as

        U = G_k * ... * G_1

    where G_1, ..., G_k are complex Givens rotations, which are invertible n x n matrices.
    We describe a complex Givens rotation by the column indices (i, j) that it
    acts on, plus two angles (theta, phi) that characterize the corresponding
    2x2 unitary matrix

        [ cos(theta)    -e^{i phi} sin(theta) ]
        [ sin(theta)     e^{i phi} cos(theta) ]

    Args:
        unitary_rows: A numpy array or matrix with orthonormal rows.
    Returns:
        V: An m x m numpy array.
        givens_rotations: A list of tuples of objects describing Givens rotations.
            The list looks something like [(G1, ), (G2, G3), ... ].
            The Givens rotations within a tuple can be implemented in parallel.
            The description of a Givens rotation is itself a tuple of the form
            (i, j, theta, phi), which represents a Givens rotation of rows
            i and j by angles theta and phi.
        diagonal: A list of the nonzero entries of D.
    """
    rows = numpy.copy(unitary_rows)
    m, n = rows.shape

    # Compute V using Givens rotations
    V = numpy.eye(m, dtype=complex)
    for k in reversed(range(n - m + 1, n)):
        # Zero out entries in column k
        for l in range(m - n + k):
            # Zero out entry in row l
            G = givens_matrix_elements(rows[l, k], rows[l + 1, k])
            expanded_G = numpy.eye(m, dtype=complex)
            expanded_G[([l], [l + 1]), (l, l + 1)] = G

            rows = expanded_G.dot(rows)
            V = expanded_G.dot(V)

    # Compute the decomposition of U into Givens rotations
    givens_rotations = list()
    if m != n:
        # There are n - 1 iterations (the circuit depth is n - 1)
        for k in range(n - 1):
            # Get the (row, column) indices of elements to zero out in parallel.
            # Get the maximum number of simultaneous rotations that will be performed
            msr = min(m, n - m)
            if k < msr - 1:
                # There are k + 1 elements to zero out
                start_row = 0
                end_row = k + 1
                start_column = n - m - k
                end_column = start_column + 2 * (k + 1)
            elif k > n - 1 - msr:
                # There are n - 1 - k elements to zero out
                start_row = m - (n - 1 - k)
                end_row = m
                start_column = m - (n - 1 - k) + 1
                end_column = start_column + 2 * (n - 1 - k)
            else:
                # There are msr elements to zero out
                if msr == m:
                    start_row = 0
                    end_row = m
                    start_column = n - m - k
                    end_column = start_column + 2 * m
                else:
                    start_row = k + 1 - msr
                    end_row = k + 1
                    start_column = k + 1 - msr + 1
                    end_column = start_column + 2 * msr

            row_indices = range(start_row, end_row)
            column_indices = range(start_column, end_column, 2)
            indices_to_zero_out = zip(row_indices, column_indices)

            parallel_rotations = list()
            for i, j in indices_to_zero_out:
                # Compute the Givens rotation to zero out the (i, j) element
                a = rows[i, j - 1].conj()
                b = rows[i, j].conj()
                G = givens_matrix_elements(a, b)
                G = G[(1, 0), :]

                # Add the parameters to the list
                theta = numpy.arccos(numpy.real(G[0, 0]))
                phi = numpy.angle(G[1, 1])
                parallel_rotations.append((j - 1 , j, theta, phi))

                # Update the matrix
                expanded_G = numpy.eye(n, dtype=complex)
                expanded_G[([j - 1], [j]), (j - 1, j)] = G
                rows = rows.dot(expanded_G.T.conj())

            # Append the current list of parallel rotations to the list
            givens_rotations.append(tuple(parallel_rotations))

    diagonal = rows.diagonal()

    return V, givens_rotations, diagonal

def givens_matrix_elements(a, b):
    """Compute the matrix elements of the Givens rotation that zeroes out one of two
    row entries.

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
    G = numpy.array([[c, -phase * s],
                     [s, phase * c]], dtype=complex)
    return G
