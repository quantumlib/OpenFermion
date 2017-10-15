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

"""This module contains functions for preparing fermionic Gaussian states."""
from __future__ import absolute_import

import numpy
from scipy.linalg import schur

from openfermion.config import EQ_TOLERANCE
from openfermion.utils._givens_rotations import (givens_matrix_elements,
                                                 expand_two_by_two)


def fermionic_gaussian_decomposition(unitary_rows):
    """Decompose a matrix into a sequence of Givens rotations and
    particle-hole transformations on the first fermionic mode.

    The input is an n x (2 * n) matrix W  with orthonormal rows.
    Furthermore, W has the block form::

        W = [ W_1  |  W_2 ]

    where W_1 and W_2 satisfy::

        W_1 * W_1^\dagger + W_2 * W_2^\dagger = I
        W_1 * W_2^T + W_2 * W_1^T = 0

    W can be decomposed as::
        
        V * W * U^\dagger = [ 0  |  L ]

    where V and U are unitary matrices and L is an antidiagonal unitary matrix.
    Furthermore, we can decompose U as a sequence of Givens rotations
    and particle-hole transformations on the first fermionic mode.

    The decomposition of U is returned as a list of tuples of objects
    describing rotations and particle-hole transformations. The list looks
    something like [('p-h', ), (G_1, ), ('p-h', G_2), ... ].
    The objects within a tuple are either the string 'p-h', which indicates
    a particle-hole transformation on the first fermionic mode, or a tuple
    of the form (i, j, theta, phi), which indicates a Givens roation
    of rows i and j by angles theta and phi.

    Args:
        unitary_rows(ndarray): A matrix with orthonormal rows and
            additional structure described above.

    Returns:
        left_unitary(ndarray): An n x n matrix representing V.
        decomposition(list[tuple]): The decomposition of U.
        antidiagonal(ndarray): A list of the nonzero entries of L.
    """
    current_matrix = numpy.copy(unitary_rows)
    n, p = current_matrix.shape

    # Check that p = 2 * n
    if p != 2 * n:
        raise ValueError('The input matrix must have twice as many columns '
                         'as rows.')

    # Compute left_unitary using Givens rotations
    left_unitary = numpy.eye(n, dtype=complex)
    for k in reversed(range(1, n)):
        # Zero out entries in column k
        for l in range(k):
            # Zero out entry in row l
            givens_rotation = givens_matrix_elements(current_matrix[l, k],
                                                     current_matrix[l + 1, k])
            expanded_givens_rotation = expand_two_by_two(givens_rotation,
                                                         l, l + 1, n)
            current_matrix = expanded_givens_rotation.dot(current_matrix)
            left_unitary = expanded_givens_rotation.dot(left_unitary)

    # Initialize list to store decomposition of current_matrix
    decomposition = list()
    # There are 2 * n - 1 iterations (that is the circuit depth)
    for k in range(2 * n - 1):
        # Initialize the list of parallel operations to perform
        # in this iteration
        parallel_ops = list()

        # Perform a particle-hole transformation if necessary
        if k % 2 == 0 and abs(current_matrix[k // 2, 0]) > EQ_TOLERANCE:
            parallel_ops.append('p-h')
            swap_columns(current_matrix, 0, n)

        # Get the (row, column) indices of elements to zero out in parallel.
        if k < n:
            end_row = k
            end_column = k
        else:
            end_row = n - 1
            end_column = 2 * (n - 1) - k
        column_indices = range(end_column, 0, -2)
        row_indices = range(end_row, end_row - len(column_indices), -1)
        indices_to_zero_out = zip(row_indices, column_indices)

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
            parallel_ops.append((j - 1, j, theta, phi))

            # Update the matrix
            double_givens_rotate(current_matrix, givens_rotation,
                                 j - 1, j, which='col')

        # Append the current list of parallel rotations to the list
        decomposition.append(tuple(parallel_ops))

    # Get the antidiagonal entries
    antidiagonal = current_matrix[range(n), range(2 * n - 1, n - 1, -1)]
    return left_unitary, decomposition, antidiagonal


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

    if m != n:
        raise ValueError('The input matrix must be square.')
    if n % 2 != 0:
        raise ValueError('The input matrix must have even dimension')

    # Check that input matrix is antisymmetric
    matrix_plus_transpose = antisymmetric_matrix + antisymmetric_matrix.T
    maxval = numpy.max(numpy.abs(matrix_plus_transpose))
    if maxval > EQ_TOLERANCE:
        raise ValueError('The input matrix must be antisymmetric.')

    n_qubits = n // 2

    # Get the orthogonal transformation that puts antisymmetric_matrix
    # into canonical form
    canonical, orthogonal = antisymmetric_canonical_form(antisymmetric_matrix)

    # Create the matrix that converts between fermionic ladder and
    # Majorana bases
    identity = numpy.eye(n_qubits)
    majorana_basis_change = numpy.block([
            [identity, identity],
            [1.j * identity, -1.j * identity]
            ]) / numpy.sqrt(2)

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

    where D is a diagonal matrix.

    Args:
        antisymmetric_matrix(ndarray): An antisymmetric matrix with even
            dimension.

    Returns:
        canonical(ndarray): The canonical form C of antisymmetric_matrix
        orthogonal(ndarray): The orthogonal transformation R.
    """
    m, n = antisymmetric_matrix.shape

    if m != n:
        raise ValueError('The input matrix must be square.')
    if n % 2 != 0:
        raise ValueError('The input matrix must have even dimension')

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

    return canonical, orthogonal.T


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
    else:
        if p % 2 != 0:
            raise ValueError('To apply a double Givens rotation on columns, '
                             'the number of columns must be even.')
        n = p // 2
        
    if which == 'row':
        # Rotate rows i and j
        row_i = operator[i].copy()
        row_j = operator[j].copy()
        operator[i] = (givens_rotation[0, 0] * row_i
                       + givens_rotation[0, 1] * row_j)
        operator[j] = (givens_rotation[1, 0] * row_i
                       + givens_rotation[1, 1] * row_j)
        
        # Rotate rows n + i and n + j
        row_i = operator[n + i].copy()
        row_j = operator[n + j].copy()
        operator[n + i] = (givens_rotation[0, 0] * row_i
                           + givens_rotation[0, 1].conj() * row_j)
        operator[n + j] = (givens_rotation[1, 0] * row_i
                           + givens_rotation[1, 1].conj() * row_j)
    else:
        # Rotate columns i and j
        col_i = operator[:, i].copy()
        col_j = operator[:, j].copy()
        operator[:, i] = (givens_rotation[0, 0] * col_i
                          + givens_rotation[0, 1].conj() * col_j)
        operator[:, j] = (givens_rotation[1, 0] * col_i
                          + givens_rotation[1, 1].conj() * col_j)
        
        # Rotate cols n + i and n + j
        col_i = operator[:, n + i].copy()
        col_j = operator[:, n + j].copy()
        operator[:, n + i] = (givens_rotation[0, 0] * col_i
                              + givens_rotation[0, 1] * col_j)
        operator[:, n + j] = (givens_rotation[1, 0] * col_i
                              + givens_rotation[1, 1] * col_j)


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
