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
