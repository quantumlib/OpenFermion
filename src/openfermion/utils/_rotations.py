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

def trivial_givens_decomposition(columns):
    m, n = columns.shape

    givens_rotations = []

    if m > n:
        num_columns_to_zero_out = n
    else:
        num_columns_to_zero_out = n - 1

    for k in range(num_columns_to_zero_out):
        # Zero out entries in column k
        for l in reversed(range(k + 1, m)):
            # Zero out entries in row l
            f, g = columns[(l - 1, l), k]
            # We want to rotate f and g to zero out g
            f2 = abs(f) ** 2
            g2 = abs(g) ** 2
            denom = numpy.sqrt(f2 + g2)

            c = abs(f) / denom
            s = (f / abs(f)) * g.conjugate() / denom

            givens_rotations.append((l, k, numpy.arccos(c), numpy.angle(s)))

            # Apply givens rotation to matrix
            old_upper_row = numpy.copy(columns[l - 1])
            old_lower_row = numpy.copy(columns[l])
            columns[l - 1] = c * old_upper_row + s * old_lower_row
            columns[l] = -s.conjugate() * old_upper_row + c * old_lower_row

    diagonal = columns.diagonal()

    return givens_rotations, diagonal, columns

def givens_decomposition(unitary_columns):
    """Decompose a matrix into a sequence of Givens rotations.

    The input is an m x n matrix U with m >= n. The columns of U are orthonormal.
    U can be decomposed as follows:

        G_k * ... * G_1 * U = D

    where G_1, ..., G_k are complex Givens rotations (invertible m x m matrices)
    and D is an m x n matrix with the first n rows forming a diagonal matrix and
    the rest of the rows being zero. This gives a QR decomposition of U.
    We describe a complex Givens rotation by the coordinates (i, j) that it
    acts on, plus two angles (theta, phi) that characterize the corresponding
    2x2 unitary matrix
        [ cos(theta)                e^{i phi} sin(theta) ]
        [ -e^{-i phi} sin(theta)    cos(theta)           ]

    Args:
        unitary_columns: A numpy array or matrix with orthonormal columns.
    
    Returns:
        givens_rotations: A list of tuples of the form (i, j, theta, phi), which
            represents a Givens rotation of the coordinates i and j
            by angles theta and phi. The first element of the list represents G_1.
        diagonal: A list of the nonzero entries of D.
    """
    return
