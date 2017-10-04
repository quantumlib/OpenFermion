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

def givens_decomposition(unitary_columns):
    """Decompose a matrix into a sequence of Givens rotations.

    The input is an m x n matrix U with m >= n. The columns of U are orthonormal.
    U can be decomposed as follows:

        G_k * ... * G_1 * U = D

    where G_1, ..., G_k are complex Givens rotations (invertible m x m matrices)
    and D is an m x n matrix with the first n rows forming a diagonal matrix and
    the rest of the rows being zero. This is related to the QR decomposition.

    Args:
        unitary_columns: A numpy array or matrix with orthonormal columns.
    
    Returns:
        givens_rotations: A list of tuples of the form (i, j, z), which
            represents a Givens rotation of the coordinates i and j
            by complex angle z. The first element of the list represents G_1.
        diagonal: A list of the nonzero entries of D.
    """
    return
