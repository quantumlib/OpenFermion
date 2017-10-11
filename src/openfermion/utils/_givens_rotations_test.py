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

"""Tests for sparse_tools.py."""
from __future__ import absolute_import

import numpy
import unittest
from scipy.linalg import qr

from openfermion.utils import givens_decomposition
from openfermion.utils._givens_rotations import expand_two_by_two

class GivensDecompositionTest(unittest.TestCase):
    def test_3_by_6(self):
        m, n = (3, 6)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of U
        V, givens_rotations, diagonal = givens_decomposition(Q)

        # Compute U
        U = numpy.eye(n, dtype=complex)
        for parallel_set in givens_rotations:
            combined_givens = numpy.eye(n)
            for i, j, theta, phi in parallel_set:
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                expanded_G = expand_two_by_two(G, i, j, n)
                combined_givens = combined_givens.dot(expanded_G)
            U = combined_givens.dot(U)

        # Compute V * Q * U^\dagger
        W = V.dot(Q.dot(U.T.conj()))

        # Construct the diagonal matrix
        D = numpy.zeros((m, n), dtype=complex)
        D[numpy.diag_indices(m)] = diagonal

        # Assert that W and D are the same
        for i in range(m):
            for j in range(n):
                self.assertAlmostEqual(D[i, j], W[i, j])
