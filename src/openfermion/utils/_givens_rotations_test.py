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

"""Tests for givens_rotations.py."""
from __future__ import absolute_import

import numpy
import unittest
from scipy.linalg import qr

from openfermion.utils import givens_decomposition
from openfermion.utils._givens_rotations import expand_two_by_two


class GivensDecompositionTest(unittest.TestCase):

    def test_bad_dimensions(self):
        m, n = (3, 2)

        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(m, m)
        y = numpy.random.randn(m, m)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :n]

        with self.assertRaises(ValueError):
            V, givens_rotations, diagonal = givens_decomposition(Q)

    def test_identity(self):
        n = 3
        Q = numpy.eye(n, dtype=complex)
        V, givens_rotations, diagonal = givens_decomposition(Q)

        # V should be the identity
        I = numpy.eye(n, dtype=complex)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(V[i, j], I[i, j])

        # There should be no Givens rotations
        self.assertEqual(givens_rotations, list())

        # The diagonal should be ones
        for d in diagonal:
            self.assertAlmostEqual(d, 1.)

    def test_antidiagonal(self):
        m, n = (3, 3)
        Q = numpy.zeros((m, n), dtype=complex)
        Q[0, 2] = 1.
        Q[1, 1] = 1.
        Q[2, 0] = 1.
        V, givens_rotations, diagonal = givens_decomposition(Q)

        # There should be no Givens rotations
        self.assertEqual(givens_rotations, list())

        # VQ should equal the diagonal
        VQ = V.dot(Q)
        D = numpy.zeros((m, n), dtype=complex)
        D[numpy.diag_indices(m)] = diagonal
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(VQ[i, j], D[i, j])

    def test_3_by_3(self):
        m, n = (3, 3)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
        V, givens_rotations, diagonal = givens_decomposition(Q)

        # There should be no Givens rotations
        self.assertEqual(givens_rotations, list())

        # Compute V * Q * U^\dagger
        W = V.dot(Q)

        # Construct the diagonal matrix
        D = numpy.zeros((m, n), dtype=complex)
        D[numpy.diag_indices(m)] = diagonal

        # Assert that W and D are the same
        for i in range(m):
            for j in range(n):
                self.assertAlmostEqual(D[i, j], W[i, j])

    def test_3_by_4(self):
        m, n = (3, 4)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_3_by_5(self):
        m, n = (3, 5)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_3_by_6(self):
        m, n = (3, 6)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_3_by_7(self):
        m, n = (3, 7)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_3_by_8(self):
        m, n = (3, 8)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_3_by_9(self):
        m, n = (3, 9)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_4_by_5(self):
        m, n = (4, 5)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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

    def test_4_by_9(self):
        m, n = (4, 9)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
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
