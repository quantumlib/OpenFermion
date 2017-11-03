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

"""Tests for slater_determinants.py."""
from __future__ import absolute_import

import numpy
import unittest
from scipy.linalg import qr

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import QuadraticHamiltonian
from openfermion.utils import (fermionic_gaussian_decomposition,
                               givens_decomposition,
                               ground_state_preparation_circuit)
from openfermion.utils._slater_determinants import (
        antisymmetric_canonical_form,
        diagonalizing_fermionic_unitary,
        double_givens_rotate,
        givens_rotate,
        swap_rows)


class GroundStatePreparationCircuitTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.constant = 1.7
        self.chemical_potential = 2.

        # Obtain random Hermitian and antisymmetric matrices
        rand_mat_A = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat_B = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat = rand_mat_A + 1.j * rand_mat_B
        self.hermitian_mat = rand_mat + rand_mat.T.conj()
        rand_mat_A = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat_B = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat = rand_mat_A + 1.j * rand_mat_B
        self.antisymmetric_mat = rand_mat - rand_mat.T

        self.combined_hermitian = (
                self.hermitian_mat -
                self.chemical_potential * numpy.eye(self.n_qubits))

        # Initialize a particle-number-conserving Hamiltonian
        self.quad_ham_pc = QuadraticHamiltonian(
                self.constant, self.hermitian_mat)

        # Initialize a non-particle-number-conserving Hamiltonian
        self.quad_ham_npc = QuadraticHamiltonian(
                self.constant, self.hermitian_mat, self.antisymmetric_mat,
                self.chemical_potential)

    def test_no_error(self):
        """Test that the procedure runs without error."""
        # Test a particle-number-conserving Hamiltonian
        circuit_description = (
                ground_state_preparation_circuit(self.quad_ham_pc))
        # Test a non-particle-number-conserving Hamiltonian
        circuit_description = (
                ground_state_preparation_circuit(self.quad_ham_npc))


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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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
            combined_givens = numpy.eye(n, dtype=complex)
            for i, j, theta, phi in reversed(parallel_set):
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                phase = numpy.exp(1.j * phi)
                G = numpy.array([[c, -phase * s],
                                [s, phase * c]], dtype=complex)
                givens_rotate(combined_givens, G, i, j)
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


class FermionicGaussianDecompositionTest(unittest.TestCase):

    def test_bad_dimensions(self):
        n, p = (3, 7)
        rand_mat = numpy.random.randn(n, p)
        with self.assertRaises(ValueError):
            left_unitary, decomposition, antidiagonal = (
                    fermionic_gaussian_decomposition(rand_mat))

    def test_bad_constraints(self):
        n = 3
        ones_mat = numpy.ones((n, 2 * n))
        with self.assertRaises(ValueError):
            left_unitary, decomposition, antidiagonal = (
                    fermionic_gaussian_decomposition(ones_mat))

    def test_n_equals_3(self):
        n = 3
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])

    def test_n_equals_4(self):
        n = 4
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])

    def test_n_equals_5(self):
        n = 5
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])

    def test_n_equals_6(self):
        n = 6
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])

    def test_n_equals_7(self):
        n = 7
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])

    def test_n_equals_8(self):
        n = 8
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])

    def test_n_equals_9(self):
        n = 9
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]

        # Get fermionic Gaussian decomposition of lower_unitary
        left_unitary, decomposition, diagonal = (
                fermionic_gaussian_decomposition(lower_unitary))

        # Check that left_unitary zeroes out the correct entries of
        # lower_unitary
        product = left_unitary.dot(lower_unitary)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                self.assertAlmostEqual(product[i, j], 0.)

        # Compute right_unitary
        right_unitary = numpy.eye(2 * n, dtype=complex)
        for parallel_set in decomposition:
            combined_op = numpy.eye(2 * n, dtype=complex)
            for op in reversed(parallel_set):
                if op == 'pht':
                    swap_rows(combined_op, n - 1, 2 * n - 1)
                else:
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    double_givens_rotate(combined_op, givens_rotation, i, j)
            right_unitary = combined_op.dot(right_unitary)

        # Compute left_unitary * lower_unitary * right_unitary^\dagger
        product = left_unitary.dot(lower_unitary.dot(right_unitary.T.conj()))

        # Construct the diagonal matrix
        diag = numpy.zeros((n, 2 * n), dtype=complex)
        diag[range(n), range(n, 2 * n)] = diagonal

        # Assert that W and D are the same
        for i in numpy.ndindex((n, 2 * n)):
            self.assertAlmostEqual(diag[i], product[i])


class DiagonalizingFermionicUnitaryTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.constant = 1.7
        self.chemical_potential = 2.

        # Obtain random Hermitian and antisymmetric matrices
        rand_mat_A = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat_B = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat = rand_mat_A + 1.j * rand_mat_B
        self.hermitian_mat = rand_mat + rand_mat.T.conj()
        rand_mat_A = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat_B = numpy.random.randn(self.n_qubits, self.n_qubits)
        rand_mat = rand_mat_A + 1.j * rand_mat_B
        self.antisymmetric_mat = rand_mat - rand_mat.T

        # Initialize a non-particle-number-conserving Hamiltonian
        self.quad_ham_npc = QuadraticHamiltonian(
                self.constant, self.hermitian_mat, self.antisymmetric_mat,
                self.chemical_potential)

    def test_diagonalizes_quadratic_hamiltonian(self):
        """Test that the unitary returned indeed diagonalizes a
        quadratic Hamiltonian."""
        hermitian_part = self.quad_ham_npc.combined_hermitian_part()
        antisymmetric_part = self.quad_ham_npc.antisymmetric_part()
        block_matrix = numpy.zeros((2 * self.n_qubits, 2 * self.n_qubits),
                                    dtype=complex)
        block_matrix[:self.n_qubits, :self.n_qubits] = antisymmetric_part
        block_matrix[:self.n_qubits, self.n_qubits:] = hermitian_part
        block_matrix[self.n_qubits:, :self.n_qubits] = -hermitian_part.conj()
        block_matrix[self.n_qubits:, self.n_qubits:] = (
                -antisymmetric_part.conj())

        majorana_matrix, majorana_constant = self.quad_ham_npc.majorana_form()
        canonical, orthogonal = antisymmetric_canonical_form(majorana_matrix)
        ferm_unitary = diagonalizing_fermionic_unitary(majorana_matrix)
        diagonalized = ferm_unitary.conj().dot(
                block_matrix.dot(ferm_unitary.T.conj()))
        for i in numpy.ndindex((2 * self.n_qubits, 2 * self.n_qubits)):
            self.assertAlmostEqual(diagonalized[i], canonical[i])

    def test_bad_dimensions(self):
        n, p = (3, 4)
        ones_mat = numpy.ones((n, p))
        with self.assertRaises(ValueError):
            ferm_unitary = diagonalizing_fermionic_unitary(ones_mat)

    def test_not_antisymmetric(self):
        n = 4
        ones_mat = numpy.ones((n, n))
        with self.assertRaises(ValueError):
            ferm_unitary = diagonalizing_fermionic_unitary(ones_mat)

    def test_n_equals_3(self):
        n = 3
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T

        # Get the diagonalizing fermionic unitary
        ferm_unitary = diagonalizing_fermionic_unitary(antisymmetric_matrix)
        lower_unitary = ferm_unitary[n:]
        lower_left = lower_unitary[:, :n]
        lower_right = lower_unitary[:, n:]

        # Check that lower_left and lower_right satisfy the constraints
        # necessary for the transformed fermionic operators to satisfy
        # the fermionic anticommutation relations
        constraint_matrix_1 = (lower_left.dot(lower_left.T.conj()) +
                               lower_right.dot(lower_right.T.conj()))
        constraint_matrix_2 = (lower_left.dot(lower_right.T) +
                               lower_right.dot(lower_left.T))

        identity = numpy.eye(n, dtype=complex)
        for i in numpy.ndindex((n, n)):
            self.assertAlmostEqual(identity[i], constraint_matrix_1[i])
            self.assertAlmostEqual(0., constraint_matrix_2[i])


class AntisymmetricCanonicalFormTest(unittest.TestCase):

    def test_equality(self):
        """Test that the decomposition is valid."""
        n = 7
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T
        canonical, orthogonal = antisymmetric_canonical_form(
                antisymmetric_matrix)
        result_matrix = orthogonal.dot(antisymmetric_matrix.dot(orthogonal.T))
        for i in numpy.ndindex(result_matrix.shape):
            self.assertAlmostEqual(result_matrix[i], canonical[i])

    def test_canonical(self):
        """Test that the returned canonical matrix has the right form."""
        n = 7
        # Obtain a random antisymmetric matrix
        rand_mat = numpy.random.randn(2 * n, 2 * n)
        antisymmetric_matrix = rand_mat - rand_mat.T
        canonical, orthogonal = antisymmetric_canonical_form(
                antisymmetric_matrix)
        for i in range(2 * n):
            for j in range(2 * n):
                if i < n and j == n + i:
                    self.assertTrue(canonical[i, j] > -EQ_TOLERANCE)
                elif i >= n and j == i - n:
                    self.assertTrue(canonical[i, j] < EQ_TOLERANCE)
                else:
                    self.assertAlmostEqual(canonical[i, j], 0.)
