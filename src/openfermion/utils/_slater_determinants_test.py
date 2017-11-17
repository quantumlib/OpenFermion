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
from openfermion.ops._quadratic_hamiltonian import (
        diagonalizing_fermionic_unitary, swap_rows)
from openfermion.ops._quadratic_hamiltonian_test import (
        random_hermitian_matrix, random_antisymmetric_matrix)
from openfermion.transforms import get_sparse_operator
from openfermion.utils import (fermionic_gaussian_decomposition,
                               get_ground_state,
                               givens_decomposition,
                               gaussian_state_preparation_circuit,
                               jw_get_gaussian_state,
                               jw_slater_determinant)
from openfermion.utils._slater_determinants import (
        double_givens_rotate,
        givens_rotate,
        givens_matrix_elements)
from openfermion.utils._sparse_tools import (
        jw_sparse_givens_rotation,
        jw_sparse_particle_hole_transformation_last_mode)


class GaussianStatePreparationCircuitTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits_range = range(3, 6)

    def test_ground_state_particle_conserving(self):
        """Test getting the ground state preparation circuit for a Hamiltonian
        that conserves particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                    n_qubits, True, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, ground_state = get_ground_state(sparse_operator)

            # Obtain the circuit
            circuit_description, start_orbitals = (
                    gaussian_state_preparation_circuit(quadratic_hamiltonian))

            # Initialize the starting state
            state = jw_slater_determinant(start_orbitals, n_qubits)

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

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * state - ground_energy * state
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_ground_state_particle_nonconserving(self):
        """Test getting the ground state preparation circuit for a Hamiltonian
        that does not conserve particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                    n_qubits, False, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, ground_state = get_ground_state(sparse_operator)

            # Obtain the circuit
            circuit_description, start_orbitals = (
                    gaussian_state_preparation_circuit(quadratic_hamiltonian))

            # Initialize the starting state
            state = jw_slater_determinant(start_orbitals, n_qubits)

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

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * state - ground_energy * state
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            description, n_electrons = gaussian_state_preparation_circuit('a')


class GivensDecompositionTest(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [(3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
                                (4, 7), (4, 8), (4, 9)]

    def test_main_procedure(self):
        for m, n in self.test_dimensions:
            # Obtain a random matrix of orthonormal rows
            x = numpy.random.randn(n, n)
            y = numpy.random.randn(n, n)
            A = x + 1.j*y
            Q, R = qr(A)
            Q = Q[:m, :]

            # Get Givens decomposition of Q
            givens_rotations, V, diagonal = givens_decomposition(Q)

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

    def test_real_numbers(self):
        for m, n in self.test_dimensions:
            # Obtain a random matrix of orthonormal rows
            A = numpy.random.randn(n, n)
            Q, R = qr(A)
            Q = Q[:m, :]

            # Get Givens decomposition of Q
            givens_rotations, V, diagonal = givens_decomposition(Q)

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

    def test_bad_dimensions(self):
        m, n = (3, 2)

        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(m, m)
        y = numpy.random.randn(m, m)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :n]

        with self.assertRaises(ValueError):
            givens_rotations, V, diagonal = givens_decomposition(Q)

    def test_identity(self):
        n = 3
        Q = numpy.eye(n, dtype=complex)
        givens_rotations, V, diagonal = givens_decomposition(Q)

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
        givens_rotations, V, diagonal = givens_decomposition(Q)

        # There should be no Givens rotations
        self.assertEqual(givens_rotations, list())

        # VQ should equal the diagonal
        VQ = V.dot(Q)
        D = numpy.zeros((m, n), dtype=complex)
        D[numpy.diag_indices(m)] = diagonal
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(VQ[i, j], D[i, j])

    def test_square(self):
        m, n = (3, 3)
        # Obtain a random matrix of orthonormal rows
        x = numpy.random.randn(n, n)
        y = numpy.random.randn(n, n)
        A = x + 1.j*y
        Q, R = qr(A)
        Q = Q[:m, :]

        # Get Givens decomposition of Q
        givens_rotations, V, diagonal = givens_decomposition(Q)

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


class FermionicGaussianDecompositionTest(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [3, 4, 5, 6, 7, 8, 9]

    def test_main_procedure(self):
        for n in self.test_dimensions:
            # Obtain a random antisymmetric matrix
            antisymmetric_mat = random_antisymmetric_matrix(2 * n, real=True)

            # Get the diagonalizing fermionic unitary
            ferm_unitary = diagonalizing_fermionic_unitary(
                    antisymmetric_mat)
            lower_unitary = ferm_unitary[n:]

            # Get fermionic Gaussian decomposition of lower_unitary
            decomposition, left_decomposition, diagonal, left_diagonal = (
                    fermionic_gaussian_decomposition(lower_unitary))

            # Compute left_unitary
            left_unitary = numpy.eye(n, dtype=complex)
            for parallel_set in left_decomposition:
                combined_op = numpy.eye(n, dtype=complex)
                for op in reversed(parallel_set):
                    i, j, theta, phi = op
                    c = numpy.cos(theta)
                    s = numpy.sin(theta)
                    phase = numpy.exp(1.j * phi)
                    givens_rotation = numpy.array(
                            [[c, -phase * s],
                             [s, phase * c]], dtype=complex)
                    givens_rotate(combined_op, givens_rotation, i, j)
                left_unitary = combined_op.dot(left_unitary)
            for i in range(n):
                left_unitary[i] *= left_diagonal[i]
            left_unitary = left_unitary.T
            for i in range(n):
                left_unitary[i] *= diagonal[i]

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
                        double_givens_rotate(
                                combined_op, givens_rotation, i, j)
                right_unitary = combined_op.dot(right_unitary)

            # Compute left_unitary * lower_unitary * right_unitary^\dagger
            product = left_unitary.dot(lower_unitary.dot(
                                       right_unitary.T.conj()))

            # Construct the diagonal matrix
            diag = numpy.zeros((n, 2 * n), dtype=complex)
            diag[range(n), range(n, 2 * n)] = diagonal

            # Assert that W and D are the same
            for i in numpy.ndindex((n, 2 * n)):
                self.assertAlmostEqual(diag[i], product[i])

    def test_bad_dimensions(self):
        n, p = (3, 7)
        rand_mat = numpy.random.randn(n, p)
        with self.assertRaises(ValueError):
            decomposition, left_unitary, antidiagonal = (
                    fermionic_gaussian_decomposition(rand_mat))

    def test_bad_constraints(self):
        n = 3
        ones_mat = numpy.ones((n, 2 * n))
        with self.assertRaises(ValueError):
            decomposition, left_unitary, antidiagonal = (
                    fermionic_gaussian_decomposition(ones_mat))


class GivensMatrixElementsTest(unittest.TestCase):

    def setUp(self):
        self.num_test_repetitions = 5

    def test_already_zero(self):
        """Test when some entries are already zero."""
        # Test when left entry is zero
        v = numpy.array([0., numpy.random.randn()])
        G = givens_matrix_elements(v[0], v[1])
        self.assertAlmostEqual(G.dot(v)[0], 0.)
        # Test when right entry is zero
        v = numpy.array([numpy.random.randn(), 0.])
        G = givens_matrix_elements(v[0], v[1])
        self.assertAlmostEqual(G.dot(v)[0], 0.)

    def test_real(self):
        """Test that the procedure works for real numbers."""
        for _ in range(self.num_test_repetitions):
            v = numpy.random.randn(2)
            G_left = givens_matrix_elements(v[0], v[1], which='left')
            G_right = givens_matrix_elements(v[0], v[1], which='right')
            self.assertAlmostEqual(G_left.dot(v)[0], 0.)
            self.assertAlmostEqual(G_right.dot(v)[1], 0.)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            v = numpy.random.randn(2)
            G = givens_matrix_elements(v[0], v[1], which='a')


class GivensRotateTest(unittest.TestCase):

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            v = numpy.random.randn(2)
            G = givens_matrix_elements(v[0], v[1])
            givens_rotate(v, G, 0, 1, which='a')


class DoubleGivensRotateTest(unittest.TestCase):

    def test_odd_dimension(self):
        """Test that it raises an error for odd-dimensional input."""
        A = numpy.random.randn(3, 3)
        v = numpy.random.randn(2)
        G = givens_matrix_elements(v[0], v[1])
        with self.assertRaises(ValueError):
            double_givens_rotate(A, G, 0, 1, which='row')
        with self.assertRaises(ValueError):
            double_givens_rotate(A, G, 0, 1, which='col')

    def test_bad_input(self):
        """Test bad input."""
        A = numpy.random.randn(3, 3)
        v = numpy.random.randn(2)
        G = givens_matrix_elements(v[0], v[1])
        with self.assertRaises(ValueError):
            double_givens_rotate(A, G, 0, 1, which='a')


class JWSparseGivensRotationTest(unittest.TestCase):

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            givens_matrix = jw_sparse_givens_rotation(0, 2, 1., 1., 5)
        with self.assertRaises(ValueError):
            givens_matrix = jw_sparse_givens_rotation(4, 5, 1., 1., 5)


def random_quadratic_hamiltonian(n_qubits,
                                 conserves_particle_number=False,
                                 real=False):
    """Generate a random instance of QuadraticHamiltonian

    Args:
        n_qubits(int): the number of qubits
        conserves_particle_number(bool): whether the returned Hamiltonian
            should conserve particle number
        real(bool): whether to use only real numbers

    Returns:
        QuadraticHamiltonian
    """
    constant = numpy.random.randn()
    chemical_potential = numpy.random.randn()
    hermitian_mat = random_hermitian_matrix(n_qubits, real)
    if conserves_particle_number:
        antisymmetric_mat = None
    else:
        antisymmetric_mat = random_antisymmetric_matrix(n_qubits, real)
    return QuadraticHamiltonian(constant, hermitian_mat,
                                antisymmetric_mat, chemical_potential)
