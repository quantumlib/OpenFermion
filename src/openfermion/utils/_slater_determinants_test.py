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
import unittest

import numpy

from openfermion.config import EQ_TOLERANCE
from openfermion.transforms import get_sparse_operator
from openfermion.utils import (jw_configuration_state,
                               get_ground_state)

from openfermion.ops._quadratic_hamiltonian import givens_rotate
from openfermion.utils._sparse_tools import (
    jw_sparse_givens_rotation,
    jw_sparse_particle_hole_transformation_last_mode)
from openfermion.utils._testing_utils import (
        random_quadratic_hamiltonian,
        random_unitary_matrix)

from openfermion.utils._slater_determinants import (
        gaussian_state_preparation_circuit, givens_decomposition)


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
            state = jw_configuration_state(start_orbitals, n_qubits)

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
            state = jw_configuration_state(start_orbitals, n_qubits)

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
            Q = random_unitary_matrix(n)
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
            # Obtain a random real matrix of orthonormal rows
            Q = random_unitary_matrix(n, real=True)
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
        Q = random_unitary_matrix(m)
        Q = Q[:m, :]
        Q = Q[:m, :n]

        with self.assertRaises(ValueError):
            givens_rotations, V, diagonal = givens_decomposition(Q)

    def test_identity(self):
        n = 3
        Q = numpy.eye(n, dtype=complex)
        givens_rotations, V, diagonal = givens_decomposition(Q)

        # V should be the identity
        identity = numpy.eye(n, dtype=complex)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(V[i, j], identity[i, j])

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
        Q = random_unitary_matrix(n)
        Q = Q[:m, :]
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
