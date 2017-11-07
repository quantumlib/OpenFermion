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
from openfermion.transforms import get_fermion_operator
from openfermion.utils import (fermionic_gaussian_decomposition,
                               get_ground_state,
                               givens_decomposition,
                               ground_state_preparation_circuit,
                               jordan_wigner_sparse,
                               jw_get_quadratic_hamiltonian_ground_state)
from openfermion.utils._slater_determinants import (
        antisymmetric_canonical_form,
        diagonalizing_fermionic_unitary,
        double_givens_rotate,
        givens_rotate,
        givens_matrix_elements,
        swap_rows)

class GroundStatePreparationCircuitTest(unittest.TestCase):

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            description, n_electrons = ground_state_preparation_circuit('a')

class GetQuadraticHamiltonianGroundStateTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits_range = range(2, 10)

    def test_particle_conserving(self):
        """Test the case that the Hamiltonian conserves particle number."""
        constant = 1.7
        chemical_potential = 2.4
        for n_qubits in self.n_qubits_range:
            # Obtain a random Hermitian matrix
            rand_mat_A = numpy.random.randn(n_qubits, n_qubits)
            rand_mat_B = numpy.random.randn(n_qubits, n_qubits)
            rand_mat = rand_mat_A + 1.j * rand_mat_B
            hermitian_mat = rand_mat + rand_mat.T.conj()

            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = QuadraticHamiltonian(
                    constant, hermitian_mat,
                    chemical_potential=chemical_potential)

            # Compute the true ground state
            fermion_operator = get_fermion_operator(quadratic_hamiltonian)
            sparse_operator = jordan_wigner_sparse(fermion_operator)
            ground_energy, ground_state = get_ground_state(sparse_operator)

            # Compute the ground state using the circuit
            circuit_energy, circuit_state = (
                    jw_get_quadratic_hamiltonian_ground_state(
                        quadratic_hamiltonian))

            # Check that the energies match
            self.assertAlmostEqual(ground_energy, circuit_energy)

            # Check that the state obtained using the circuit is a ground state
            difference = (sparse_operator * circuit_state -
                          ground_energy * circuit_state)
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_particle_nonconserving(self):
        """Test the case that the Hamiltonian does not conserve particle
        number."""
        constant = 1.7
        chemical_potential = 2.4
        for n_qubits in self.n_qubits_range:
            # Obtain random Hermitian and antisymmetric matrices
            rand_mat_A = numpy.random.randn(n_qubits, n_qubits)
            rand_mat_B = numpy.random.randn(n_qubits, n_qubits)
            rand_mat = rand_mat_A + 1.j * rand_mat_B
            hermitian_mat = rand_mat + rand_mat.T.conj()
            rand_mat_A = numpy.random.randn(n_qubits, n_qubits)
            rand_mat_B = numpy.random.randn(n_qubits, n_qubits)
            rand_mat = rand_mat_A + 1.j * rand_mat_B
            antisymmetric_mat = rand_mat - rand_mat.T

            # Initialize a non-particle-number-conserving Hamiltonian
            quadratic_hamiltonian = QuadraticHamiltonian(
                    constant, hermitian_mat, antisymmetric_mat,
                    chemical_potential)

            # Compute the true ground state
            fermion_operator = get_fermion_operator(quadratic_hamiltonian)
            sparse_operator = jordan_wigner_sparse(fermion_operator)
            ground_energy, ground_state = get_ground_state(sparse_operator)

            # Compute the ground state using the circuit
            circuit_energy, circuit_state = (
                    jw_get_quadratic_hamiltonian_ground_state(
                        quadratic_hamiltonian))

            # Check that the energies match
            self.assertAlmostEqual(ground_energy, circuit_energy)

            # Check that the state obtained using the circuit is a ground state
            difference = (sparse_operator * circuit_state -
                          ground_energy * circuit_state)
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            energy, state = jw_get_quadratic_hamiltonian_ground_state('a')


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

    def test_real_numbers(self):
        for m, n in self.test_dimensions:
            # Obtain a random matrix of orthonormal rows
            A = numpy.random.randn(n, n)
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

    def test_square(self):
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


class FermionicGaussianDecompositionTest(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [3, 4, 5, 6, 7, 8, 9]

    def test_main_procedure(self):
        for n in self.test_dimensions:
            # Obtain a random antisymmetric matrix
            rand_mat = numpy.random.randn(2 * n, 2 * n)
            antisymmetric_matrix = rand_mat - rand_mat.T

            # Get the diagonalizing fermionic unitary
            ferm_unitary = diagonalizing_fermionic_unitary(
                    antisymmetric_matrix)
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
            left_unitary, decomposition, antidiagonal = (
                    fermionic_gaussian_decomposition(rand_mat))

    def test_bad_constraints(self):
        n = 3
        ones_mat = numpy.ones((n, 2 * n))
        with self.assertRaises(ValueError):
            left_unitary, decomposition, antidiagonal = (
                    fermionic_gaussian_decomposition(ones_mat))


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
