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

"""Tests for quadratic_hamiltonian.py."""
from __future__ import absolute_import

import numpy
import scipy.sparse
import unittest

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import (normal_ordered, FermionOperator)
from openfermion.transforms import get_fermion_operator, get_sparse_operator
from openfermion.utils import (
        get_ground_state, jw_configuration_state, majorana_operator)
from openfermion.utils._sparse_tools import (
        jw_sparse_givens_rotation,
        jw_sparse_occupation_phase,
        jw_sparse_particle_hole_transformation_last_mode)
from openfermion.utils._testing_utils import (random_antisymmetric_matrix,
                                              random_hermitian_matrix,
                                              random_quadratic_hamiltonian,
                                              random_unitary_matrix)

from openfermion.ops._quadratic_hamiltonian import (
        QuadraticHamiltonian,
        antisymmetric_canonical_form,
        double_givens_rotate,
        fermionic_gaussian_decomposition,
        givens_decomposition,
        givens_matrix_elements,
        givens_rotate,
        swap_rows)


class QuadraticHamiltonianTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5
        self.constant = 1.7
        self.chemical_potential = 2.

        # Obtain random Hermitian and antisymmetric matrices
        self.hermitian_mat = random_hermitian_matrix(self.n_qubits)
        self.antisymmetric_mat = random_antisymmetric_matrix(self.n_qubits)

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

        # Initialize the sparse operators and get their ground energies
        self.quad_ham_pc_sparse = get_sparse_operator(self.quad_ham_pc)
        self.quad_ham_npc_sparse = get_sparse_operator(self.quad_ham_npc)

        self.pc_ground_energy, self.pc_ground_state = get_ground_state(
            self.quad_ham_pc_sparse)
        self.npc_ground_energy, self.npc_ground_state = get_ground_state(
            self.quad_ham_npc_sparse)

    def test_combined_hermitian_part(self):
        """Test getting the combined Hermitian part."""
        combined_hermitian_part = self.quad_ham_pc.combined_hermitian_part
        for i in numpy.ndindex(combined_hermitian_part.shape):
            self.assertAlmostEqual(self.hermitian_mat[i],
                                   combined_hermitian_part[i])

        combined_hermitian_part = self.quad_ham_npc.combined_hermitian_part
        for i in numpy.ndindex(combined_hermitian_part.shape):
            self.assertAlmostEqual(self.combined_hermitian[i],
                                   combined_hermitian_part[i])

    def test_hermitian_part(self):
        """Test getting the Hermitian part."""
        hermitian_part = self.quad_ham_pc.hermitian_part
        for i in numpy.ndindex(hermitian_part.shape):
            self.assertAlmostEqual(self.hermitian_mat[i], hermitian_part[i])

        hermitian_part = self.quad_ham_npc.hermitian_part
        for i in numpy.ndindex(hermitian_part.shape):
            self.assertAlmostEqual(self.hermitian_mat[i], hermitian_part[i])

    def test_antisymmetric_part(self):
        """Test getting the antisymmetric part."""
        antisymmetric_part = self.quad_ham_pc.antisymmetric_part
        for i in numpy.ndindex(antisymmetric_part.shape):
            self.assertAlmostEqual(0., antisymmetric_part[i])

        antisymmetric_part = self.quad_ham_npc.antisymmetric_part
        for i in numpy.ndindex(antisymmetric_part.shape):
            self.assertAlmostEqual(self.antisymmetric_mat[i],
                                   antisymmetric_part[i])

    def test_conserves_particle_number(self):
        """Test checking whether Hamiltonian conserves particle number."""
        self.assertTrue(self.quad_ham_pc.conserves_particle_number)
        self.assertFalse(self.quad_ham_npc.conserves_particle_number)

    def test_add_chemical_potential(self):
        """Test adding a chemical potential."""
        self.quad_ham_npc.add_chemical_potential(2.4)

        combined_hermitian_part = self.quad_ham_npc.combined_hermitian_part
        hermitian_part = self.quad_ham_npc.hermitian_part

        want_combined = (self.combined_hermitian -
                         2.4 * numpy.eye(self.n_qubits))
        want_hermitian = self.hermitian_mat

        for i in numpy.ndindex(combined_hermitian_part.shape):
            self.assertAlmostEqual(combined_hermitian_part[i],
                                   want_combined[i])

        for i in numpy.ndindex(hermitian_part.shape):
            self.assertAlmostEqual(hermitian_part[i], want_hermitian[i])

        self.assertAlmostEqual(2.4 + self.chemical_potential,
                               self.quad_ham_npc.chemical_potential)

    def test_orbital_energies(self):
        """Test getting the orbital energies."""
        # Test the particle-number-conserving case
        orbital_energies, constant = self.quad_ham_pc.orbital_energies()
        # Test the ground energy
        energy = numpy.sum(
            orbital_energies[orbital_energies < -EQ_TOLERANCE]) + constant
        self.assertAlmostEqual(energy, self.pc_ground_energy)

        # Test the non-particle-number-conserving case
        orbital_energies, constant = self.quad_ham_npc.orbital_energies()
        # Test the ground energy
        energy = constant
        self.assertAlmostEqual(energy, self.npc_ground_energy)

    def test_ground_energy(self):
        """Test getting the ground energy."""
        # Test particle-number-conserving case
        energy = self.quad_ham_pc.ground_energy()
        self.assertAlmostEqual(energy, self.pc_ground_energy)
        # Test non-particle-number-conserving case
        energy = self.quad_ham_npc.ground_energy()
        self.assertAlmostEqual(energy, self.npc_ground_energy)

    def test_majorana_form(self):
        """Test getting the Majorana form."""
        majorana_matrix, majorana_constant = self.quad_ham_npc.majorana_form()
        # Convert the Majorana form to a FermionOperator
        majorana_op = FermionOperator((), majorana_constant)
        normalization = 1. / numpy.sqrt(2.)
        for i in range(2 * self.n_qubits):
            if i < self.n_qubits:
                left_op = majorana_operator((i, 0), normalization)
            else:
                left_op = majorana_operator((i - self.n_qubits, 1),
                                            normalization)
            for j in range(2 * self.n_qubits):
                if j < self.n_qubits:
                    right_op = majorana_operator((j, 0),
                            majorana_matrix[i, j] * normalization)
                else:
                    right_op = majorana_operator((j - self.n_qubits, 1),
                            majorana_matrix[i, j] * normalization)
                majorana_op += .5j * left_op * right_op
        # Get FermionOperator for original Hamiltonian
        fermion_operator = normal_ordered(
            get_fermion_operator(self.quad_ham_npc))
        self.assertTrue(
            normal_ordered(majorana_op) == fermion_operator)

    def test_diagonalizing_bogoliubov_transform(self):
        """Test getting the diagonalizing Bogoliubov transformation."""
        hermitian_part = self.quad_ham_npc.combined_hermitian_part
        antisymmetric_part = self.quad_ham_npc.antisymmetric_part
        block_matrix = numpy.zeros((2 * self.n_qubits, 2 * self.n_qubits),
                                   dtype=complex)
        block_matrix[:self.n_qubits, :self.n_qubits] = antisymmetric_part
        block_matrix[:self.n_qubits, self.n_qubits:] = hermitian_part
        block_matrix[self.n_qubits:, :self.n_qubits] = -hermitian_part.conj()
        block_matrix[self.n_qubits:, self.n_qubits:] = (
            -antisymmetric_part.conj())

        ferm_unitary = self.quad_ham_npc.diagonalizing_bogoliubov_transform()

        # Check that the transformation is diagonalizing
        majorana_matrix, majorana_constant = self.quad_ham_npc.majorana_form()
        canonical, orthogonal = antisymmetric_canonical_form(majorana_matrix)
        diagonalized = ferm_unitary.conj().dot(
            block_matrix.dot(ferm_unitary.T.conj()))
        for i in numpy.ndindex((2 * self.n_qubits, 2 * self.n_qubits)):
            self.assertAlmostEqual(diagonalized[i], canonical[i])

        lower_unitary = ferm_unitary[self.n_qubits:]
        lower_left = lower_unitary[:, :self.n_qubits]
        lower_right = lower_unitary[:, self.n_qubits:]

        # Check that lower_left and lower_right satisfy the constraints
        # necessary for the transformed fermionic operators to satisfy
        # the fermionic anticommutation relations
        constraint_matrix_1 = (lower_left.dot(lower_left.T.conj()) +
                               lower_right.dot(lower_right.T.conj()))
        constraint_matrix_2 = (lower_left.dot(lower_right.T) +
                               lower_right.dot(lower_left.T))

        identity = numpy.eye(self.n_qubits, dtype=complex)
        for i in numpy.ndindex((self.n_qubits, self.n_qubits)):
            self.assertAlmostEqual(identity[i], constraint_matrix_1[i])
            self.assertAlmostEqual(0., constraint_matrix_2[i])


class DiagonalizingCircuitTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits_range = range(3, 9)

    def test_particle_conserving(self):
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, True)
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)

            # Diagonalize the Hamiltonian using the circuit
            circuit_description = reversed(
                    quadratic_hamiltonian.diagonalizing_circuit())

            for parallel_ops in circuit_description:
                for op in parallel_ops:
                    if len(op) == 2:
                        j, phi = op
                        gate = jw_sparse_occupation_phase(j, phi, n_qubits)
                    else:
                        i, j, theta, phi = op
                        gate = jw_sparse_givens_rotation(
                                i, j, theta, phi, n_qubits)
                    sparse_operator = (
                            gate.getH().dot(sparse_operator).dot(gate))

            # Check that the result is diagonal
            diag = scipy.sparse.diags(sparse_operator.diagonal())
            difference = sparse_operator - diag
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_non_particle_conserving(self):
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, False)
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)

            # Diagonalize the Hamiltonian using the circuit
            circuit_description = reversed(
                    quadratic_hamiltonian.diagonalizing_circuit())

            particle_hole_transformation = (
                jw_sparse_particle_hole_transformation_last_mode(n_qubits))
            for parallel_ops in circuit_description:
                for op in parallel_ops:
                    if op == 'pht':
                        gate = particle_hole_transformation
                    elif len(op) == 2:
                        j, phi = op
                        gate = jw_sparse_occupation_phase(j, phi, n_qubits)
                    else:
                        i, j, theta, phi = op
                        gate = jw_sparse_givens_rotation(
                                i, j, theta, phi, n_qubits)
                    sparse_operator = (
                            gate.getH().dot(sparse_operator).dot(gate))

            # Check that the result is diagonal
            diag = scipy.sparse.diags(sparse_operator.diagonal())
            difference = sparse_operator - diag
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)


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

        diagonal = canonical[range(n), range(n, 2 * n)]
        for i in range(n - 1):
            self.assertTrue(diagonal[i] <= diagonal[i + 1])

    def test_bad_dimensions(self):
        n, p = (3, 4)
        ones_mat = numpy.ones((n, p))
        with self.assertRaises(ValueError):
            _ = antisymmetric_canonical_form(ones_mat)

    def test_not_antisymmetric(self):
        n = 4
        ones_mat = numpy.ones((n, n))
        with self.assertRaises(ValueError):
            _ = antisymmetric_canonical_form(ones_mat)


class FermionicGaussianDecompositionTest(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [3, 4, 5, 6, 7, 8, 9]

    def test_main_procedure(self):
        for n in self.test_dimensions:
            # Obtain a random quadratic Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n)

            # Get the diagonalizing fermionic unitary
            ferm_unitary = (
                quadratic_hamiltonian.diagonalizing_bogoliubov_transform())
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
                        double_givens_rotate(combined_op, givens_rotation,
                                             i, j)
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
