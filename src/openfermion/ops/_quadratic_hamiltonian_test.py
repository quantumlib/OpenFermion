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
import numpy
import scipy.sparse
import unittest

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import (normal_ordered, FermionOperator)
from openfermion.transforms import get_fermion_operator, get_sparse_operator
from openfermion.utils import get_ground_state, majorana_operator
from openfermion.utils._sparse_tools import (
        jw_sparse_givens_rotation,
        jw_sparse_particle_hole_transformation_last_mode)
from openfermion.utils._testing_utils import (random_antisymmetric_matrix,
                                              random_hermitian_matrix,
                                              random_quadratic_hamiltonian)

from openfermion.ops._quadratic_hamiltonian import (
        QuadraticHamiltonian,
        antisymmetric_canonical_form)


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
            self.hermitian_mat, constant=self.constant)

        # Initialize a non-particle-number-conserving Hamiltonian
        self.quad_ham_npc = QuadraticHamiltonian(
            self.hermitian_mat, self.antisymmetric_mat,
            self.constant, self.chemical_potential)

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

        transformation_matrix = (
                self.quad_ham_npc.diagonalizing_bogoliubov_transform())
        left_block = transformation_matrix[:, :self.n_qubits]
        right_block = transformation_matrix[:, self.n_qubits:]
        ferm_unitary = numpy.zeros((2 * self.n_qubits, 2 * self.n_qubits),
                                   dtype=complex)
        ferm_unitary[:self.n_qubits, :self.n_qubits] = left_block
        ferm_unitary[:self.n_qubits, self.n_qubits:] = right_block
        ferm_unitary[self.n_qubits:, :self.n_qubits] = numpy.conjugate(
                right_block)
        ferm_unitary[self.n_qubits:, self.n_qubits:] = numpy.conjugate(
                left_block)

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

            # Check that the eigenvalues are in the expected order
            orbital_energies, constant = (
                    quadratic_hamiltonian.orbital_energies())
            for index in range(2 ** n_qubits):
                bitstring = bin(index)[2:].zfill(n_qubits)
                subset = [j for j in range(n_qubits) if bitstring[j] == '1']
                energy = sum([orbital_energies[j] for j in subset]) + constant
                self.assertAlmostEqual(sparse_operator[index, index], energy)

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

            # Check that the eigenvalues are in the expected order
            orbital_energies, constant = (
                    quadratic_hamiltonian.orbital_energies())
            for index in range(2 ** n_qubits):
                bitstring = bin(index)[2:].zfill(n_qubits)
                subset = [j for j in range(n_qubits) if bitstring[j] == '1']
                energy = sum([orbital_energies[j] for j in subset]) + constant
                self.assertAlmostEqual(sparse_operator[index, index], energy)


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
