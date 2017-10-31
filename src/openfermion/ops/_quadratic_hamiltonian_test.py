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
import unittest

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import (FermionOperator,
                             QuadraticHamiltonian,
                             normal_ordered)
from openfermion.ops._quadratic_hamiltonian import (
        antisymmetric_canonical_form,
        diagonalizing_fermionic_unitary,
        majorana_operator)
from openfermion.transforms import get_fermion_operator


class QuadraticHamiltoniansTest(unittest.TestCase):

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

    def test_combined_hermitian_part(self):
        """Test getting the combined Hermitian part."""
        combined_hermitian_part = self.quad_ham_pc.combined_hermitian_part()
        for i in numpy.ndindex(combined_hermitian_part.shape):
            self.assertAlmostEqual(self.hermitian_mat[i],
                                   combined_hermitian_part[i])

        combined_hermitian_part = self.quad_ham_npc.combined_hermitian_part()
        for i in numpy.ndindex(combined_hermitian_part.shape):
            self.assertAlmostEqual(self.combined_hermitian[i],
                                   combined_hermitian_part[i])

    def test_hermitian_part(self):
        """Test getting the Hermitian part."""
        hermitian_part = self.quad_ham_pc.hermitian_part()
        for i in numpy.ndindex(hermitian_part.shape):
            self.assertAlmostEqual(self.hermitian_mat[i], hermitian_part[i])

        hermitian_part = self.quad_ham_npc.hermitian_part()
        for i in numpy.ndindex(hermitian_part.shape):
            self.assertAlmostEqual(self.hermitian_mat[i], hermitian_part[i])

    def test_antisymmetric_part(self):
        """Test getting the antisymmetric part."""
        antisymmetric_part = self.quad_ham_pc.antisymmetric_part()
        for i in numpy.ndindex(antisymmetric_part.shape):
            self.assertAlmostEqual(0., antisymmetric_part[i])

        antisymmetric_part = self.quad_ham_npc.antisymmetric_part()
        for i in numpy.ndindex(antisymmetric_part.shape):
            self.assertAlmostEqual(self.antisymmetric_mat[i],
                                   antisymmetric_part[i])

    def test_conserves_particle_number(self):
        """Test checking whether Hamiltonian conserves particle number."""
        self.assertTrue(self.quad_ham_pc.conserves_particle_number())
        self.assertFalse(self.quad_ham_npc.conserves_particle_number())

    def test_majorana_form(self):
        """Test getting the Majorana form."""
        majorana_matrix, majorana_constant = self.quad_ham_npc.majorana_form()
        # Convert the Majorana form to a FermionOperator
        majorana_op = FermionOperator((), majorana_constant)
        for i in range(2 * self.n_qubits):
            if i < self.n_qubits:
                left_op = majorana_operator((i, 1))
            else:
                left_op = majorana_operator((i - self.n_qubits, 0))
            for j in range(2 * self.n_qubits):
                if j < self.n_qubits:
                    right_op = majorana_operator((j, 1), majorana_matrix[i, j])
                else:
                    right_op = majorana_operator((j - self.n_qubits, 0),
                                                 majorana_matrix[i, j])
                majorana_op += .5j * left_op * right_op
        # Get FermionOperator for original Hamiltonian
        fermion_operator = normal_ordered(
                get_fermion_operator(self.quad_ham_npc))
        self.assertTrue(
                normal_ordered(majorana_op).isclose(fermion_operator))

    def test_ground_state_preparation_circuit(self):
        """Test obtaining the ground state preparation circuit."""
        # Test a particle-number-conserving Hamiltonian
        circuit_description = (
                self.quad_ham_pc.ground_state_preparation_circuit())
        # Test a non-particle-number-conserving Hamiltonian
        circuit_description = (
                self.quad_ham_npc.ground_state_preparation_circuit())


class DiagonalizingFermionicUnitaryTest(unittest.TestCase):

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
                    self.assertTrue(canonical[i, j] < -EQ_TOLERANCE)
                else:
                    self.assertAlmostEqual(canonical[i, j], 0.)
