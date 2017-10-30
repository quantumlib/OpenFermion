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

from openfermion.ops import QuadraticHamiltonian


class QuadraticHamiltoniansTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.constant = 1.
        self.chemical_potential = 2.

        # Obtain random Hermitian and antisymmetric matrices
        rand_mat = numpy.random.randn(self.n_qubits, self.n_qubits)
        self.hermitian_mat = rand_mat + rand_mat.T.conj()
        rand_mat = numpy.random.randn(self.n_qubits, self.n_qubits)
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
        majorana_matrix = self.quad_ham_npc.majorana_form()
