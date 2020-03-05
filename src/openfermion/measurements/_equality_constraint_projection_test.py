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

"""Tests for _variance_reduction.py"""
import os
import unittest
import numpy

from openfermion.hamiltonians import MolecularData
from openfermion.config import THIS_DIRECTORY
from openfermion.transforms import get_fermion_operator, get_sparse_operator
from openfermion.utils import (count_qubits,
                               expectation,
                               get_ground_state,
                               jw_number_restrict_operator,
                               sparse_eigenspectrum)
from ._equality_constraint_projection import (apply_constraints,
                                              constraint_matrix, linearize_term,
                                              operator_to_vector,
                                              unlinearize_term,
                                              vector_to_operator)


class EqualityConstraintProjectionTest(unittest.TestCase):

    def setUp(self):

        # Set up molecule.
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        molecule.load()
        self.n_fermions = molecule.n_electrons
        self.n_orbitals = molecule.n_qubits

        # Get molecular Hamiltonian.
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        self.fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

    def test_linearize_term(self):
        past_terms = set()
        for term, _ in self.fermion_hamiltonian.terms.items():
            index = linearize_term(term, self.n_orbitals)
            self.assertTrue(isinstance(index, int))
            self.assertFalse(index in past_terms)
            past_terms.add(index)

    def test_unlinearize_term_consistency(self):
        for term, _ in self.fermion_hamiltonian.terms.items():
            index = linearize_term(term, self.n_orbitals)
            new_term = unlinearize_term(index, self.n_orbitals)
            self.assertEqual(term, new_term)

    def test_operator_to_vector_consistency(self):
        vector = operator_to_vector(self.fermion_hamiltonian)
        operator = vector_to_operator(vector, self.n_orbitals)
        magnitude = 0.
        difference = operator - self.fermion_hamiltonian
        for _, coefficient in difference.terms.items():
            magnitude += abs(coefficient)
        self.assertAlmostEqual(0, magnitude)

    def test_constraint_matrix(self):

        # Randomly project operator with constraints.
        numpy.random.seed(8)
        constraints = constraint_matrix(self.n_orbitals, self.n_fermions)
        n_constraints, n_terms = constraints.shape
        self.assertEqual(
            1 + self.n_orbitals ** 2 + self.n_orbitals ** 4, n_terms)
        random_weights = numpy.random.randn(n_constraints)
        vectorized_operator = operator_to_vector(self.fermion_hamiltonian)
        modification_vector = constraints.transpose() * random_weights
        new_operator_vector = vectorized_operator + modification_vector
        modified_operator = vector_to_operator(
            new_operator_vector, self.n_orbitals)

        # Map both to sparse matrix under Jordan-Wigner.
        sparse_original = get_sparse_operator(self.fermion_hamiltonian)
        sparse_modified = get_sparse_operator(modified_operator)

        # Check expectation value.
        energy, wavefunction = get_ground_state(sparse_original)
        modified_energy = expectation(sparse_modified, wavefunction)
        self.assertAlmostEqual(modified_energy, energy)

    def test_apply_constraints(self):

        # Get norm of original operator.
        original_norm = 0.
        for term, coefficient in self.fermion_hamiltonian.terms.items():
            if term != ():
                original_norm += abs(coefficient)

        # Get modified operator.
        modified_operator = apply_constraints(
            self.fermion_hamiltonian, self.n_fermions)
        modified_operator.compress()

        # Get norm of modified operator.
        modified_norm = 0.
        for term, coefficient in modified_operator.terms.items():
            if term != ():
                modified_norm += abs(coefficient)
        self.assertTrue(modified_norm < original_norm)

        # Map both to sparse matrix under Jordan-Wigner.
        sparse_original = get_sparse_operator(self.fermion_hamiltonian)
        sparse_modified = get_sparse_operator(modified_operator)

        # Check spectra.
        sparse_original = jw_number_restrict_operator(sparse_original,
                                                      self.n_fermions)
        sparse_modified = jw_number_restrict_operator(sparse_modified,
                                                      self.n_fermions)
        original_spectrum = sparse_eigenspectrum(sparse_original)
        modified_spectrum = sparse_eigenspectrum(sparse_modified)
        spectral_deviation = numpy.amax(numpy.absolute(
            original_spectrum - modified_spectrum))
        self.assertAlmostEqual(spectral_deviation, 0.)

        # Check expectation value.
        energy, wavefunction = get_ground_state(sparse_original)
        modified_energy = expectation(sparse_modified, wavefunction)
        self.assertAlmostEqual(modified_energy, energy)
