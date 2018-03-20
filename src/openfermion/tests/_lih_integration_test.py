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

"""Tests many modules to compute energy of LiH."""
from __future__ import absolute_import

import os
import numpy
import scipy.sparse
import unittest

from openfermion.config import *
from openfermion.hamiltonians import *
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *


class LiHIntegrationTest(unittest.TestCase):

    def setUp(self):
        # Set up molecule.
        geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H1-Li1_sto-3g_singlet_1.45')
        self.molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        self.molecule.load()

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()
        self.molecular_hamiltonian_no_core = (
            self.molecule.
            get_molecular_hamiltonian(occupied_indices=[0],
                                      active_indices=range(1,
                                                           self.molecule.
                                                           n_orbitals)))

        # Get FCI RDM.
        self.fci_rdm = self.molecule.get_molecular_rdm(use_fci=1)

        # Get explicit coefficients.
        self.nuclear_repulsion = self.molecular_hamiltonian.constant
        self.one_body = self.molecular_hamiltonian.one_body_tensor
        self.two_body = self.molecular_hamiltonian.two_body_tensor

        # Get fermion Hamiltonian.
        self.fermion_hamiltonian = normal_ordered(get_fermion_operator(
            self.molecular_hamiltonian))

        # Get qubit Hamiltonian.
        self.qubit_hamiltonian = jordan_wigner(self.fermion_hamiltonian)

        # Get explicit coefficients.
        self.nuclear_repulsion = self.molecular_hamiltonian.constant
        self.one_body = self.molecular_hamiltonian.one_body_tensor
        self.two_body = self.molecular_hamiltonian.two_body_tensor

        # Get matrix form.
        self.hamiltonian_matrix = get_sparse_operator(
            self.molecular_hamiltonian)
        self.hamiltonian_matrix_no_core = get_sparse_operator(
            self.molecular_hamiltonian_no_core)

    def test_all(self):

        # Test reverse Jordan-Wigner.
        fermion_hamiltonian = reverse_jordan_wigner(self.qubit_hamiltonian)
        fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
        self.assertTrue(self.fermion_hamiltonian.isclose(fermion_hamiltonian))

        # Test mapping to interaction operator.
        fermion_hamiltonian = get_fermion_operator(self.molecular_hamiltonian)
        fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
        self.assertTrue(self.fermion_hamiltonian.isclose(fermion_hamiltonian))

        # Test RDM energy.
        fci_rdm_energy = self.nuclear_repulsion
        fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_tensor *
                                    self.one_body)
        fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_tensor *
                                    self.two_body)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Confirm expectation on qubit Hamiltonian using reverse JW matches.
        qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
        qubit_energy = 0.0
        for term, coefficient in qubit_rdm.terms.items():
            qubit_energy += coefficient * self.qubit_hamiltonian.terms[term]
        self.assertAlmostEqual(qubit_energy, self.molecule.fci_energy)

        # Confirm fermionic RDMs can be built from measured qubit RDMs.
        new_fermi_rdm = get_interaction_rdm(qubit_rdm)
        fermi_rdm_energy = new_fermi_rdm.expectation(
            self.molecular_hamiltonian)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Test sparse matrices.
        energy, wavefunction = get_ground_state(self.hamiltonian_matrix)
        self.assertAlmostEqual(energy, self.molecule.fci_energy)
        expected_energy = expectation(self.hamiltonian_matrix, wavefunction)
        self.assertAlmostEqual(expected_energy, energy)

        # Make sure you can reproduce Hartree-Fock energy.
        hf_state = jw_hartree_fock_state(
            self.molecule.n_electrons, count_qubits(self.qubit_hamiltonian))
        hf_density = get_density_matrix([hf_state], [1.])
        expected_hf_density_energy = expectation(self.hamiltonian_matrix,
                                                 hf_density)
        expected_hf_energy = expectation(self.hamiltonian_matrix, hf_state)
        self.assertAlmostEqual(expected_hf_energy, self.molecule.hf_energy)
        self.assertAlmostEqual(expected_hf_density_energy,
                               self.molecule.hf_energy)

        # Check that frozen core result matches frozen core FCI from psi4.
        # Recore frozen core result from external calculation.
        self.frozen_core_fci_energy = -7.8807607374168
        no_core_fci_energy = scipy.linalg.eigh(
            self.hamiltonian_matrix_no_core.todense())[0][0]
        self.assertAlmostEqual(no_core_fci_energy,
                               self.frozen_core_fci_energy)

        # Check that the freeze_orbitals function has the same effect as the
        # as the occupied_indices option of get_molecular_hamiltonian.
        frozen_hamiltonian = freeze_orbitals(
                get_fermion_operator(self.molecular_hamiltonian), [0, 1])
        self.assertTrue(frozen_hamiltonian.isclose(
                get_fermion_operator(self.molecular_hamiltonian_no_core)))
