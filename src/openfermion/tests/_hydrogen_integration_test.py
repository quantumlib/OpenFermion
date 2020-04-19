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

"""Tests many modules to compute energy of hydrogen."""

import os
import unittest
import numpy
import scipy.sparse

from openfermion.config import *
from openfermion.hamiltonians import *
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *
from openfermion._version import __version__


class HydrogenIntegrationTest(unittest.TestCase):

    def setUp(self):
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        self.molecule.load()

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

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

        # Get the sparse matrix.
        self.hamiltonian_matrix = get_sparse_operator(
            self.molecular_hamiltonian)

    def test_integral_data(self):

        # Initialize coefficients given in arXiv 1208.5986:
        g0 = 0.71375
        g1 = -1.2525
        g2 = -0.47593
        g3 = 0.67449 / 2.
        g4 = 0.69740 / 2.
        g5 = 0.66347 / 2.
        g6 = 0.18129 / 2.

        # Check the integrals in the test data.
        self.assertAlmostEqual(self.nuclear_repulsion, g0, places=4)

        self.assertAlmostEqual(self.one_body[0, 0], g1, places=4)
        self.assertAlmostEqual(self.one_body[1, 1], g1, places=4)

        self.assertAlmostEqual(self.one_body[2, 2], g2, places=4)
        self.assertAlmostEqual(self.one_body[3, 3], g2, places=4)

        self.assertAlmostEqual(self.two_body[0, 1, 1, 0], g3, places=4)
        self.assertAlmostEqual(self.two_body[1, 0, 0, 1], g3, places=4)

        self.assertAlmostEqual(self.two_body[2, 3, 3, 2], g4, places=4)
        self.assertAlmostEqual(self.two_body[3, 2, 2, 3], g4, places=4)

        self.assertAlmostEqual(self.two_body[0, 2, 2, 0], g5, places=4)
        self.assertAlmostEqual(self.two_body[0, 3, 3, 0], g5, places=4)
        self.assertAlmostEqual(self.two_body[1, 2, 2, 1], g5, places=4)
        self.assertAlmostEqual(self.two_body[1, 3, 3, 1], g5, places=4)
        self.assertAlmostEqual(self.two_body[2, 0, 0, 2], g5, places=4)
        self.assertAlmostEqual(self.two_body[3, 0, 0, 3], g5, places=4)
        self.assertAlmostEqual(self.two_body[2, 1, 1, 2], g5, places=4)
        self.assertAlmostEqual(self.two_body[3, 1, 1, 3], g5, places=4)

        self.assertAlmostEqual(self.two_body[0, 2, 0, 2], g6, places=4)
        self.assertAlmostEqual(self.two_body[1, 3, 1, 3], g6, places=4)
        self.assertAlmostEqual(self.two_body[2, 1, 3, 0], g6, places=4)
        self.assertAlmostEqual(self.two_body[2, 3, 1, 0], g6, places=4)
        self.assertAlmostEqual(self.two_body[0, 3, 1, 2], g6, places=4)
        self.assertAlmostEqual(self.two_body[0, 1, 3, 2], g6, places=4)

    def test_qubit_operator(self):

        # Below are qubit term coefficients, also from arXiv 1208.5986:
        f1 = 0.1712
        f2 = -0.2228
        f3 = 0.1686
        f4 = 0.1205
        f5 = 0.1659
        f6 = 0.1743
        f7 = 0.04532

        # Test the local Hamiltonian terms.
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'Z'),)], f1, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((1, 'Z'),)], f1, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((2, 'Z'),)], f2, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((3, 'Z'),)], f2, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'Z'), (1, 'Z'))],
            f3, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'Z'), (2, 'Z'))],
            f4, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((1, 'Z'), (3, 'Z'))],
            f4, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((1, 'Z'), (2, 'Z'))],
            f5, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'Z'), (3, 'Z'))],
            f5, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((2, 'Z'), (3, 'Z'))],
            f6, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'Y'), (1, 'Y'),
                                          (2, 'X'), (3, 'X'))],
            -f7, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'X'), (1, 'X'),
                                          (2, 'Y'), (3, 'Y'))],
            -f7, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'X'), (1, 'Y'),
                                          (2, 'Y'), (3, 'X'))],
            f7, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian.terms[((0, 'Y'), (1, 'X'),
                                          (2, 'X'), (3, 'Y'))],
            f7, places=4)

    def test_reverse_jordan_wigner(self):
        fermion_hamiltonian = reverse_jordan_wigner(self.qubit_hamiltonian)
        fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
        self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

    def test_interaction_operator_mapping(self):

        # Make sure mapping of FermionOperator to InteractionOperator works.
        molecular_hamiltonian = get_interaction_operator(
            self.fermion_hamiltonian)
        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

        # Make sure mapping of InteractionOperator to QubitOperator works.
        qubit_hamiltonian = jordan_wigner(self.molecular_hamiltonian)
        self.assertTrue(self.qubit_hamiltonian == qubit_hamiltonian)

    def test_rdm_numerically(self):

        # Test energy of RDM.
        fci_rdm_energy = self.nuclear_repulsion
        fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_tensor *
                                    self.molecular_hamiltonian.one_body_tensor)
        fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_tensor *
                                    self.molecular_hamiltonian.two_body_tensor)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Confirm expectation on qubit Hamiltonian using reverse JW matches.
        qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
        qubit_energy = 0.0
        for term, expectation in qubit_rdm.terms.items():
            qubit_energy += expectation * self.qubit_hamiltonian.terms[term]
        self.assertAlmostEqual(qubit_energy, self.molecule.fci_energy)

        # Confirm fermionic RDMs can be built from measured qubit RDMs
        new_fermi_rdm = get_interaction_rdm(qubit_rdm)
        new_fermi_rdm.expectation(self.molecular_hamiltonian)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

    def test_sparse_numerically(self):

        # Check FCI energy.
        energy, wavefunction = get_ground_state(self.hamiltonian_matrix)
        self.assertAlmostEqual(energy, self.molecule.fci_energy)
        expected_energy = expectation(self.hamiltonian_matrix, wavefunction)
        self.assertAlmostEqual(expected_energy, energy)

        # Make sure you can reproduce Hartree energy.
        hf_state = jw_hartree_fock_state(
            self.molecule.n_electrons, count_qubits(self.qubit_hamiltonian))
        hf_density = get_density_matrix([hf_state], [1.])

        # Make sure you can reproduce Hartree-Fock energy.
        hf_state = jw_hartree_fock_state(
            self.molecule.n_electrons, count_qubits(self.qubit_hamiltonian))
        hf_density = get_density_matrix([hf_state], [1.])
        expected_hf_density_energy = expectation(self.hamiltonian_matrix,
                                                 hf_density)
        expected_hf_energy = expectation(self.hamiltonian_matrix,
                                         hf_state)
        self.assertAlmostEqual(expected_hf_energy, self.molecule.hf_energy)
        self.assertAlmostEqual(expected_hf_density_energy,
                               self.molecule.hf_energy)
