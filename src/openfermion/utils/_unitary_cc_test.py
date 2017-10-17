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

"""Module to test unitary coupled cluster operators."""

from __future__ import absolute_import

import numpy
import os
import scipy
import unittest

from numpy.random import randn

from openfermion.config import THIS_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *


class UnitaryCC(unittest.TestCase):

    def test_uccsd_anti_hermitian(self):
        """Test operators are anti-Hermitian independent of inputs"""
        test_orbitals = 4

        single_amplitudes = randn(*(test_orbitals,) * 2)
        double_amplitudes = randn(*(test_orbitals,) * 4)

        generator = uccsd_operator(single_amplitudes, double_amplitudes)
        conj_generator = hermitian_conjugated(generator)

        self.assertTrue(generator.isclose(-1. * conj_generator))

    def test_uccsd_singlet_anti_hermitian(self):
        """Test that the singlet version is anti-Hermitian"""
        test_orbitals = 8
        test_electrons = 4

        packed_amplitude_size = uccsd_singlet_paramsize(test_orbitals,
                                                        test_electrons)

        packed_amplitudes = randn(int(packed_amplitude_size))

        generator = uccsd_singlet_operator(packed_amplitudes,
                                           test_orbitals,
                                           test_electrons)

        conj_generator = hermitian_conjugated(generator)

        self.assertTrue(generator.isclose(-1. * conj_generator))

    def test_uccsd_singlet_build(self):
        """Test a specific build of the UCCSD singlet operator"""
        initial_amplitudes = [-1.14941450e-08, 5.65340614e-02]
        n_orbitals = 4
        n_electrons = 2

        generator = uccsd_singlet_operator(initial_amplitudes,
                                           n_orbitals,
                                           n_electrons)

        test_generator = (0.0565340614 * FermionOperator("2^ 0 3^ 1") +
                          1.1494145e-08 * FermionOperator("1^ 3") +
                          0.0565340614 * FermionOperator("3^ 1 2^ 0") +
                          0.0565340614 * FermionOperator("2^ 0 2^ 0") +
                          1.1494145e-08 * FermionOperator("0^ 2") +
                          (-0.0565340614) * FermionOperator("1^ 3 0^ 2") +
                          (-1.1494145e-08) * FermionOperator("3^ 1") +
                          (-0.0565340614) * FermionOperator("1^ 3 1^ 3") +
                          (-0.0565340614) * FermionOperator("0^ 2 0^ 2") +
                          (-1.1494145e-08) * FermionOperator("2^ 0") +
                          0.0565340614 * FermionOperator("3^ 1 3^ 1") +
                          (-0.0565340614) * FermionOperator("0^ 2 1^ 3"))
        self.assertTrue(test_generator.isclose(generator))

    def test_sparse_uccsd_operator_numpy_inputs(self):
        """Test numpy ndarray inputs to uccsd_operator that are sparse"""
        test_orbitals = 30
        sparse_single_amplitudes = numpy.zeros((test_orbitals, test_orbitals))
        sparse_double_amplitudes = numpy.zeros((test_orbitals, test_orbitals,
                                                test_orbitals, test_orbitals))

        sparse_single_amplitudes[3, 5] = 0.12345
        sparse_single_amplitudes[12, 4] = 0.44313

        sparse_double_amplitudes[0, 12, 6, 2] = 0.3434
        sparse_double_amplitudes[1, 4, 6, 13] = -0.23423

        generator = uccsd_operator(sparse_single_amplitudes,
                                   sparse_double_amplitudes)

        test_generator = (0.12345 * FermionOperator("3^ 5") +
                          (-0.12345) * FermionOperator("5^ 3") +
                          0.44313 * FermionOperator("12^ 4") +
                          (-0.44313) * FermionOperator("4^ 12") +
                          0.3434 * FermionOperator("0^ 12 6^ 2") +
                          (-0.3434) * FermionOperator("2^ 6 12^ 0") +
                          (-0.23423) * FermionOperator("1^ 4 6^ 13") +
                          0.23423 * FermionOperator("13^ 6 4^ 1"))
        self.assertTrue(test_generator.isclose(generator))

    def test_sparse_uccsd_operator_list_inputs(self):
        """Test list inputs to uccsd_operator that are sparse"""
        sparse_single_amplitudes = [[[3, 5], 0.12345],
                                    [[12, 4], 0.44313]]
        sparse_double_amplitudes = [[[0, 12, 6, 2], 0.3434],
                                    [[1, 4, 6, 13], -0.23423]]

        generator = uccsd_operator(sparse_single_amplitudes,
                                   sparse_double_amplitudes)

        test_generator = (0.12345 * FermionOperator("3^ 5") +
                          (-0.12345) * FermionOperator("5^ 3") +
                          0.44313 * FermionOperator("12^ 4") +
                          (-0.44313) * FermionOperator("4^ 12") +
                          0.3434 * FermionOperator("0^ 12 6^ 2") +
                          (-0.3434) * FermionOperator("2^ 6 12^ 0") +
                          (-0.23423) * FermionOperator("1^ 4 6^ 13") +
                          0.23423 * FermionOperator("13^ 6 4^ 1"))
        self.assertTrue(test_generator.isclose(generator))

    def test_ucc(self):
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
        # Test UCCSD for accuracy against FCI using loaded t amplitudes.
        ucc_operator = uccsd_operator(
            self.molecule.ccsd_single_amps,
            self.molecule.ccsd_double_amps)

        hf_state = jw_hartree_fock_state(
            self.molecule.n_electrons, count_qubits(self.qubit_hamiltonian))
        uccsd_sparse = jordan_wigner_sparse(ucc_operator)
        uccsd_state = scipy.sparse.linalg.expm_multiply(uccsd_sparse,
                                                        hf_state)
        expected_uccsd_energy = expectation(self.hamiltonian_matrix,
                                            uccsd_state)
        self.assertAlmostEqual(expected_uccsd_energy, self.molecule.fci_energy,
                               places=4)
        print("UCCSD ENERGY: {}".format(expected_uccsd_energy))

        # Test CCSD for precise match against FCI using loaded t amplitudes.
        ccsd_operator = uccsd_operator(
            self.molecule.ccsd_single_amps,
            self.molecule.ccsd_double_amps,
            anti_hermitian=False)

        ccsd_sparse_r = jordan_wigner_sparse(ccsd_operator)
        ccsd_sparse_l = jordan_wigner_sparse(
            -hermitian_conjugated(ccsd_operator))

        # Test CCSD for precise match against FCI using loaded t amplitudes
        ccsd_operator = uccsd_operator(
            self.molecule.ccsd_single_amps,
            self.molecule.ccsd_double_amps,
            anti_hermitian=False)

        ccsd_sparse_r = jordan_wigner_sparse(ccsd_operator)
        ccsd_sparse_l = jordan_wigner_sparse(
            -hermitian_conjugated(ccsd_operator))
        ccsd_state_r = scipy.sparse.linalg.expm_multiply(ccsd_sparse_r,
                                                         hf_state)
        ccsd_state_l = scipy.sparse.linalg.expm_multiply(ccsd_sparse_l,
                                                         hf_state)
        expected_ccsd_energy = ccsd_state_l.getH().dot(
            self.hamiltonian_matrix.dot(ccsd_state_r))[0, 0]
        self.assertAlmostEqual(expected_ccsd_energy, self.molecule.fci_energy)
