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


import os
import unittest
import numpy
from numpy.random import randn
import scipy

from openfermion.config import THIS_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import (get_fermion_operator,
                                    get_sparse_operator,
                                    jordan_wigner)

from openfermion.utils import (commutator, count_qubits, expectation,
                               hermitian_conjugated, normal_ordered,
                               jordan_wigner_sparse, jw_hartree_fock_state,
                               s_squared_operator, sz_operator)
from openfermion.utils._unitary_cc import (uccsd_generator,
                                           uccsd_singlet_generator,
                                           uccsd_singlet_get_packed_amplitudes,
                                           uccsd_singlet_paramsize)


class UnitaryCC(unittest.TestCase):

    def test_uccsd_anti_hermitian(self):
        """Test operators are anti-Hermitian independent of inputs"""
        test_orbitals = 4

        single_amplitudes = randn(*(test_orbitals,) * 2)
        double_amplitudes = randn(*(test_orbitals,) * 4)

        generator = uccsd_generator(single_amplitudes, double_amplitudes)
        conj_generator = hermitian_conjugated(generator)

        self.assertEqual(generator, -1. * conj_generator)

    def test_uccsd_singlet_anti_hermitian(self):
        """Test that the singlet version is anti-Hermitian"""
        test_orbitals = 8
        test_electrons = 4

        packed_amplitude_size = uccsd_singlet_paramsize(test_orbitals,
                                                        test_electrons)

        packed_amplitudes = randn(int(packed_amplitude_size))

        generator = uccsd_singlet_generator(packed_amplitudes,
                                            test_orbitals,
                                            test_electrons)

        conj_generator = hermitian_conjugated(generator)

        self.assertEqual(generator, -1. * conj_generator)

    def test_uccsd_singlet_symmetries(self):
        """Test that the singlet generator has the correct symmetries."""
        test_orbitals = 8
        test_electrons = 4

        packed_amplitude_size = uccsd_singlet_paramsize(test_orbitals,
                                                        test_electrons)
        packed_amplitudes = randn(int(packed_amplitude_size))
        generator = uccsd_singlet_generator(packed_amplitudes,
                                            test_orbitals,
                                            test_electrons)

        # Construct symmetry operators
        sz = sz_operator(test_orbitals)
        s_squared = s_squared_operator(test_orbitals)

        # Check the symmetries
        comm_sz = normal_ordered(commutator(generator, sz))
        comm_s_squared = normal_ordered(commutator(generator, s_squared))
        zero = FermionOperator()

        self.assertEqual(comm_sz, zero)
        self.assertEqual(comm_s_squared, zero)

    def test_uccsd_singlet_builds(self):
        """Test specific builds of the UCCSD singlet operator"""
        # Build 1
        n_orbitals = 4
        n_electrons = 2
        n_params = uccsd_singlet_paramsize(n_orbitals, n_electrons)
        self.assertEqual(n_params, 2)

        initial_amplitudes = [1., 2.]

        generator = uccsd_singlet_generator(initial_amplitudes,
                                            n_orbitals,
                                            n_electrons)

        test_generator = (FermionOperator("2^ 0", 1.) +
                          FermionOperator("0^ 2", -1.) +
                          FermionOperator("3^ 1", 1.) +
                          FermionOperator("1^ 3", -1.) +
                          FermionOperator("2^ 0 3^ 1", 4.) +
                          FermionOperator("1^ 3 0^ 2", -4.))

        self.assertEqual(normal_ordered(test_generator),
                         normal_ordered(generator))

        # Build 2
        n_orbitals = 6
        n_electrons = 2

        n_params = uccsd_singlet_paramsize(n_orbitals, n_electrons)
        self.assertEqual(n_params, 5)

        initial_amplitudes = numpy.arange(1, n_params + 1, dtype=float)
        generator = uccsd_singlet_generator(initial_amplitudes,
                                            n_orbitals,
                                            n_electrons)

        test_generator = (FermionOperator("2^ 0", 1.) +
                          FermionOperator("0^ 2", -1) +
                          FermionOperator("3^ 1", 1.) +
                          FermionOperator("1^ 3", -1.) +
                          FermionOperator("4^ 0", 2.) +
                          FermionOperator("0^ 4", -2) +
                          FermionOperator("5^ 1", 2.) +
                          FermionOperator("1^ 5", -2.) +
                          FermionOperator("2^ 0 3^ 1", 6.) +
                          FermionOperator("1^ 3 0^ 2", -6.) +
                          FermionOperator("4^ 0 5^ 1", 8.) +
                          FermionOperator("1^ 5 0^ 4", -8.) +
                          FermionOperator("2^ 0 5^ 1", 5.) +
                          FermionOperator("1^ 5 0^ 2", -5.) +
                          FermionOperator("4^ 0 3^ 1", 5.) +
                          FermionOperator("1^ 3 0^ 4", -5.) +
                          FermionOperator("2^ 0 4^ 0", 5.) +
                          FermionOperator("0^ 4 0^ 2", -5.) +
                          FermionOperator("3^ 1 5^ 1", 5.) +
                          FermionOperator("1^ 5 1^ 3", -5.))

        self.assertEqual(normal_ordered(test_generator),
                         normal_ordered(generator))

    def test_sparse_uccsd_generator_numpy_inputs(self):
        """Test numpy ndarray inputs to uccsd_generator that are sparse"""
        test_orbitals = 30
        sparse_single_amplitudes = numpy.zeros((test_orbitals, test_orbitals))
        sparse_double_amplitudes = numpy.zeros((test_orbitals, test_orbitals,
                                                test_orbitals, test_orbitals))

        sparse_single_amplitudes[3, 5] = 0.12345
        sparse_single_amplitudes[12, 4] = 0.44313

        sparse_double_amplitudes[0, 12, 6, 2] = 0.3434
        sparse_double_amplitudes[1, 4, 6, 13] = -0.23423

        generator = uccsd_generator(sparse_single_amplitudes,
                                    sparse_double_amplitudes)

        test_generator = (0.12345 * FermionOperator("3^ 5") +
                          (-0.12345) * FermionOperator("5^ 3") +
                          0.44313 * FermionOperator("12^ 4") +
                          (-0.44313) * FermionOperator("4^ 12") +
                          0.3434 * FermionOperator("0^ 12 6^ 2") +
                          (-0.3434) * FermionOperator("2^ 6 12^ 0") +
                          (-0.23423) * FermionOperator("1^ 4 6^ 13") +
                          0.23423 * FermionOperator("13^ 6 4^ 1"))
        self.assertEqual(test_generator, generator)

    def test_sparse_uccsd_generator_list_inputs(self):
        """Test list inputs to uccsd_generator that are sparse"""
        sparse_single_amplitudes = [[[3, 5], 0.12345],
                                    [[12, 4], 0.44313]]
        sparse_double_amplitudes = [[[0, 12, 6, 2], 0.3434],
                                    [[1, 4, 6, 13], -0.23423]]

        generator = uccsd_generator(sparse_single_amplitudes,
                                    sparse_double_amplitudes)

        test_generator = (0.12345 * FermionOperator("3^ 5") +
                          (-0.12345) * FermionOperator("5^ 3") +
                          0.44313 * FermionOperator("12^ 4") +
                          (-0.44313) * FermionOperator("4^ 12") +
                          0.3434 * FermionOperator("0^ 12 6^ 2") +
                          (-0.3434) * FermionOperator("2^ 6 12^ 0") +
                          (-0.23423) * FermionOperator("1^ 4 6^ 13") +
                          0.23423 * FermionOperator("13^ 6 4^ 1"))
        self.assertEqual(test_generator, generator)

    def test_uccsd_singlet_get_packed_amplitudes(self):
        test_orbitals = 6
        test_electrons = 2
        sparse_single_amplitudes = numpy.zeros((test_orbitals, test_orbitals))
        sparse_double_amplitudes = numpy.zeros((test_orbitals, test_orbitals,
                                                test_orbitals, test_orbitals))

        sparse_single_amplitudes[2, 0] = 0.12345
        sparse_single_amplitudes[3, 1] = 0.12345

        sparse_double_amplitudes[2, 0, 3, 1] = 0.9

        sparse_double_amplitudes[2, 0, 4, 0] = 0.3434
        sparse_double_amplitudes[5, 1, 3, 1] = 0.3434

        packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
            sparse_single_amplitudes, sparse_double_amplitudes,
            test_orbitals, test_electrons)

        self.assertEqual(len(packed_amplitudes), 5)

        self.assertTrue(numpy.allclose(
            packed_amplitudes,
            numpy.array([0.12345, 0., 0.9, 0., 0.3434])))

    def test_ucc_h2(self):
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
        ucc_operator = uccsd_generator(
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
        ccsd_operator = uccsd_generator(
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
        expected_ccsd_energy = ccsd_state_l.conjugate().dot(
            self.hamiltonian_matrix.dot(ccsd_state_r))
        self.assertAlmostEqual(expected_ccsd_energy, self.molecule.fci_energy)

    def test_ucc_h2_singlet(self):
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
        ucc_operator = uccsd_generator(
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

        # Test CCSD singlet for precise match against FCI using loaded t
        # amplitudes
        packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
            self.molecule.ccsd_single_amps,
            self.molecule.ccsd_double_amps,
            self.molecule.n_qubits,
            self.molecule.n_electrons)
        ccsd_operator = uccsd_singlet_generator(
            packed_amplitudes,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            anti_hermitian=False)

        ccsd_sparse_r = jordan_wigner_sparse(ccsd_operator)
        ccsd_sparse_l = jordan_wigner_sparse(
            -hermitian_conjugated(ccsd_operator))

        ccsd_state_r = scipy.sparse.linalg.expm_multiply(ccsd_sparse_r,
                                                         hf_state)
        ccsd_state_l = scipy.sparse.linalg.expm_multiply(ccsd_sparse_l,
                                                         hf_state)
        expected_ccsd_energy = ccsd_state_l.conjugate().dot(
            self.hamiltonian_matrix.dot(ccsd_state_r))
        self.assertAlmostEqual(expected_ccsd_energy, self.molecule.fci_energy)

    def test_value_error_for_odd_n_qubits(self):
        # Pass odd n_qubits to singlet generators
        with self.assertRaises(ValueError):
            _ = uccsd_singlet_paramsize(3, 4)

    def test_value_error_bad_amplitudes(self):
        with self.assertRaises(ValueError):
            _ = uccsd_singlet_generator([1.], 3, 4)
