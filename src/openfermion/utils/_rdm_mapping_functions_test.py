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

"""Tests for _rdm_mapping_functions.py"""
import os
import unittest

import numpy
import h5py
from openfermion.config import DATA_DIRECTORY, THIS_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.utils._rdm_mapping_functions import (
    kronecker_delta, map_two_pdm_to_two_hole_dm, map_two_pdm_to_one_pdm,
    map_one_pdm_to_one_hole_dm, map_one_hole_dm_to_one_pdm,
    map_two_pdm_to_particle_hole_dm,
    map_two_hole_dm_to_two_pdm, map_two_hole_dm_to_one_hole_dm,
    map_particle_hole_dm_to_one_pdm,
    map_particle_hole_dm_to_two_pdm)


class RDMMappingTest(unittest.TestCase):
    def setUp(self):
        # load files and marginals from tests folder
        tqdm_h2_sto3g = os.path.join(THIS_DIRECTORY,
                                     'tests/tqdm_H2_sto-3g_singlet_1.4.hdf5')
        with h5py.File(tqdm_h2_sto3g, 'r') as fid:
            self.tqdm_h2_sto3g = fid['tqdm'][...]

        phdm_h2_sto3g = os.path.join(
            THIS_DIRECTORY, 'tests/phdm_H2_sto-3g_singlet_1.4.hdf5')
        with h5py.File(phdm_h2_sto3g, 'r') as fid:
            self.phdm_h2_sto3g = fid['phdm'][...]

        tqdm_h2_6_31g = os.path.join(THIS_DIRECTORY,
                                     'tests/tqdm_H2_6-31g_singlet_0.75.hdf5')
        with h5py.File(tqdm_h2_6_31g, 'r') as fid:
            self.tqdm_h2_6_31g = fid['tqdm'][...]

        phdm_h2_6_31g = os.path.join(THIS_DIRECTORY,
                                     'tests/phdm_H2_6-31g_singlet_0.75.hdf5')
        with h5py.File(phdm_h2_6_31g, 'r') as fid:
            self.phdm_h2_6_31g = fid['phdm'][...]

        tqdm_lih_sto3g = os.path.join(
            THIS_DIRECTORY, 'tests/tqdm_H1-Li1_sto-3g_singlet_1.45.hdf5')
        with h5py.File(tqdm_lih_sto3g, 'r') as fid:
            self.tqdm_lih_sto3g = fid['tqdm'][...]

        phdm_lih_sto3g = os.path.join(
            THIS_DIRECTORY, 'tests/phdm_H1-Li1_sto-3g_singlet_1.45.hdf5')
        with h5py.File(phdm_lih_sto3g, 'r') as fid:
            self.phdm_lih_sto3g = fid['phdm'][...]

    def test_kronecker_delta_00(self):
        assert kronecker_delta(0, 0) == 1

    def test_kronecker_delta_01(self):
        assert kronecker_delta(0, 1) == 0

    def test_kronecker_delta_10(self):
        assert kronecker_delta(1, 0) == 0

    def test_kronecker_delta_11(self):
        assert kronecker_delta(1, 1) == 1

    def test_kronecker_delta_nonunit_args(self):
        assert kronecker_delta(3, 3) == 1

    def test_tpdm_to_opdm(self):
        # for all files in datadirectory check if this map holds
        for file in filter(lambda x: x.endswith(".hdf5"),
                           os.listdir(DATA_DIRECTORY)):
            molecule = MolecularData(
                filename=os.path.join(DATA_DIRECTORY, file))
            if (molecule.fci_one_rdm is not None and
                    molecule.fci_two_rdm is not None):
                test_opdm = map_two_pdm_to_one_pdm(molecule.fci_two_rdm,
                                                   molecule.n_electrons)
                assert numpy.allclose(test_opdm, molecule.fci_one_rdm)

    def test_opdm_to_oqdm(self):
        for file in filter(lambda x: x.endswith(".hdf5"),
                           os.listdir(DATA_DIRECTORY)):
            molecule = MolecularData(
                filename=os.path.join(DATA_DIRECTORY, file))
            if molecule.fci_one_rdm is not None:
                test_oqdm = map_one_pdm_to_one_hole_dm(molecule.fci_one_rdm)
                true_oqdm = numpy.eye(molecule.n_qubits) - molecule.fci_one_rdm
                assert numpy.allclose(test_oqdm, true_oqdm)

    def test_oqdm_to_opdm(self):
        for file in filter(lambda x: x.endswith(".hdf5"),
                           os.listdir(DATA_DIRECTORY)):
            molecule = MolecularData(
                filename=os.path.join(DATA_DIRECTORY, file))
            if molecule.fci_one_rdm is not None:
                true_oqdm = numpy.eye(molecule.n_qubits) - molecule.fci_one_rdm
                test_opdm = map_one_hole_dm_to_one_pdm(true_oqdm)
                assert numpy.allclose(test_opdm, molecule.fci_one_rdm)

    def test_tqdm_conversions_h2_631g(self):
        # construct the 2-hole-RDM for LiH the slow way
        # TODO: speed up this calculation by directly contracting from the wf.
        filename = "H2_6-31g_singlet_0.75.hdf5"
        molecule = MolecularData(
            filename=os.path.join(DATA_DIRECTORY, filename))
        true_tqdm = self.tqdm_h2_6_31g

        test_tqdm = map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm,
                                               molecule.fci_one_rdm)
        assert numpy.allclose(true_tqdm, test_tqdm)

        true_oqdm = numpy.eye(molecule.n_qubits) - molecule.fci_one_rdm
        test_oqdm = map_two_hole_dm_to_one_hole_dm(
            true_tqdm, molecule.n_qubits - molecule.n_electrons)
        assert numpy.allclose(true_oqdm, test_oqdm)

        test_tpdm = map_two_hole_dm_to_two_pdm(true_tqdm, molecule.fci_one_rdm)
        assert numpy.allclose(test_tpdm, molecule.fci_two_rdm)

    def test_tqdm_conversions_h2_sto3g(self):
        filename = "H2_sto-3g_singlet_1.4.hdf5"
        molecule = MolecularData(
            filename=os.path.join(DATA_DIRECTORY, filename))
        true_tqdm = self.tqdm_h2_sto3g

        test_tqdm = map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm,
                                               molecule.fci_one_rdm)
        assert numpy.allclose(true_tqdm, test_tqdm)

        true_oqdm = numpy.eye(molecule.n_qubits) - molecule.fci_one_rdm
        test_oqdm = map_two_hole_dm_to_one_hole_dm(
            true_tqdm, molecule.n_qubits - molecule.n_electrons)
        assert numpy.allclose(true_oqdm, test_oqdm)

        test_tpdm = map_two_hole_dm_to_two_pdm(true_tqdm, molecule.fci_one_rdm)
        assert numpy.allclose(test_tpdm, molecule.fci_two_rdm)

    def test_tqdm_conversions_lih_sto3g(self):
        filename = "H1-Li1_sto-3g_singlet_1.45.hdf5"
        molecule = MolecularData(
            filename=os.path.join(DATA_DIRECTORY, filename))
        true_tqdm = self.tqdm_lih_sto3g

        test_tqdm = map_two_pdm_to_two_hole_dm(molecule.fci_two_rdm,
                                               molecule.fci_one_rdm)
        assert numpy.allclose(true_tqdm, test_tqdm)

        true_oqdm = numpy.eye(molecule.n_qubits) - molecule.fci_one_rdm
        test_oqdm = map_two_hole_dm_to_one_hole_dm(
            true_tqdm, molecule.n_qubits - molecule.n_electrons)
        assert numpy.allclose(true_oqdm, test_oqdm)

        test_tpdm = map_two_hole_dm_to_two_pdm(true_tqdm, molecule.fci_one_rdm)
        assert numpy.allclose(test_tpdm, molecule.fci_two_rdm)

    def test_phdm_conversions_h2_631g(self):
        filename = "H2_6-31g_singlet_0.75.hdf5"
        molecule = MolecularData(
            filename=os.path.join(DATA_DIRECTORY, filename))
        true_phdm = self.phdm_h2_6_31g

        test_phdm = map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm,
                                                    molecule.fci_one_rdm)
        assert numpy.allclose(test_phdm, true_phdm)

        test_opdm = map_particle_hole_dm_to_one_pdm(true_phdm,
                                                    molecule.n_electrons,
                                                    molecule.n_qubits)
        assert numpy.allclose(test_opdm, molecule.fci_one_rdm)

        test_tpdm = map_particle_hole_dm_to_two_pdm(true_phdm,
                                                    molecule.fci_one_rdm)
        assert numpy.allclose(test_tpdm, molecule.fci_two_rdm)

    def test_phdm_conversions_h2_sto3g(self):
        filename = "H2_sto-3g_singlet_1.4.hdf5"
        molecule = MolecularData(
            filename=os.path.join(DATA_DIRECTORY, filename))
        true_phdm = self.phdm_h2_sto3g

        test_phdm = map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm,
                                                    molecule.fci_one_rdm)
        assert numpy.allclose(test_phdm, true_phdm)

        test_opdm = map_particle_hole_dm_to_one_pdm(true_phdm,
                                                    molecule.n_electrons,
                                                    molecule.n_qubits)
        assert numpy.allclose(test_opdm, molecule.fci_one_rdm)

        test_tpdm = map_particle_hole_dm_to_two_pdm(true_phdm,
                                                    molecule.fci_one_rdm)
        assert numpy.allclose(test_tpdm, molecule.fci_two_rdm)

    def test_phdm_conversions_lih_sto3g(self):
        filename = "H1-Li1_sto-3g_singlet_1.45.hdf5"
        molecule = MolecularData(
            filename=os.path.join(DATA_DIRECTORY, filename))
        true_phdm = self.phdm_lih_sto3g

        test_phdm = map_two_pdm_to_particle_hole_dm(molecule.fci_two_rdm,
                                                    molecule.fci_one_rdm)
        assert numpy.allclose(test_phdm, true_phdm)

        test_opdm = map_particle_hole_dm_to_one_pdm(true_phdm,
                                                    molecule.n_electrons,
                                                    molecule.n_qubits)
        assert numpy.allclose(test_opdm, molecule.fci_one_rdm)

        test_tpdm = map_particle_hole_dm_to_two_pdm(true_phdm,
                                                    molecule.fci_one_rdm)
        assert numpy.allclose(test_tpdm, molecule.fci_two_rdm)
