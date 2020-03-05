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

"""Tests for molecular_data."""

import unittest
import numpy.random
import scipy.linalg

from openfermion.config import *
from openfermion.hamiltonians import jellium_model, make_atom
from openfermion.hamiltonians._molecular_data import *
from openfermion.transforms import (get_interaction_operator,
                                    get_molecular_data)
from openfermion.utils import *


class MolecularDataTest(unittest.TestCase):

    def setUp(self):
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.filename = os.path.join(THIS_DIRECTORY, 'data',
                                     'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity,
            filename=self.filename)
        self.molecule.load()

    def testUnitConversion(self):
        """Test the unit conversion routines"""
        unit_angstrom = 1.0
        bohr = angstroms_to_bohr(unit_angstrom)
        self.assertAlmostEqual(bohr, 1.889726)
        inverse_transform = bohr_to_angstroms(bohr)
        self.assertAlmostEqual(inverse_transform, 1.0)

    def test_name_molecule(self):
        charge = 0
        correct_name = str('H2_sto-3g_singlet_0.7414')
        computed_name = name_molecule(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      charge,
                                      description="0.7414")
        self.assertEqual(correct_name, computed_name)
        self.assertEqual(correct_name, self.molecule.name)

        # Check (+) charge
        charge = 1
        correct_name = "H2_sto-3g_singlet_1+_0.7414"
        computed_name = name_molecule(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      charge,
                                      description="0.7414")
        self.assertEqual(correct_name, computed_name)

        # Check > 1 atom type
        charge = 0
        correct_name = "H1-F1_sto-3g_singlet_1.0"
        test_geometry = [('H', (0, 0, 0)), ('F', (0, 0, 1.0))]
        computed_name = name_molecule(test_geometry,
                                      self.basis,
                                      self.multiplicity,
                                      charge,
                                      description="1.0")
        self.assertEqual(correct_name, computed_name)

        # Check errors in naming
        with self.assertRaises(TypeError):
            test_molecule = MolecularData(self.geometry, self.basis,
                                          self.multiplicity, description=5)
        correct_name = str('H2_sto-3g_singlet')
        test_molecule = self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity,
            data_directory=DATA_DIRECTORY)
        self.assertSequenceEqual(correct_name, test_molecule.name)

    def test_invalid_multiplicity(self):
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = -1
        with self.assertRaises(MoleculeNameError):
            MolecularData(geometry, basis, multiplicity)

    def test_geometry_from_file(self):
        water_geometry = [('O', (0., 0., 0.)),
                          ('H', (0.757, 0.586, 0.)),
                          ('H', (-.757, 0.586, 0.))]
        filename = os.path.join(THIS_DIRECTORY, 'data', 'geometry_example.txt')
        test_geometry = geometry_from_file(filename)
        for atom in range(3):
            self.assertAlmostEqual(water_geometry[atom][0],
                                   test_geometry[atom][0])
            for coordinate in range(3):
                self.assertAlmostEqual(water_geometry[atom][1][coordinate],
                                       test_geometry[atom][1][coordinate])

    def test_save_load(self):
        n_atoms = self.molecule.n_atoms
        orbitals = self.molecule.canonical_orbitals
        self.assertFalse(orbitals is None)
        self.molecule.n_atoms += 1
        self.assertEqual(self.molecule.n_atoms, n_atoms + 1)
        self.molecule.load()
        self.assertEqual(self.molecule.n_atoms, n_atoms)
        dummy_data = self.molecule.get_from_file("dummy_entry")
        self.assertTrue(dummy_data is None)

    def test_dummy_save(self):

        # Make fake molecule.
        filename = os.path.join(THIS_DIRECTORY, 'data', 'dummy_molecule')
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = '6-31g*'
        multiplicity = 7
        charge = -1
        description = 'openfermion_forever'
        molecule = MolecularData(geometry, basis, multiplicity,
                                 charge, description, filename)

        # Make some attributes to save.
        molecule.n_orbitals = 10
        molecule.n_qubits = 10
        molecule.nuclear_repulsion = -12.3
        molecule.hf_energy = 99.
        molecule.canonical_orbitals = [1, 2, 3, 4]
        molecule.orbital_energies = [5, 6, 7, 8]
        molecule.one_body_integrals = [5, 6, 7, 8]
        molecule.two_body_integrals = [5, 6, 7, 8]
        molecule.mp2_energy = -12.
        molecule.cisd_energy = 32.
        molecule.cisd_one_rdm = numpy.arange(10)
        molecule.cisd_two_rdm = numpy.arange(10)
        molecule.fci_energy = 232.
        molecule.fci_one_rdm = numpy.arange(11)
        molecule.fci_two_rdm = numpy.arange(11)
        molecule.ccsd_energy = 88.
        molecule.ccsd_single_amps = [1, 2, 3]
        molecule.ccsd_double_amps = [1, 2, 3]
        molecule.general_calculations['Fake CI'] = 1.2345
        molecule.general_calculations['Fake CI 2'] = 5.2345

        # Test missing calculation and information exceptions
        molecule.one_body_integrals = None
        with self.assertRaises(MissingCalculationError):
            molecule.get_integrals()
        molecule.hf_energy = 99.

        with self.assertRaises(ValueError):
            molecule.get_active_space_integrals([], [])

        molecule.fci_energy = None
        with self.assertRaises(MissingCalculationError):
            molecule.get_molecular_rdm(use_fci=True)
        molecule.fci_energy = 232.

        molecule.cisd_energy = None
        with self.assertRaises(MissingCalculationError):
            molecule.get_molecular_rdm(use_fci=False)
        molecule.cisd_energy = 232.

        # Save molecule.
        molecule.save()

        try:
            # Change attributes and load.
            molecule.ccsd_energy = -2.232

            # Load molecule.
            new_molecule = MolecularData(filename=filename)
            molecule.general_calculations = {}
            molecule.load()
            self.assertEqual(molecule.general_calculations['Fake CI'],
                             1.2345)
            # Tests re-load functionality
            molecule.save()

            # Check CCSD energy.
            self.assertAlmostEqual(new_molecule.ccsd_energy,
                                   molecule.ccsd_energy)
            self.assertAlmostEqual(molecule.ccsd_energy, 88.)
        finally:
            os.remove(filename + '.hdf5')

    def test_file_loads(self):
        """Test different filename specs"""
        data_directory = os.path.join(THIS_DIRECTORY, 'data')
        molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity,
            filename=self.filename)
        test_hf_energy = molecule.hf_energy
        molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity,
            filename=self.filename + ".hdf5",
            data_directory=data_directory)
        self.assertAlmostEqual(test_hf_energy, molecule.hf_energy)

        molecule = MolecularData(filename=self.filename + ".hdf5")
        integrals = molecule.one_body_integrals
        self.assertTrue(integrals is not None)

        with self.assertRaises(ValueError):
            MolecularData()

    def test_active_space(self):
        """Test simple active space truncation features"""

        # Start w/ no truncation
        core_const, one_body_integrals, two_body_integrals = (
            self.molecule.get_active_space_integrals(active_indices=[0, 1]))

        self.assertAlmostEqual(core_const, 0.0)
        self.assertAlmostEqual(scipy.linalg.norm(one_body_integrals -
                               self.molecule.one_body_integrals), 0.0)
        self.assertAlmostEqual(scipy.linalg.norm(two_body_integrals -
                               self.molecule.two_body_integrals), 0.0)

    def test_energies(self):
        self.assertAlmostEqual(self.molecule.hf_energy, -1.1167, places=4)
        self.assertAlmostEqual(self.molecule.mp2_energy, -1.1299, places=4)
        self.assertAlmostEqual(self.molecule.cisd_energy, -1.1373, places=4)
        self.assertAlmostEqual(self.molecule.ccsd_energy, -1.1373, places=4)
        self.assertAlmostEqual(self.molecule.ccsd_energy, -1.1373, places=4)

    def test_rdm_and_rotation(self):

        # Compute total energy from RDM.
        molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()
        molecular_rdm = self.molecule.get_molecular_rdm()
        total_energy = molecular_rdm.expectation(molecular_hamiltonian)
        self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)

        # Build random rotation with correction dimension.
        num_spatial_orbitals = self.molecule.n_orbitals
        rotation_generator = numpy.random.randn(
            num_spatial_orbitals, num_spatial_orbitals)
        rotation_matrix = scipy.linalg.expm(
            rotation_generator - rotation_generator.T)

        # Compute total energy from RDM under some basis set rotation.
        molecular_rdm.rotate_basis(rotation_matrix)
        molecular_hamiltonian.rotate_basis(rotation_matrix)
        total_energy = molecular_rdm.expectation(molecular_hamiltonian)
        self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)

    def test_get_up_down_electrons(self):
        largest_atom = 10
        # Test first row
        correct_alpha = [0, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5]
        correct_beta = [0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 5]
        for n_electrons in range(1, largest_atom  + 1):
            # Make molecule.
            basis = 'sto-3g'
            atom_name = periodic_table[n_electrons]
            molecule = make_atom(atom_name, basis)

            # Test.
            self.assertAlmostEqual(molecule.get_n_alpha_electrons(),
                                   correct_alpha[n_electrons])
            self.assertAlmostEqual(molecule.get_n_beta_electrons(),
                                   correct_beta[n_electrons])

    def test_abstract_molecule(self):
        """Test an abstract molecule like jellium for saving and loading"""
        jellium_interaction = get_interaction_operator(
            jellium_model(Grid(2, 2, 1.0)))
        jellium_molecule = get_molecular_data(jellium_interaction,
                                              geometry="Jellium",
                                              basis="PlaneWave22",
                                              multiplicity=1,
                                              n_electrons=4)

        jellium_filename = jellium_molecule.filename
        jellium_molecule.save()
        jellium_molecule.load()
        correct_name = "Jellium_PlaneWave22_singlet"
        self.assertEqual(jellium_molecule.name, correct_name)
        os.remove("{}.hdf5".format(jellium_filename))

    def test_load_molecular_hamiltonian(self):
        bond_length = 1.45
        geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., bond_length))]

        lih_hamiltonian = load_molecular_hamiltonian(
                geometry, 'sto-3g', 1, format(bond_length), 2, 2)
        self.assertEqual(count_qubits(lih_hamiltonian), 4)

        lih_hamiltonian = load_molecular_hamiltonian(
                geometry, 'sto-3g', 1, format(bond_length), 2, 3)
        self.assertEqual(count_qubits(lih_hamiltonian), 6)

        lih_hamiltonian = load_molecular_hamiltonian(
                geometry, 'sto-3g', 1, format(bond_length), None, None)
        self.assertEqual(count_qubits(lih_hamiltonian), 12)
