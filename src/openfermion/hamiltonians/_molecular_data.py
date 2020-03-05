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

"""Class and functions to store quantum chemistry data."""

import uuid
import shutil
import os
import numpy
import h5py

from openfermion.config import *
from openfermion.ops import InteractionOperator, InteractionRDM


r"""NOTE ON PQRS CONVENTION:
  The data structures which hold fermionic operators / integrals /
  coefficients assume a particular convention which depends on how integrals
  are labeled:
  h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
  h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
  With this convention, the molecular Hamiltonian becomes
  H =\sum_{p,q} h[p,q] a_p^\dagger a_q
    + 0.5 * \sum_{p,q,r,s} h[p,q,r,s] a_p^\dagger a_q^\dagger a_r a_s
"""

# Define a compatible basestring for checking between Python 2 and 3
try:
    basestring
except NameError:  # pragma: no cover
    basestring = str


# Define error objects which inherit from Exception.
class MoleculeNameError(Exception):
    pass


class MissingCalculationError(Exception):
    pass


# Functions to change from Bohr to angstroms and back.
def bohr_to_angstroms(distance):
    # Value defined so it is the inverse to numerical precision of angs to bohr
    return 0.5291772458017723 * distance


def angstroms_to_bohr(distance):
    return 1.889726 * distance


# The Periodic Table as a python list and dictionary.
periodic_table = [
    '?',
    'H', 'He',
    'Li', 'Be',
    'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
periodic_hash_table = {}
for atomic_number, atom in enumerate(periodic_table):
    periodic_hash_table[atom] = atomic_number


# Spin polarization of atoms on period table.
periodic_polarization = [-1,
                         1, 0,
                         1, 0, 1, 2, 3, 2, 1, 0,
                         1, 0, 1, 2, 3, 2, 1, 0,
                         1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
                         1, 0, 1, 2, 5, 6, 5, 8, 9, 0, 1, 0, 1, 2, 3, 2, 1, 0]


def name_molecule(geometry,
                  basis,
                  multiplicity,
                  charge,
                  description):
    """Function to name molecules.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the total molecular charge.
        description: A string giving a description. As an example,
            for dimers a likely description is the bond length (e.g. 0.7414).

    Returns:
        name: A string giving the name of the instance.

    Raises:
        MoleculeNameError: If spin multiplicity is not valid.
    """
    if not isinstance(geometry, basestring):
        # Get sorted atom vector.
        atoms = [item[0] for item in geometry]
        atom_charge_info = [(atom, atoms.count(atom)) for atom in set(atoms)]
        sorted_info = sorted(atom_charge_info,
                             key=lambda atom: periodic_hash_table[atom[0]])

        # Name molecule.
        name = '{}{}'.format(sorted_info[0][0], sorted_info[0][1])
        for info in sorted_info[1::]:
            name += '-{}{}'.format(info[0], info[1])
    else:
        name = geometry

    # Add basis.
    name += '_{}'.format(basis)

    # Add multiplicity.
    multiplicity_dict = {1: 'singlet',
                         2: 'doublet',
                         3: 'triplet',
                         4: 'quartet',
                         5: 'quintet',
                         6: 'sextet',
                         7: 'septet',
                         8: 'octet',
                         9: 'nonet',
                         10: 'dectet',
                         11: 'undectet',
                         12: 'duodectet'}
    if (multiplicity not in multiplicity_dict):
        raise MoleculeNameError('Invalid spin multiplicity provided.')
    else:
        name += '_{}'.format(multiplicity_dict[multiplicity])

    # Add charge.
    if charge > 0:
        name += '_{}+'.format(charge)
    elif charge < 0:
        name += '_{}-'.format(charge)

    # Optionally add descriptive tag and return.
    if description:
        name += '_{}'.format(description)
    return name


def geometry_from_file(file_name):
    """Function to create molecular geometry from text file.

    Args:
        file_name: a string giving the location of the geometry file.
            It is assumed that geometry is given for each atom on line, e.g.:
            H 0. 0. 0.
            H 0. 0. 0.7414

    Returns:
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to specify atoms.
    """
    geometry = []
    with open(file_name, 'r') as stream:
        for line in stream:
            data = line.split()
            if len(data) == 4:
                atom = data[0]
                coordinates = (float(data[1]), float(data[2]), float(data[3]))
                geometry += [(atom, coordinates)]
    return geometry


class MolecularData(object):

    """Class for storing molecule data from a fixed basis set at a fixed
    geometry that is obtained from classical electronic structure
    packages. Not every field is filled in every calculation. All data
    that can (for some instance) exceed 10 MB should be saved
    separately. Data saved in HDF5 format.

    Attributes:
        geometry: A list of tuples giving the coordinates of each atom. An
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))]. Distances
            in angstrom. Use atomic symbols to specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
        charge: An integer giving the total molecular charge. Defaults to 0.
        multiplicity: An integer giving the spin multiplicity.
        description: An optional string giving a description. As an example,
            for dimers a likely description is the bond length (e.g. 0.7414).
        name: A string giving a characteristic name for the instance.
        filename: The name of the file where the molecule data is saved.
        n_atoms: Integer giving the number of atoms in the molecule.
        n_electrons: Integer giving the number of electrons in the molecule.
        atoms: List of the atoms in molecule sorted by atomic number.
        protons: List of atomic charges in molecule sorted by atomic number.
        hf_energy: Energy from open or closed shell Hartree-Fock.
        nuclear_repulsion: Energy from nuclei-nuclei interaction.
        canonical_orbitals: numpy array giving canonical orbital coefficients.
        n_orbitals: Integer giving total number of spatial orbitals.
        n_qubits: Integer giving total number of qubits that would be needed.
        orbital_energies: Numpy array giving the canonical orbital energies.
        fock_matrix: Numpy array giving the Fock matrix.
        overlap_integrals: Numpy array of AO overlap integrals
        one_body_integrals: Numpy array of one-electron integrals
        two_body_integrals: Numpy array of two-electron integrals
        mp2_energy: Energy from MP2 perturbation theory.
        cisd_energy: Energy from configuration interaction singles + doubles.
        cisd_one_rdm: Numpy array giving 1-RDM from CISD calculation.
        cisd_two_rdm: Numpy array giving 2-RDM from CISD calculation.
        fci_energy: Exact energy of molecule within given basis.
        fci_one_rdm: Numpy array giving 1-RDM from FCI calculation.
        fci_two_rdm: Numpy array giving 2-RDM from FCI calculation.
        ccsd_energy: Energy from coupled cluster singles + doubles.
        ccsd_single_amps: Numpy array holding single amplitudes
        ccsd_double_amps: Numpy array holding double amplitudes
        general_calculations: A dictionary storing general calculation results
            for this system annotated by the key.
    """
    def __init__(self, geometry=None, basis=None, multiplicity=None,
                 charge=0, description="", filename="", data_directory=None):
        """Initialize molecular metadata which defines class.

        Args:
            geometry: A list of tuples giving the coordinates of each atom.
                An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
                Distances in angstrom. Use atomic symbols to
                specify atoms. Only optional if loading from file.
            basis: A string giving the basis set. An example is 'cc-pvtz'.
                Only optional if loading from file.
            charge: An integer giving the total molecular charge. Defaults
                to 0.  Only optional if loading from file.
            multiplicity: An integer giving the spin multiplicity.  Only
                optional if loading from file.
            description: A optional string giving a description. As an
                example, for dimers a likely description is the bond length
                (e.g. 0.7414).
            filename: An optional string giving name of file.
                If filename is not provided, one is generated automatically.
            data_directory: Optional data directory to change from default
                data directory specified in config file.
        """
        # Check appropriate data as been provided and autoload if requested.
        if ((geometry is None) or
                (basis is None) or
                (multiplicity is None)):
            if filename:
                if filename[-5:] == '.hdf5':
                    self.filename = filename[:(len(filename) - 5)]
                else:
                    self.filename = filename
                self.load()
                self.init_lazy_properties()
                return
            else:
                raise ValueError("Geometry, basis, multiplicity must be"
                                 "specified when not loading from file.")

        # Metadata fields which must be provided.
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity

        # Metadata fields with default values.
        self.charge = charge
        if (not isinstance(description, basestring)):
            raise TypeError("description must be a string.")
        self.description = description

        # Name molecule and get associated filename
        self.name = name_molecule(geometry, basis, multiplicity,
                                  charge, description)
        if filename:
            if filename[-5:] == '.hdf5':
                filename = filename[:(len(filename) - 5)]
            self.filename = filename
        else:
            if data_directory is None:
                self.filename = DATA_DIRECTORY + '/' + self.name
            else:
                self.filename = data_directory + '/' + self.name

        # Attributes generated automatically by class.
        if not isinstance(geometry, basestring):
            self.n_atoms = len(geometry)
            self.atoms = sorted([row[0] for row in geometry],
                                key=lambda atom: periodic_hash_table[atom])
            self.protons = [periodic_hash_table[atom] for atom in self.atoms]
            self.n_electrons = sum(self.protons) - charge
        else:
            self.n_atoms = 0
            self.atoms = []
            self.protons = 0
            self.n_electrons = 0

        # Generic attributes from calculations.
        self.n_orbitals = None
        self.n_qubits = None
        self.nuclear_repulsion = None

        # Attributes generated from SCF calculation.
        self.hf_energy = None
        self.orbital_energies = None

        # Attributes generated from MP2 calculation.
        self.mp2_energy = None

        # Attributes generated from CISD calculation.
        self.cisd_energy = None

        # Attributes generated from exact diagonalization.
        self.fci_energy = None

        # Attributes generated from CCSD calculation.
        self.ccsd_energy = None

        # General calculation results
        self.general_calculations = {}

        # Initialize attributes that will be loaded only upon demand
        self.init_lazy_properties()

    def init_lazy_properties(self):
        """Initializes properties loaded on demand to None"""

        # Molecular orbitals
        self._canonical_orbitals = None

        # Overlap matrix corresponding to bare orbitals defining MOs
        self._overlap_integrals = None

        # Electronic Integrals
        self._one_body_integrals = None
        self._two_body_integrals = None

        # CI RDMs
        self._cisd_one_rdm = None
        self._cisd_two_rdm = None

        # FCI RDMs
        self._fci_one_rdm = None
        self._fci_two_rdm = None

        # Coupled cluster amplitudes
        self._ccsd_single_amps = None
        self._ccsd_double_amps = None

    # The following block of property getters and setters allow class
    # attributes to be used as if they were stored in the class, but are
    # actually loaded only upon request from file.  This greatly speeds up
    # calculations and saves considerable memory in cases where some of the
    # 4-index quantities are not used.

    @property
    def canonical_orbitals(self):
        if self._canonical_orbitals is None:
            data = self.get_from_file("canonical_orbitals")
            self._canonical_orbitals = (data if data is not None and
                                        data.dtype.num != 0 else None)
        return self._canonical_orbitals

    @canonical_orbitals.setter
    def canonical_orbitals(self, value):
        self._canonical_orbitals = value

    @property
    def overlap_integrals(self):
        if self._overlap_integrals is None:
            data = self.get_from_file("overlap_integrals")
            self._overlap_integrals = (data if data is not None and
                                       data.dtype.num != 0 else None)
        return self._overlap_integrals

    @overlap_integrals.setter
    def overlap_integrals(self, value):
        self._overlap_integrals = value

    @property
    def one_body_integrals(self):
        if self._one_body_integrals is None:
            data = self.get_from_file("one_body_integrals")
            self._one_body_integrals = (data if data is not None and
                                        data.dtype.num != 0 else None)
        return self._one_body_integrals

    @one_body_integrals.setter
    def one_body_integrals(self, value):
        self._one_body_integrals = value

    @property
    def two_body_integrals(self):
        if self._two_body_integrals is None:
            data = self.get_from_file("two_body_integrals")
            self._two_body_integrals = (data if data is not None and
                                        data.dtype.num != 0 else None)
        return self._two_body_integrals

    @two_body_integrals.setter
    def two_body_integrals(self, value):
        self._two_body_integrals = value

    @property
    def cisd_one_rdm(self):
        if self._cisd_one_rdm is None:
            data = self.get_from_file("cisd_one_rdm")
            self._cisd_one_rdm = (data if data is not None and
                                  data.dtype.num != 0 else None)
        return self._cisd_one_rdm

    @cisd_one_rdm.setter
    def cisd_one_rdm(self, value):
        self._cisd_one_rdm = value

    @property
    def cisd_two_rdm(self):
        if self._cisd_two_rdm is None:
            data = self.get_from_file("cisd_two_rdm")
            self._cisd_two_rdm = (data if data is not None and
                                  data.dtype.num != 0 else None)
        return self._cisd_two_rdm

    @cisd_two_rdm.setter
    def cisd_two_rdm(self, value):
        self._cisd_two_rdm = value

    @property
    def fci_one_rdm(self):
        if self._fci_one_rdm is None:
            data = self.get_from_file("fci_one_rdm")
            self._fci_one_rdm = (data if data is not None and
                                 data.dtype.num != 0 else None)
        return self._fci_one_rdm

    @fci_one_rdm.setter
    def fci_one_rdm(self, value):
        self._fci_one_rdm = value

    @property
    def fci_two_rdm(self):
        if self._fci_two_rdm is None:
            data = self.get_from_file("fci_two_rdm")
            self._fci_two_rdm = (data if data is not None and
                                 data.dtype.num != 0 else None)
        return self._fci_two_rdm

    @fci_two_rdm.setter
    def fci_two_rdm(self, value):
        self._fci_two_rdm = value

    @property
    def ccsd_single_amps(self):
        if self._ccsd_single_amps is None:
            data = self.get_from_file("ccsd_single_amps")
            self._ccsd_single_amps = (data if data is not None and
                                      data.dtype.num != 0 else None)
        return self._ccsd_single_amps

    @ccsd_single_amps.setter
    def ccsd_single_amps(self, value):
        self._ccsd_single_amps = value

    @property
    def ccsd_double_amps(self):
        if self._ccsd_double_amps is None:
            data = self.get_from_file("ccsd_double_amps")
            self._ccsd_double_amps = (data if data is not None and
                                      data.dtype.num != 0 else None)
        return self._ccsd_double_amps

    @ccsd_double_amps.setter
    def ccsd_double_amps(self, value):
        self._ccsd_double_amps = value

    def save(self):
        """Method to save the class under a systematic name."""
        # Create a temporary file and swap it to the original name in case
        # data needs to be loaded while saving
        tmp_name = uuid.uuid4()
        with h5py.File("{}.hdf5".format(tmp_name), "w") as f:
            # Save geometry (atoms and positions need to be separate):
            d_geom = f.create_group("geometry")
            if not isinstance(self.geometry, basestring):
                atoms = [numpy.string_(item[0]) for item in self.geometry]
                positions = numpy.array([list(item[1])
                                         for item in self.geometry])
            else:
                atoms = numpy.string_(self.geometry)
                positions = None
            d_geom.create_dataset("atoms", data=(atoms if atoms is not None
                                                 else False))
            d_geom.create_dataset("positions", data=(positions if positions
                                                     is not None else False))
            # Save basis:
            f.create_dataset("basis", data=numpy.string_(self.basis))
            # Save multiplicity:
            f.create_dataset("multiplicity", data=self.multiplicity)
            # Save charge:
            f.create_dataset("charge", data=self.charge)
            # Save description:
            f.create_dataset("description",
                             data=numpy.string_(self.description))
            # Save name:
            f.create_dataset("name", data=numpy.string_(self.name))
            # Save n_atoms:
            f.create_dataset("n_atoms", data=self.n_atoms)
            # Save atoms:
            f.create_dataset("atoms", data=numpy.string_(self.atoms))
            # Save protons:
            f.create_dataset("protons", data=self.protons)
            # Save n_electrons:
            f.create_dataset("n_electrons", data=self.n_electrons)
            # Save generic attributes from calculations:
            f.create_dataset("n_orbitals",
                             data=(self.n_orbitals if self.n_orbitals
                                   is not None else False))
            f.create_dataset("n_qubits",
                             data=(self.n_qubits if
                                   self.n_qubits is not None else False))
            f.create_dataset("nuclear_repulsion",
                             data=(self.nuclear_repulsion if
                                   self.nuclear_repulsion is not None else
                                   False))
            # Save attributes generated from SCF calculation.
            f.create_dataset("hf_energy", data=(self.hf_energy if
                                                self.hf_energy is not None
                                                else False))
            f.create_dataset("canonical_orbitals",
                             data=(self.canonical_orbitals if
                                   self.canonical_orbitals is
                                   not None else False),
                             compression=("gzip" if self.canonical_orbitals
                                          is not None else None))
            f.create_dataset("overlap_integrals",
                             data=(self.overlap_integrals if
                                   self.overlap_integrals is
                                   not None else False),
                             compression=("gzip" if self.overlap_integrals
                                          is not None else None))
            f.create_dataset("orbital_energies",
                             data=(self.orbital_energies if
                                   self.orbital_energies is not None else
                                   False))
            # Save attributes generated from integrals.
            f.create_dataset("one_body_integrals",
                             data=(self.one_body_integrals if
                                   self.one_body_integrals is
                                   not None else False),
                             compression=("gzip" if self.one_body_integrals
                                          is not None else None))
            f.create_dataset("two_body_integrals",
                             data=(self.two_body_integrals if
                                   self.two_body_integrals is
                                   not None else False),
                             compression=("gzip" if self.two_body_integrals
                                          is not None else None))
            # Save attributes generated from MP2 calculation.
            f.create_dataset("mp2_energy",
                             data=(self.mp2_energy if
                                   self.mp2_energy is not None else False))
            # Save attributes generated from CISD calculation.
            f.create_dataset("cisd_energy",
                             data=(self.cisd_energy if
                                   self.cisd_energy is not None else False))
            f.create_dataset("cisd_one_rdm",
                             data=(self.cisd_one_rdm if
                                   self.cisd_one_rdm is not None else False),
                             compression=("gzip" if self.cisd_one_rdm
                                          is not None else None))
            f.create_dataset("cisd_two_rdm",
                             data=(self.cisd_two_rdm if
                                   self.cisd_two_rdm is not None else False),
                             compression=("gzip" if self.cisd_two_rdm
                                          is not None else None))
            # Save attributes generated from exact diagonalization.
            f.create_dataset("fci_energy",
                             data=(self.fci_energy if
                                   self.fci_energy is not None else False))
            f.create_dataset("fci_one_rdm",
                             data=(self.fci_one_rdm if
                                   self.fci_one_rdm is not None else False),
                             compression=("gzip" if self.fci_one_rdm
                                          is not None else None))
            f.create_dataset("fci_two_rdm",
                             data=(self.fci_two_rdm if
                                   self.fci_two_rdm is not None else False),
                             compression=("gzip" if self.fci_two_rdm is not
                                          None else None))
            # Save attributes generated from CCSD calculation.
            f.create_dataset("ccsd_energy",
                             data=(self.ccsd_energy if
                                   self.ccsd_energy is not None else False))
            f.create_dataset("ccsd_single_amps",
                             data=(self.ccsd_single_amps
                                   if self.ccsd_single_amps is not None else
                                   False),
                             compression=("gzip" if self.ccsd_single_amps
                                          is not None else None))
            f.create_dataset("ccsd_double_amps",
                             data=(self.ccsd_double_amps
                                   if self.ccsd_double_amps is
                                   not None else False),
                             compression=("gzip" if self.ccsd_double_amps
                                          is not None else None))

            # Save general calculation data
            key_list = list(self.general_calculations.keys())
            f.create_dataset("general_calculations_keys",
                             data=([numpy.string_(key) for key in key_list] if
                                   len(key_list) > 0 else False))
            f.create_dataset("general_calculations_values",
                             data=([self.general_calculations[key] for
                                   key in key_list] if
                                   len(key_list) > 0 else False))

        # Remove old file first for compatibility with systems that don't allow
        # rename replacement.  Catching OSError for when file does not exist
        # yet
        try:
            os.remove("{}.hdf5".format(self.filename))
        except OSError:
            pass

        shutil.move("{}.hdf5".format(tmp_name), "{}.hdf5".format(self.filename))

    def load(self):
        geometry = []

        with h5py.File("{}.hdf5".format(self.filename), "r") as f:
            # Load geometry:
            data = f["geometry/atoms"]
            if data.shape != (()):
                for atom, pos in zip(f["geometry/atoms"][...],
                                     f["geometry/positions"][...]):
                    geometry.append((atom.tobytes().
                                     decode('utf-8'), list(pos)))
                self.geometry = geometry
            else:
                self.geometry = data[...].tobytes().decode('utf-8')
            # Load basis:
            self.basis = f["basis"][...].tobytes().decode('utf-8')
            # Load multiplicity:
            self.multiplicity = int(f["multiplicity"][...])
            # Load charge:
            self.charge = int(f["charge"][...])
            # Load description:
            self.description = f["description"][...].tobytes().decode(
                    'utf-8').rstrip(u'\x00')
            # Load name:
            self.name = f["name"][...].tobytes().decode('utf-8')
            # Load n_atoms:
            self.n_atoms = int(f["n_atoms"][...])
            # Load atoms:
            self.atoms = f["atoms"][...]
            # Load protons:
            self.protons = f["protons"][...]
            # Load n_electrons:
            self.n_electrons = int(f["n_electrons"][...])
            # Load generic attributes from calculations:
            data = f["n_orbitals"][...]
            self.n_orbitals = int(data) if data.dtype.num != 0 else None
            data = f["n_qubits"][...]
            self.n_qubits = int(data) if data.dtype.num != 0 else None
            data = f["nuclear_repulsion"][...]
            self.nuclear_repulsion = (float(data) if data.dtype.num != 0 else
                                      None)
            # Load attributes generated from SCF calculation.
            data = f["hf_energy"][...]
            self.hf_energy = data if data.dtype.num != 0 else None
            data = f["orbital_energies"][...]
            self.orbital_energies = data if data.dtype.num != 0 else None
            # Load attributes generated from MP2 calculation.
            data = f["mp2_energy"][...]
            self.mp2_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from CISD calculation.
            data = f["cisd_energy"][...]
            self.cisd_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from exact diagonalization.
            data = f["fci_energy"][...]
            self.fci_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from CCSD calculation.
            data = f["ccsd_energy"][...]
            self.ccsd_energy = data if data.dtype.num != 0 else None
            # Load general calculations
            if ("general_calculations_keys" in f and
                    "general_calculations_values" in f):
                keys = f["general_calculations_keys"]
                values = f["general_calculations_values"]
                if keys.shape != (()):
                    self.general_calculations = {
                        key.tobytes().decode('utf-8'): value for key, value
                        in zip(keys[...], values[...])}
            else:
                self.general_calculations = None

    def get_from_file(self, property_name):
        """Helper routine to re-open HDF5 file and pull out single property

        Args:
            property_name: Property name to load from self.filename

        Returns:
            The data located at file[property_name] for the HDF5 file at
                self.filename. Returns None if the key is not found in the
                file.
        """
        try:
            with h5py.File("{}.hdf5".format(self.filename), "r") as f:
                data = f[property_name][...]
        except KeyError:
            data = None
        except IOError:
            data = None
        return data

    def get_n_alpha_electrons(self):
        """Return number of alpha electrons."""
        return int((self.n_electrons + (self.multiplicity - 1)) // 2)

    def get_n_beta_electrons(self):
        """Return number of beta electrons."""
        return int((self.n_electrons - (self.multiplicity - 1)) // 2)

    def get_integrals(self):
        """Method to return 1-electron and 2-electron integrals in MO basis.

        Returns:
            one_body_integrals: An array of the one-electron integrals having
                shape of (n_orbitals, n_orbitals).
            two_body_integrals: An array of the two-electron integrals having
                shape of (n_orbitals, n_orbitals, n_orbitals, n_orbitals).

        Raises:
          MisissingCalculationError: If integrals are not calculated.
        """
        # Make sure integrals have been computed.
        if self.one_body_integrals is None or self.two_body_integrals is None:
            raise MissingCalculationError(
                'Missing integral calculation in {}, run before loading '
                'integrals.'.format(self.filename))
        return self.one_body_integrals, self.two_body_integrals

    def get_active_space_integrals(self,
                                   occupied_indices=None,
                                   active_indices=None):
        """Restricts a molecule at a spatial orbital level to an active space

        This active space may be defined by a list of active indices and
            doubly occupied indices. Note that one_body_integrals and
            two_body_integrals must be defined
            n an orthonormal basis set.

        Args:
            occupied_indices: A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
            active_indices: A list of spatial orbital indices indicating
                which orbitals should be considered active.

        Returns:
            tuple: Tuple with the following entries:

            **core_constant**: Adjustment to constant shift in Hamiltonian
            from integrating out core orbitals

            **one_body_integrals_new**: one-electron integrals over active
            space.

            **two_body_integrals_new**: two-electron integrals over active
            space.
        """
        # Fix data type for a few edge cases
        occupied_indices = [] if occupied_indices is None else occupied_indices
        if (len(active_indices) < 1):
            raise ValueError('Some active indices required for reduction.')

        # Get integrals.
        one_body_integrals, two_body_integrals = self.get_integrals()

        # Determine core constant
        core_constant = 0.0
        for i in occupied_indices:
            core_constant += 2 * one_body_integrals[i, i]
            for j in occupied_indices:
                core_constant += (2 * two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])

        # Modified one electron integrals
        one_body_integrals_new = numpy.copy(one_body_integrals)
        for u in active_indices:
            for v in active_indices:
                for i in occupied_indices:
                    one_body_integrals_new[u, v] += (
                        2 * two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])

        # Restrict integral ranges and change M appropriately
        return (core_constant,
                one_body_integrals_new[numpy.ix_(active_indices,
                                                 active_indices)],
                two_body_integrals[numpy.ix_(active_indices,
                                             active_indices,
                                             active_indices,
                                             active_indices)])

    def get_molecular_hamiltonian(self, occupied_indices=None,
                                  active_indices=None):
        """Output arrays of the second quantized Hamiltonian coefficients.

        Args:
            occupied_indices(list): A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
            active_indices(list): A list of spatial orbital indices indicating
                which orbitals should be considered active.

        Returns:
            molecular_hamiltonian: An instance of the MolecularOperator class.

        Note:
            The indexing convention used is that even indices correspond to
            spin-up (alpha) modes and odd indices correspond to spin-down
            (beta) modes.
        """
        # Get active space integrals.
        if occupied_indices is None and active_indices is None:
            one_body_integrals, two_body_integrals = self.get_integrals()
            constant = self.nuclear_repulsion
        else:
            core_adjustment, one_body_integrals, two_body_integrals = self. \
                get_active_space_integrals(occupied_indices, active_indices)
            constant = self.nuclear_repulsion + core_adjustment

        n_qubits = 2 * one_body_integrals.shape[0]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
        two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                             n_qubits, n_qubits))
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[
                    p, q]
                one_body_coefficients[2 * p + 1, 2 *
                                      q + 1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):

                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1,
                                              2 * r + 1, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q,
                                              2 * r, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q,
                                              2 * r, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q + 1,
                                              2 * r + 1, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

        # Truncate.
        one_body_coefficients[
            numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
        two_body_coefficients[
            numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

        # Cast to InteractionOperator class and return.
        molecular_hamiltonian = InteractionOperator(
            constant, one_body_coefficients, two_body_coefficients)

        return molecular_hamiltonian

    def get_molecular_rdm(self, use_fci=False):
        """Method to return 1-RDM and 2-RDMs from CISD or FCI.

        Args:
            use_fci: Boolean indicating whether to use RDM from FCI
                calculation.

        Returns:
            rdm: An instance of the MolecularRDM class.

        Raises:
            MisissingCalculationError: If the CI calculation has not been
                performed.
        """
        # Make sure requested RDM has been computed and load.
        if use_fci:
            if self.fci_energy is None:
                raise MissingCalculationError(
                    'Missing FCI RDM in {}'.format(self.filename) +
                    'Run FCI calculation before loading FCI RDMs.')
            else:
                one_rdm = self.fci_one_rdm
                two_rdm = self.fci_two_rdm
        else:
            if self.cisd_energy is None:
                raise MissingCalculationError(
                    'Missing CISD RDM in {}'.format(self.filename) +
                    'Run CISD calculation before loading CISD RDMs.')
            else:
                one_rdm = self.cisd_one_rdm
                two_rdm = self.cisd_two_rdm

        # Truncate.
        one_rdm[numpy.absolute(one_rdm) < EQ_TOLERANCE] = 0.
        two_rdm[numpy.absolute(two_rdm) < EQ_TOLERANCE] = 0.

        # Cast to InteractionRDM class.
        rdm = InteractionRDM(one_rdm, two_rdm)
        return rdm


def load_molecular_hamiltonian(
        geometry,
        basis,
        multiplicity,
        description,
        n_active_electrons=None,
        n_active_orbitals=None):
    """Attempt to load a molecular Hamiltonian with the given properties.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to
            specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
            Only optional if loading from file.
        multiplicity: An integer giving the spin multiplicity.
        description: A string giving a description.
        n_active_electrons: An optional integer specifying the number of
            electrons desired in the active space.
        n_active_orbitals: An optional integer specifying the number of
            spatial orbitals desired in the active space.

    Returns:
        The Hamiltonian as an InteractionOperator.
    """

    molecule = MolecularData(
            geometry, basis, multiplicity, description=description)
    molecule.load()

    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)
