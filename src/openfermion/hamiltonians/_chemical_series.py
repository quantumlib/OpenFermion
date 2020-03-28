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

"""Functions to create chemical series data sets."""

import numpy

from openfermion.hamiltonians._molecular_data import (MolecularData,
                                                      periodic_hash_table,
                                                      periodic_polarization)


# Define error objects which inherit from Exception.
class MolecularLatticeError(Exception):
    pass


def make_atomic_ring(n_atoms, spacing, basis,
                     atom_type='H', charge=0, filename=''):
    """Function to create atomic rings with n_atoms.

    Note that basic geometry suggests that for spacing L between atoms
    the radius of the ring should be L / (2 * cos (pi / 2 - theta / 2))

    Args:
        n_atoms: Integer, the number of atoms in the ring.
        spacing: The spacing between atoms in the ring in Angstroms.
        basis: The basis in which to perform the calculation.
        atom_type: String, the atomic symbol of the element in the ring.
            this defaults to 'H' for Hydrogen.
        charge: An integer giving the total molecular charge. Defaults to 0.
        filename: An optional string to give a filename for the molecule.

    Returns:
        molecule: A an instance of the MolecularData class.
    """
    # Make geometry.
    geometry = []
    theta = 2. * numpy.pi / float(n_atoms)
    radius = spacing / (2. * numpy.cos(numpy.pi / 2. - theta / 2.))
    for atom in range(n_atoms):
        x_coord = radius * numpy.cos(atom * theta)
        y_coord = radius * numpy.sin(atom * theta)
        geometry += [(atom_type, (x_coord, y_coord, 0.))]

    # Set multiplicity.
    n_electrons = n_atoms * periodic_hash_table[atom_type]
    n_electrons -= charge
    if (n_electrons % 2):
        multiplicity = 2
    else:
        multiplicity = 1

    # Create molecule and return.
    description = 'ring_{}'.format(spacing)
    molecule = MolecularData(geometry,
                             basis,
                             multiplicity,
                             charge,
                             description,
                             filename)
    return molecule


def make_atomic_lattice(nx_atoms, ny_atoms, nz_atoms, spacing, basis,
                        atom_type='H', charge=0, filename=''):
    """Function to create atomic lattice with n_atoms.

    Args:
        nx_atoms: Integer, the length of lattice (in number of atoms).
        ny_atoms: Integer, the width of lattice (in number of atoms).
        nz_atoms: Integer, the depth of lattice (in number of atoms).
        spacing: The spacing between atoms in the lattice in Angstroms.
        basis: The basis in which to perform the calculation.
        atom_type: String, the atomic symbol of the element in the ring.
            this defaults to 'H' for Hydrogen.
        charge: An integer giving the total molecular charge. Defaults to 0.
        filename: An optional string to give a filename for the molecule.

    Returns:
        molecule: A an instance of the MolecularData class.

    Raises:
        MolecularLatticeError: If lattice specification is invalid.
    """
    # Make geometry.
    geometry = []
    for x_dimension in range(nx_atoms):
        for y_dimension in range(ny_atoms):
            for z_dimension in range(nz_atoms):
                x_coord = spacing * x_dimension
                y_coord = spacing * y_dimension
                z_coord = spacing * z_dimension
                geometry += [(atom_type, (x_coord, y_coord, z_coord))]

    # Set multiplicity.
    n_atoms = nx_atoms * ny_atoms * nz_atoms
    n_electrons = n_atoms * periodic_hash_table[atom_type]
    n_electrons -= charge
    if (n_electrons % 2):
        multiplicity = 2
    else:
        multiplicity = 1

    # Name molecule.
    dimensions = bool(nx_atoms > 1) + bool(ny_atoms > 1) + bool(nz_atoms > 1)
    if dimensions == 1:
        description = 'linear_{}'.format(spacing)
    elif dimensions == 2:
        description = 'planar_{}'.format(spacing)
    elif dimensions == 3:
        description = 'cubic_{}'.format(spacing)
    else:
        raise MolecularLatticeError('Invalid lattice dimensions.')

    # Create molecule and return.
    molecule = MolecularData(geometry,
                             basis,
                             multiplicity,
                             charge,
                             description,
                             filename)
    return molecule


def make_atom(atom_type, basis, filename=''):
    """Prepare a molecular data instance for a single element.

    Args:
        atom_type: Float giving atomic symbol.
        basis: The basis in which to perform the calculation.

    Returns:
        atom: An instance of the MolecularData class.
    """
    geometry = [(atom_type, (0., 0., 0.))]
    atomic_number = periodic_hash_table[atom_type]
    spin = periodic_polarization[atomic_number] / 2.
    multiplicity = int(2 * spin + 1)
    atom = MolecularData(geometry,
                         basis,
                         multiplicity,
                         filename=filename)
    return atom
