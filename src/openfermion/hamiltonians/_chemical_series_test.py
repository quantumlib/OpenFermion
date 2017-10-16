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

"""Tests for chemical_series."""
from __future__ import absolute_import

import numpy
import os
import unittest

from openfermion.hamiltonians import (make_atom,
                                      make_atomic_lattice,
                                      make_atomic_ring,
                                      periodic_table)
from openfermion.hamiltonians._chemical_series import MolecularLatticeError
from openfermion.hamiltonians._molecular_data import periodic_polarization


class ChemicalSeries(unittest.TestCase):

    def test_make_atomic_ring(self):
        spacing = 1.
        basis = 'sto-3g'
        for n_atoms in range(2, 10):
            molecule = make_atomic_ring(n_atoms, spacing, basis)

            # Check that ring is centered.
            vector_that_should_sum_to_zero = 0.
            for atom in molecule.geometry:
                for coordinate in atom[1]:
                    vector_that_should_sum_to_zero += coordinate
            self.assertAlmostEqual(vector_that_should_sum_to_zero, 0.)

            # Check that the spacing between the atoms is correct.
            for atom_index in range(n_atoms):
                if atom_index:
                    atom_b = molecule.geometry[atom_index]
                    coords_b = atom_b[1]
                    atom_a = molecule.geometry[atom_index - 1]
                    coords_a = atom_a[1]
                    observed_spacing = numpy.sqrt(numpy.square(
                        coords_b[0] - coords_a[0]) + numpy.square(
                        coords_b[1] - coords_a[1]) + numpy.square(
                        coords_b[2] - coords_a[2]))
                    self.assertAlmostEqual(observed_spacing, spacing)

    def test_make_atomic_lattice_1d(self):
        spacing = 1.7
        basis = 'sto-3g'
        atom_type = 'H'
        for n_atoms in range(2, 10):
            molecule = make_atomic_lattice(n_atoms, 1, 1,
                                           spacing, basis, atom_type)

            # Check that the spacing between the atoms is correct.
            for atom_index in range(n_atoms):
                if atom_index:
                    atom_b = molecule.geometry[atom_index]
                    coords_b = atom_b[1]
                    atom_a = molecule.geometry[atom_index - 1]
                    coords_a = atom_a[1]
                    self.assertAlmostEqual(coords_b[0] - coords_a[0], spacing)
                    self.assertAlmostEqual(coords_b[1] - coords_a[1], 0)
                    self.assertAlmostEqual(coords_b[2] - coords_a[2], 0)

    def test_make_atomic_lattice_2d(self):
        spacing = 1.7
        basis = 'sto-3g'
        atom_type = 'H'
        atom_dim = 7
        molecule = make_atomic_lattice(atom_dim, atom_dim, 1,
                                       spacing, basis, atom_type)

        # Check that the spacing between the atoms is correct.
        for atom in range(atom_dim ** 2):
            coords = molecule.geometry[atom][1]

            # Check y-coord.
            grid_y = atom % atom_dim
            self.assertAlmostEqual(coords[1], spacing * grid_y)

            # Check x-coord.
            grid_x = atom // atom_dim
            self.assertAlmostEqual(coords[0], spacing * grid_x)

    def test_make_atomic_lattice_3d(self):
        spacing = 1.7
        basis = 'sto-3g'
        atom_type = 'H'
        atom_dim = 4
        molecule = make_atomic_lattice(atom_dim, atom_dim, atom_dim,
                                       spacing, basis, atom_type)

        # Check that the spacing between the atoms is correct.
        for atom in range(atom_dim ** 3):
            coords = molecule.geometry[atom][1]

            # Check z-coord.
            grid_z = atom % atom_dim
            self.assertAlmostEqual(coords[2], spacing * grid_z)

            # Check y-coord.
            grid_y = (atom // atom_dim) % atom_dim
            self.assertAlmostEqual(coords[1], spacing * grid_y)

            # Check x-coord.
            grid_x = atom // atom_dim ** 2
            self.assertAlmostEqual(coords[0], spacing * grid_x)

    def test_make_atomic_lattice_0d_raise_error(self):
        spacing = 1.7
        basis = 'sto-3g'
        atom_type = 'H'
        atom_dim = 0
        with self.assertRaises(MolecularLatticeError):
            make_atomic_lattice(atom_dim, atom_dim, atom_dim,
                                spacing, basis, atom_type)

    def test_make_atom(self):
        basis = 'sto-3g'
        largest_atom = 30
        for n_electrons in range(1, largest_atom):
            atom_name = periodic_table[n_electrons]
            atom = make_atom(atom_name, basis)
            expected_spin = periodic_polarization[n_electrons] / 2.
            expected_multiplicity = int(2 * expected_spin + 1)
            self.assertAlmostEqual(expected_multiplicity, atom.multiplicity)
