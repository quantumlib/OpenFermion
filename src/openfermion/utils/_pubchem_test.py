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

"""Tests for _pubchem.py."""

import unittest
import numpy
import pytest

from openfermion.utils import (geometry_from_pubchem,
                               module_importable)


using_pubchempy = pytest.mark.skipif(module_importable('pubchempy') is False,
                                     reason='Not detecting `pubchempy`.')


@using_pubchempy
class OpenFermionPubChemTest(unittest.TestCase):

    def test_water(self):
        water_geometry = geometry_from_pubchem('water')
        self.water_natoms = len(water_geometry)
        self.water_atoms = [water_atom[0] for water_atom in water_geometry]
        water_oxygen_index = self.water_atoms.index('O')
        water_oxygen = water_geometry.pop(water_oxygen_index)
        water_oxygen_coordinate = numpy.array(water_oxygen[1])
        water_hydrogen1_coordinate = numpy.array(water_geometry[0][1])
        water_hydrogen2_coordinate = numpy.array(water_geometry[1][1])
        water_oxygen_hydrogen1 = \
            water_hydrogen1_coordinate - water_oxygen_coordinate
        water_oxygen_hydrogen2 = \
            water_hydrogen2_coordinate - water_oxygen_coordinate

        self.water_bond_length_1 = numpy.linalg.norm(water_oxygen_hydrogen1)
        self.water_bond_length_2 = numpy.linalg.norm(water_oxygen_hydrogen2)
        self.water_bond_angle = \
            numpy.arccos(numpy.dot(water_oxygen_hydrogen1,
                                   water_oxygen_hydrogen2 /
                                   (numpy.linalg.norm(water_oxygen_hydrogen1) *
                                    numpy.linalg.norm(water_oxygen_hydrogen2))))

        water_natoms = 3
        self.assertEqual(water_natoms, self.water_natoms)

        self.assertAlmostEqual(self.water_bond_length_1,
                               self.water_bond_length_2, places=4)
        water_bond_length_low = 0.9
        water_bond_length_high = 1.1
        self.assertTrue(water_bond_length_low <= self.water_bond_length_1)
        self.assertTrue(water_bond_length_high >= self.water_bond_length_1)

        water_bond_angle_low = 100. / 360 * 2 * numpy.pi
        water_bond_angle_high = 110. / 360 * 2 * numpy.pi
        self.assertTrue(water_bond_angle_low <= self.water_bond_angle)
        self.assertTrue(water_bond_angle_high >= self.water_bond_angle)

    def test_helium(self):
        helium_geometry = geometry_from_pubchem('helium')
        self.helium_natoms = len(helium_geometry)

        helium_natoms = 1
        self.assertEqual(helium_natoms, self.helium_natoms)

    def test_none(self):
        none_geometry = geometry_from_pubchem('none')

        self.assertIsNone(none_geometry)

    def test_water_2d(self):
        water_geometry = geometry_from_pubchem('water', structure='2d')
        self.water_natoms = len(water_geometry)

        water_natoms = 3
        self.assertEqual(water_natoms, self.water_natoms)

        self.oxygen_z_1 = water_geometry[0][1][2]
        self.oxygen_z_2 = water_geometry[1][1][2]
        z = 0
        self.assertEqual(z, self.oxygen_z_1)
        self.assertEqual(z, self.oxygen_z_2)

        with pytest.raises(ValueError,
                           match='Incorrect value for the argument structure'):
            _ = geometry_from_pubchem('water', structure='foo')
