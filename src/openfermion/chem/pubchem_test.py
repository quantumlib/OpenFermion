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

"""Tests for pubchem.py."""

import numpy
import pytest
import pubchempy

from openfermion.chem.pubchem import geometry_from_pubchem
from openfermion.testing.testing_utils import module_importable


class MockCompound:
    def __init__(self, atoms):
        self._atoms = atoms

    def to_dict(self, properties):
        return {'atoms': self._atoms}


def mock_get_compounds(name, searchtype, record_type='2d'):
    match (name, record_type):
        case ('water', '3d'):
            return [
                MockCompound(
                    [
                        {'aid': 1, 'number': 8, 'element': 'O', 'y': 0, 'z': 0, 'x': 0},
                        {
                            'aid': 2,
                            'number': 1,
                            'element': 'H',
                            'y': 0.8929,
                            'z': 0.2544,
                            'x': 0.2774,
                        },
                        {
                            'aid': 3,
                            'number': 1,
                            'element': 'H',
                            'y': -0.2383,
                            'z': -0.7169,
                            'x': 0.6068,
                        },
                    ]
                )
            ]
        case ('water', '2d'):
            return [
                MockCompound(
                    [
                        {'aid': 1, 'number': 8, 'element': 'O', 'y': -0.155, 'x': 2.5369},
                        {'aid': 2, 'number': 1, 'element': 'H', 'y': 0.155, 'x': 3.0739},
                        {'aid': 3, 'number': 1, 'element': 'H', 'y': 0.155, 'x': 2},
                    ]
                )
            ]
        case ('helium', '2d'):
            return [MockCompound([{'aid': 1, 'number': 2, 'element': 'He', 'y': 0, 'x': 2}])]
        case _:
            return []


using_pubchempy = pytest.mark.skipif(
    module_importable('pubchempy') is False, reason='Not detecting `pubchempy`.'
)


@using_pubchempy
class TestOpenFermionPubChem:

    def test_water(self, monkeypatch):
        monkeypatch.setattr(pubchempy, 'get_compounds', mock_get_compounds)
        water_geometry = geometry_from_pubchem('water')
        water_natoms_actual = len(water_geometry)
        water_atoms = [water_atom[0] for water_atom in water_geometry]
        water_oxygen_index = water_atoms.index('O')
        water_oxygen = water_geometry.pop(water_oxygen_index)
        water_oxygen_coordinate = numpy.array(water_oxygen[1])
        water_hydrogen1_coordinate = numpy.array(water_geometry[0][1])
        water_hydrogen2_coordinate = numpy.array(water_geometry[1][1])
        water_oxygen_hydrogen1 = water_hydrogen1_coordinate - water_oxygen_coordinate
        water_oxygen_hydrogen2 = water_hydrogen2_coordinate - water_oxygen_coordinate

        water_bond_length_1 = numpy.linalg.norm(water_oxygen_hydrogen1)
        water_bond_length_2 = numpy.linalg.norm(water_oxygen_hydrogen2)
        water_bond_angle = numpy.arccos(
            numpy.dot(
                water_oxygen_hydrogen1,
                water_oxygen_hydrogen2
                / (
                    numpy.linalg.norm(water_oxygen_hydrogen1)
                    * numpy.linalg.norm(water_oxygen_hydrogen2)
                ),
            )
        )

        water_natoms = 3
        assert water_natoms == water_natoms_actual

        assert water_bond_length_1 == pytest.approx(water_bond_length_2, abs=1e-4)
        water_bond_length_low = 0.9
        water_bond_length_high = 1.1
        assert water_bond_length_low <= water_bond_length_1
        assert water_bond_length_high >= water_bond_length_1

        water_bond_angle_low = 100.0 / 360 * 2 * numpy.pi
        water_bond_angle_high = 110.0 / 360 * 2 * numpy.pi
        assert water_bond_angle_low <= water_bond_angle
        assert water_bond_angle_high >= water_bond_angle

    def test_helium(self, monkeypatch):
        monkeypatch.setattr(pubchempy, 'get_compounds', mock_get_compounds)
        helium_geometry = geometry_from_pubchem('helium')
        helium_natoms_actual = len(helium_geometry)

        helium_natoms = 1
        assert helium_natoms == helium_natoms_actual

    def test_none(self, monkeypatch):
        monkeypatch.setattr(pubchempy, 'get_compounds', mock_get_compounds)
        none_geometry = geometry_from_pubchem('none')

        assert none_geometry is None

    def test_water_2d(self, monkeypatch):
        monkeypatch.setattr(pubchempy, 'get_compounds', mock_get_compounds)
        water_geometry = geometry_from_pubchem('water', structure='2d')
        water_natoms_actual = len(water_geometry)

        water_natoms = 3
        assert water_natoms == water_natoms_actual

        oxygen_z_1 = water_geometry[0][1][2]
        oxygen_z_2 = water_geometry[1][1][2]
        z = 0
        assert z == oxygen_z_1
        assert z == oxygen_z_2

        with pytest.raises(ValueError, match='Incorrect value for the argument structure'):
            _ = geometry_from_pubchem('water', structure='foo')

    @pytest.mark.flaky(retries=3, delay=20, backoff=2, only_on=[pubchempy.ServerBusyError])
    def test_geometry_from_pubchem_live_api(self):
        water_geometry = geometry_from_pubchem('water')
        assert len(water_geometry) == 3
