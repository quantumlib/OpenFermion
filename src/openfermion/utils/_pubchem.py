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


def geometry_from_pubchem(name):
    """Function to extract geometry using the molecule's name from the PubChem
    database.

    Args:
        name: a string giving the molecule's name as required by the PubChem
            database.

    Returns:
        geometry: a list of tuples giving the coordinates of each atom with
        distances in Angstrom.
    """
    import pubchempy

    pubchempy_2d_molecule = pubchempy.get_compounds(name, 'name',
                                                    record_type='2d')

    # Check if 2-D geometry is available. If not then no geometry is.
    if not pubchempy_2d_molecule:
        print('Unable to find molecule in the PubChem database.')
        return None

    # Ideally get the 3-D geometry if available.
    pubchempy_3d_molecule = pubchempy.get_compounds(name, 'name',
                                                    record_type='3d')

    if pubchempy_3d_molecule:
        pubchempy_geometry = \
            pubchempy_3d_molecule[0].to_dict(properties=['atoms'])['atoms']
        geometry = [(atom['element'], (atom['x'], atom['y'], atom['z']))
                    for atom in pubchempy_geometry]
        return geometry

    # If the 3-D geometry isn't available, get the 2-D geometry instead.
    pubchempy_geometry = \
        pubchempy_2d_molecule[0].to_dict(properties=['atoms'])['atoms']
    geometry = [(atom['element'], (atom['x'], atom['y'], 0))
                for atom in pubchempy_geometry]

    return geometry
