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

"""This module extracts and parses geometry data from the PubChem database into OpenFermion."""
import psi4
import pubchempy as pcp
from openfermion.hamiltonians import MolecularData


def parser(geom, n_atoms, s):
    """Function to parse sdf (s == 0) or psi4 (s == 1) geometry into OpenFermion geometry.

    Args:
        geom: a string specifying the sdf or psi4 geometry of the molecule.

        n_atoms: the number of atoms of the molecule.

        s: set to 0 or 1 depending on if geom is sdf or psi4.

    Returns:
        opf_geom: a list containing the geometry of the molecule in the format required
            to create a OpenFermion MolecularData instance.
    """
    s = int(s)
    opf_geom = []
    
    if s == 0:
        vec = geom.split('\n')[2:2+n_atoms]
    
    else:
        assert s == 1
        vec = geom.split('\n')[4:4+n_atoms]
        
    for i in range(len(vec)):
        x = vec[i].split()[:4]
        if s == 0:
            tup = tuple([float(x) for x in x[1:]])   
            atom = x[0].lower().capitalize()
            opf_geom.append((atom, tup))
        else:
            assert s == 1
            tup = tuple([float(x) for x in x[:3]]) 
            atom = x[3].lower().capitalize()
            opf_geom.append((atom, tup))
        
    return opf_geom


def extract(name_):
    """Function to create geometry from the molecule's name

    Args:
        name_: a string giving the molecule's name as required by the PubChem database.
            This is quite flexible, e.g.: 'sugar' or 'glucose'.

    Returns:
        opf_geom: a list containing the geometry of the molecule in the format required
            to create a OpenFermion MolecularData instance.
    """
    try:
        name = pcp.get_compounds(name_, 'name', record_type='3d')[0]
        mlc = psi4.geometry("""
pubchem:{}
""".format(name_))
        mlc = mlc.create_psi4_string_from_molecule()
        n_atoms = len(name.atoms)
        opf_geom = parser(mlc, n_atoms, 0)
        print('3D compound found.')

    except IndexError: 
        try:
            name = pcp.get_compounds(name_, 'name', record_type='2d')[0]
            n_atoms = len(name.atoms)
            mlc = pcp.get_sdf(name.cid)
            opf_geom = parser(mlc, n_atoms, 1)
            print('2D compound found.')
        except IndexError:
            opf_geom = 0
            print("The chemical substance is not found in the PubChem database.")

    return opf_geom


if __name__ == "__main__":
    name = 'sugar'
    basis = 'sto-3g'
    opf_geom = extract(name)
    print(name, 'has coordinates: \n', opf_geom)
    multiplicity = 1
    charge = 0
    molecule = MolecularData(opf_geom, basis, multiplicity, description='first_molecule')