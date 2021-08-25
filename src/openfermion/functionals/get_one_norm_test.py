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
"""Tests for get_one_norm."""

import os
import pytest
from openfermion import (get_one_norm_mol, get_one_norm_mol_woconst,
                         get_one_norm_int, get_one_norm_int_woconst,
                         MolecularData, jordan_wigner)
from openfermion.config import DATA_DIRECTORY

filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
molecule = MolecularData(filename=filename)
molecular_hamiltonian = molecule.get_molecular_hamiltonian()
qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)


def test_one_norm_from_molecule():
    assert qubit_hamiltonian.induced_norm() == pytest.approx(
        get_one_norm_mol(molecule))


def test_one_norm_from_ints():
    assert qubit_hamiltonian.induced_norm() == pytest.approx(
        get_one_norm_int(
            molecule.nuclear_repulsion,
            molecule.one_body_integrals,
            molecule.two_body_integrals,
        ))


def test_one_norm_woconst():
    one_norm_woconst = (qubit_hamiltonian.induced_norm() -
                        abs(qubit_hamiltonian.constant))
    assert one_norm_woconst == pytest.approx(get_one_norm_mol_woconst(molecule))
    assert one_norm_woconst == pytest.approx(
        get_one_norm_int_woconst(molecule.one_body_integrals,
                                 molecule.two_body_integrals))
