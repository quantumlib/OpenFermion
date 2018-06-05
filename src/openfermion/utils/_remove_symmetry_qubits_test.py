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

""" Unit tests of remove_symmetry_qubits(). Test that it preserves the
    correct ground state energy, and reduces the number of qubits required
    by two.
"""

import unittest

from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import get_fermion_operator
from openfermion.utils import eigenspectrum, remove_symmetry_qubits


def LiH_sto3g():
    """ Generates the Hamiltonian for LiH in
        the STO-3G basis, at a distance of
        1.45 A.
    """
    geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.45))]
    basis = 'sto3g'
    charge = 0
    multiplicity = 1

    LiH_molecule = MolecularData(geometry, basis, multiplicity, charge, "1.45")
    LiH_molecule.load()

    mol_hamil = LiH_molecule.get_molecular_hamiltonian()
    num_electrons = calculated_molecule.n_electrons
    num_orbitals = 2*calculated_molecule.n_orbitals
    ferm_hamil = get_fermion_operator(mol_hamil)

    return ferm_hamil, num_orbitals, num_electrons


def H2_sto3g():
    """ Generates the Hamiltonian for H_2 in
        the STO-3G basis, at a distance of
        0.75 A.
    """
    geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.75))]
    basis = 'sto3g'
    charge = 0
    multiplicity = 1

    H2_molecule = MolecularData(geometry, basis, multiplicity, charge, '0.75')
    H2_molecule.load()

    mol_hamil = H2_molecule.get_molecular_hamiltonian()
    num_electrons = calculated_molecule.n_electrons
    num_orbitals = 2*calculated_molecule.n_orbitals
    ferm_hamil = get_fermion_operator(mol_hamil)

    return ferm_hamil, num_orbitals, num_electrons


def number_of_qubits(qubit_hamiltonian, unreduced_orbitals):
    """ Returns the number of qubits that a
        qubit Hamiltonian acts upon.
    """
    max_orbital = 0
    for i in range(unreduced_orbitals):
        for key, val in qubit_hamiltonian.terms.items():
            if (i, 'X') in key or (i, 'Y') in key or (i, 'Z') in key:
                max_orbital = max_orbital + 1
                break

    return max_orbital


class ReduceSymmetryQubitsTest(unittest.TestCase):

    # Check whether fermionic and reduced qubit Hamiltonians
    # have the same energy.
    def test_energy_reduce_symmetry_qubits(self):
        # Generate the fermionic Hamiltonians,
        # number of orbitals and number of electrons.
        h2_sto_hamil, h2_sto_numorb, h2_sto_numel = H2_sto3g()
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        # Use test function to reduce the qubits.
        h2_sto_qbt = (
            _remove_symmetry_qubits.symmetry_conserving_bravyi_kitaev(
                h2_sto_hamil, h2_sto_numorb, h2_sto_numel))
        lih_sto_qbt = (
            _remove_symmetry_qubits.symmetry_conserving_bravyi_kitaev(
                lih_sto_hamil, lih_sto_numorb, lih_sto_numel))

        self.assertAlmostEqual(
            eigenspectrum(h2_sto_qbt)[0], eigenspectrum(h2_sto_hamil)[0])
        self.assertAlmostEqual(
            eigenspectrum(lih_sto_qbt)[0], eigenspectrum(lih_sto_hamil)[0])

    # Check that the qubit Hamiltonian acts on two fewer qubits.
    def test_orbnum_reduce_symmetry_qubits(self):
        # Generate the fermionic Hamiltonians,
        # number of orbitals and number of electrons.
        h2_sto_hamil, h2_sto_numorb, h2_sto_numel = H2_sto3g()
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        # Use test function to reduce the qubits.
        h2_sto_qbt = (
            _remove_symmetry_qubits.symmetry_conserving_bravyi_kitaev(
                h2_sto_hamil, h2_sto_numorb, h2_sto_numel))
        lih_sto_qbt = (
            _remove_symmetry_qubits.symmetry_conserving_bravyi_kitaev(
                lih_sto_hamil, lih_sto_numorb, lih_sto_numel))

        self.assertEqual(number_of_qubits(h2_sto_qbt, h2_sto_numorb),
                         h2_sto_numorb-2)
        self.assertEqual(number_of_qubits(lih_sto_qbt, lih_sto_numorb),
                         lih_sto_numorb-2)
