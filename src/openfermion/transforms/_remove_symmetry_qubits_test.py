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
from openfermion.utils import eigenspectrum

from openfermion.transforms._remove_symmetry_qubits import (
    symmetry_conserving_bravyi_kitaev, edit_hamiltonian_for_spin)


def LiH_sto3g():
    """ Generates the Hamiltonian for LiH in
        the STO-3G basis, at a distance of
        1.45 A.
    """
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    molecule = MolecularData(geometry, 'sto-3g', 1,
                             description="1.45")
    molecule.load()
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = get_fermion_operator(molecular_hamiltonian)
    num_electrons = molecule.n_electrons
    num_orbitals = 2*molecule.n_orbitals
    
    return hamiltonian, num_orbitals, num_electrons


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
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        # Use test function to reduce the qubits.
        lih_sto_qbt = (
            symmetry_conserving_bravyi_kitaev(
                lih_sto_hamil, lih_sto_numorb, lih_sto_numel))

        self.assertAlmostEqual(
            eigenspectrum(lih_sto_qbt)[0], eigenspectrum(lih_sto_hamil)[0])

    # Check that the qubit Hamiltonian acts on two fewer qubits.
    def test_orbnum_reduce_symmetry_qubits(self):
        # Generate the fermionic Hamiltonians,
        # number of orbitals and number of electrons.
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        # Use test function to reduce the qubits.
        lih_sto_qbt = (
            symmetry_conserving_bravyi_kitaev(
                lih_sto_hamil, lih_sto_numorb, lih_sto_numel))

        self.assertEqual(number_of_qubits(lih_sto_qbt, lih_sto_numorb),
                         lih_sto_numorb-2)
