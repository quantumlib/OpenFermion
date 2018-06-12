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

from openfermion.hamiltonians import fermi_hubbard, MolecularData
from openfermion.transforms import (
    get_fermion_operator, get_sparse_operator)
from openfermion.utils import (eigenspectrum,
                               jw_get_ground_state_at_particle_number)

from openfermion.transforms._remove_symmetry_qubits import (
        symmetry_conserving_bravyi_kitaev)


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


def set_1D_hubbard(x_dim):
    """ Returns a 1D Fermi-Hubbard Hamiltonian.
    """
    y_dim = 1
    tunneling = 1.0
    coulomb = 1.0
    magnetic_field = 0.0
    chemical_potential = 0.0
    periodic = False
    spinless = False

    num_orbitals = 2*x_dim*y_dim

    hubbard_model = fermi_hubbard(
        x_dim, y_dim, tunneling, coulomb, chemical_potential,
        magnetic_field, periodic, spinless)

    return hubbard_model, num_orbitals


def number_of_qubits(qubit_hamiltonian, unreduced_orbitals):
    """ Returns the number of qubits that a
        qubit Hamiltonian acts upon.
    """
    max_orbital = 0
    for i in range(unreduced_orbitals):
        for term in qubit_hamiltonian.terms:
            if (i, 'X') in term or (i, 'Y') in term or (i, 'Z') in term:
                max_orbital = max_orbital + 1
                break

    return max_orbital


class ReduceSymmetryQubitsTest(unittest.TestCase):

    # Check whether fermionic and reduced qubit Hamiltonians
    # have the same energy for LiH
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

    # Check that the qubit Hamiltonian acts on two fewer qubits
    # for LiH.
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

    # Check ValueErrors arise correctly.
    def test_errors_reduce_symmetry_qubits(self):
        # Generate the fermionic Hamiltonians,
        # number of orbitals and number of electrons.
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        with self.assertRaises(ValueError):
            symmetry_conserving_bravyi_kitaev(
                0, lih_sto_numorb, lih_sto_numel)
        with self.assertRaises(ValueError):
            symmetry_conserving_bravyi_kitaev(
                lih_sto_hamil, 1.5, lih_sto_numel)
        with self.assertRaises(ValueError):
            symmetry_conserving_bravyi_kitaev(
                lih_sto_hamil, lih_sto_numorb, 3.6)

    # Check energy is the same for Fermi-Hubbard model.
    def test_hubbard_reduce_symmetry_qubits(self):
        for i in range(4):
            n_sites = i+2
            n_ferm = n_sites
            hub_hamil, n_orb = set_1D_hubbard(n_sites)

            # Use test function to reduce the qubits.
            hub_qbt = (
                symmetry_conserving_bravyi_kitaev(
                    hub_hamil, n_orb, n_ferm))

            sparse_op = get_sparse_operator(hub_hamil)
            ground_energy, _ = jw_get_ground_state_at_particle_number(
                    sparse_op, n_ferm)

            self.assertAlmostEqual(
                eigenspectrum(hub_qbt)[0], ground_energy)
