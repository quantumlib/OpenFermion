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

import numpy
import unittest

from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import get_fermion_operator
from openfermion.utils import eigenspectrum, freeze_orbitals

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


def LiH_reduced():
    """ Generates the Hamiltonian for LiH in
        the STO-3G basis, at a distance of
        1.45 A in a reduced active space.
    """
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    molecule = MolecularData(geometry, 'sto-3g', 1,
                             description="1.45")
    molecule.load()
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = get_fermion_operator(molecular_hamiltonian)
    num_electrons = molecule.n_electrons
    num_orbitals = 2*molecule.n_orbitals

    one_rdm = molecule.cisd_one_rdm
    diag_rdm, diag_rotation = diagonalise_one_rdm(one_rdm)
    molecular_hamiltonian.rotate_basis(diag_rotation)
    hamiltonian = get_fermion_operator(molecular_hamiltonian)
    occ, virt = restrict_orbitals(diag_rdm)
    hamiltonian = freeze_orbitals(
        hamiltonian, occupied=occ, unoccupied=virt, prune=True)
    # Update the number of electrons and orbitals being considered.
    num_orbitals = num_orbitals - len(occ) - len(virt)
    num_electrons = num_electrons - len(occ)

    return hamiltonian, num_orbitals, num_electrons


def diagonalise_one_rdm(one_rdm):
    """ Diagonalises the 1-RDM and returns
        the unitary rotation which diagonalises it.
    """
    noon_values, diag_transform = numpy.linalg.eig(one_rdm)
    # Get the unitary matrix which diagonalises the 1-RDM, and the diagonal
    # values, which are the natural orbital occupation numbers (NOONs).

    diag_one_rdm = numpy.dot(
        numpy.linalg.inv(diag_transform), numpy.dot(one_rdm, diag_transform))
    # Diagonalise the 1-RDM (=P) by doing U^{-1} P U.

    return diag_one_rdm, diag_transform


def get_2Darray_min(array, tolerance):
    """ Returns the minimum element (and its location) larger than
        a certain tolerance from a 2D array.
    """
    min_element = numpy.amax(array)
    for i in range(len(array)):
        for j in range(len(array[i])):
            element = array[i][j]
            if element < min_element and element >= tolerance:
                min_element = element
                row_position = i
                col_position = j

    return min_element, row_position, col_position


def get_2Darray_max(array):
    """ Returns the maximum element (and its location) larger than
        a certain tolerance from a 2D array.
    """
    max_element = 0
    for i in range(len(array)):
        for j in range(len(array[i])):
            element = array[i][j]
            if element > max_element:
                max_element = element
                row_position = i
                col_position = j

    return max_element, row_position, col_position


def restrict_orbitals(diag_one_rdm):
    """ Returns arrays of the occupied and virtual orbitals to remove
        from the Hamiltonian.
    """
    tolerance = 0.00001
    max_noon, max_row, max_col = get_2Darray_max(diag_one_rdm)
    min_noon, min_row, min_col = get_2Darray_min(diag_one_rdm, tolerance)

    occ_orbs = [2*max_col, (2*max_col+1)]
    virt_orbs = [2*min_col, (2*min_col+1)]
    # Consider the NMOs with the largest NOONs to be occupied, and with the
    # smallest to be unoccupied.

    return occ_orbs, virt_orbs


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

    # Check ValueErrors arise correctly.
    def test_errors_reduce_symmetry_qubits(self):
        # Generate the fermionic Hamiltonians,
        # number of orbitals and number of electrons.
        lih_reduc_hamil, lih_reduc_numorb, lih_reduc_numel = LiH_reduced()
        
        lih_reduc_qbt = (
            symmetry_conserving_bravyi_kitaev(
                lih_reduc_hamil, lih_reduc_numorb, lih_reduc_numel))

        with self.assertRaises(ValueError):
            symmetry_conserving_bravyi_kitaev(
                0, lih_reduc_numorb, lih_reduc_numel)
        with self.assertRaises(ValueError):
            symmetry_conserving_bravyi_kitaev(
                lih_reduc_hamil, 1.5, lih_reduc_numel)
        with self.assertRaises(ValueError):
            symmetry_conserving_bravyi_kitaev(
                lih_reduc_hamil, lih_reduc_numorb, 3.6)
