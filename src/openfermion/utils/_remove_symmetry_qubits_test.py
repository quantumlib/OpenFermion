""" Unit tests of remove_symmetry_qubits().
    Test that it preserves the correct
    ground state energy, and reduces the number
    of qubits required by 2.
"""

import unittest

import openfermion
import openfermionpyscf
from openfermion.hamiltonians import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.utils import eigenspectrum
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import get_fermion_operator

import _remove_symmetry_qubits





def LiH_sto3g():
    """ Generates the Hamiltonian for LiH in
        the STO-3G basis, at a distance of
        1.45 A.
    """

    geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.45))]
    basis = 'sto3g'
    charge = 0
    multiplicity = 1

    LiH_molecule = MolecularData(geometry, basis, multiplicity, charge, 'LiH')
    calculated_molecule = run_pyscf(LiH_molecule)

    num_electrons = calculated_molecule.n_electrons
    num_orbitals = 2*calculated_molecule.n_orbitals
    mol_hamil = calculated_molecule.get_molecular_hamiltonian()
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

    H2_molecule = MolecularData(geometry, basis, multiplicity, charge, 'H2')
    calculated_molecule = run_pyscf(H2_molecule)

    num_electrons = calculated_molecule.n_electrons
    num_orbitals = 2*calculated_molecule.n_orbitals
    mol_hamil = calculated_molecule.get_molecular_hamiltonian()
    ferm_hamil = get_fermion_operator(mol_hamil)

    return ferm_hamil, num_orbitals, num_electrons


def H2_631g():
    """ Generates the Hamiltonian for H_2 in
        the 6-31G basis, at a distance of
        1.00 A.
    """

    geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.00))]
    basis = '6-31g'
    charge = 0
    multiplicity = 1

    H2_molecule = MolecularData(geometry, basis, multiplicity, charge, 'H2')
    calculated_molecule = run_pyscf(H2_molecule)

    num_electrons = calculated_molecule.n_electrons
    num_orbitals = 2*calculated_molecule.n_orbitals
    mol_hamil = calculated_molecule.get_molecular_hamiltonian()
    ferm_hamil = get_fermion_operator(mol_hamil)

    return ferm_hamil, num_orbitals, num_electrons


def number_of_qubits(qubit_hamiltonian, unreduced_orbitals):
    """ Returns the number of qubits that a
        qubit Hamiltonian acts upon.
    """

    max_orbital = 0
    for i in range(unreduced_orbitals):
        for key, val in qubit_hamiltonian.terms.items():
            if ((i, 'X') in key or
                (i, 'Y') in key or
                (i, 'Z') in key):
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
        h2_631_hamil, h2_631_numorb, h2_631_numel = H2_631g()
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        # Use test function to reduce the qubits.
        h2_sto_qbt = _remove_symmetry_qubits.remove_symmetry_qubits(h2_sto_hamil,
                                                                 h2_sto_numorb,
                                                                 h2_sto_numel)
        h2_631_qbt = _remove_symmetry_qubits.remove_symmetry_qubits(h2_631_hamil,
                                                                 h2_631_numorb,
                                                                 h2_631_numel)
        lih_sto_qbt = _remove_symmetry_qubits.remove_symmetry_qubits(lih_sto_hamil,
                                                                  lih_sto_numorb,
                                                                  lih_sto_numel)
        
        self.assertAlmostEqual(eigenspectrum(h2_sto_qbt)[0],
                         eigenspectrum(h2_sto_hamil)[0])

        self.assertAlmostEqual(eigenspectrum(h2_631_qbt)[0],
                         eigenspectrum(h2_631_hamil)[0])

        self.assertAlmostEqual(eigenspectrum(lih_sto_qbt)[0],
                         eigenspectrum(lih_sto_hamil)[0])




    # Check that the qubit Hamiltonian acts on two fewer qubits.
    def test_orbnum_reduce_symmetry_qubits(self):
        # Generate the fermionic Hamiltonians,
        # number of orbitals and number of electrons.
        h2_sto_hamil, h2_sto_numorb, h2_sto_numel = H2_sto3g()
        h2_631_hamil, h2_631_numorb, h2_631_numel = H2_631g()
        lih_sto_hamil, lih_sto_numorb, lih_sto_numel = LiH_sto3g()

        # Use test function to reduce the qubits.
        h2_sto_qbt = _remove_symmetry_qubits.remove_symmetry_qubits(h2_sto_hamil,
                                                                 h2_sto_numorb,
                                                                 h2_sto_numel)
        h2_631_qbt = _remove_symmetry_qubits.remove_symmetry_qubits(h2_631_hamil,
                                                                 h2_631_numorb,
                                                                 h2_631_numel)
        lih_sto_qbt = _remove_symmetry_qubits.remove_symmetry_qubits(lih_sto_hamil,
                                                                  lih_sto_numorb,
                                                                  lih_sto_numel)
        
        self.assertEqual(number_of_qubits(h2_sto_qbt, h2_sto_numorb),
                         h2_sto_numorb-2)

        self.assertEqual(number_of_qubits(h2_631_qbt, h2_631_numorb),
                         h2_631_numorb-2)

        self.assertEqual(number_of_qubits(lih_sto_qbt, lih_sto_numorb),
                         lih_sto_numorb-2)


if __name__ == '__main__':
    unittest.main()
