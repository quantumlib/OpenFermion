import os
import numpy as np
import openfermion as of
from openfermion.config import DATA_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.ops import InteractionOperator
from openfermion.utils._reduced_hamiltonian import make_reduced_hamiltonian


def test_mrd_return_type():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)

    assert isinstance(reduced_ham, InteractionOperator)


def test_constant_one_body():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)

    assert np.isclose(reduced_ham.constant, molecule.nuclear_repulsion)
    assert np.allclose(reduced_ham.one_body_tensor, 0)


def test_fci_energy():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    np_ham = of.get_number_preserving_sparse_operator(
        of.get_fermion_operator(reduced_ham),
        molecule.n_qubits,
        num_electrons=molecule.n_electrons,
        spin_preserving=True)

    w, _ = np.linalg.eigh(np_ham.toarray())
    assert np.isclose(molecule.fci_energy, w[0])

    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    np_ham = of.get_number_preserving_sparse_operator(
        of.get_fermion_operator(reduced_ham),
        molecule.n_qubits,
        num_electrons=molecule.n_electrons,
        spin_preserving=True)

    w, _ = np.linalg.eigh(np_ham.toarray())
    assert np.isclose(molecule.fci_energy, w[0])
