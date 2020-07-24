import os
import numpy
from openfermion.config import DATA_DIRECTORY
from openfermion.chem import MolecularData
from openfermion.ops.representations import InteractionOperator
from openfermion.chem.reduced_hamiltonian import make_reduced_hamiltonian
from openfermion.linalg.sparse_tools import (
    get_number_preserving_sparse_operator)
from openfermion.transforms.opconversions import get_fermion_operator


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

    assert numpy.isclose(reduced_ham.constant, molecule.nuclear_repulsion)
    assert numpy.allclose(reduced_ham.one_body_tensor, 0)


def test_fci_energy():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    numpy_ham = get_number_preserving_sparse_operator(
        get_fermion_operator(reduced_ham),
        molecule.n_qubits,
        num_electrons=molecule.n_electrons,
        spin_preserving=True)

    w, _ = numpy.linalg.eigh(numpy_ham.toarray())
    assert numpy.isclose(molecule.fci_energy, w[0])

    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    numpy_ham = get_number_preserving_sparse_operator(
        get_fermion_operator(reduced_ham),
        molecule.n_qubits,
        num_electrons=molecule.n_electrons,
        spin_preserving=True)

    w, _ = numpy.linalg.eigh(numpy_ham.toarray())
    assert numpy.isclose(molecule.fci_energy, w[0])
