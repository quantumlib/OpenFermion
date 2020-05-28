from itertools import product
import os
import numpy as np
import openfermion as of
from openfermion.config import DATA_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.utils import make_reduced_hamiltonian
from openfermion.utils import singlet_erpa, erpa_eom_hamiltonian


def test_h2_rpa():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    hf_opdm = np.diag([1] * molecule.n_electrons + [0] *
                      (molecule.n_qubits - molecule.n_electrons))
    hf_tpdm = 2 * of.wedge(hf_opdm, hf_opdm, (1, 1), (1, 1))

    pos_spectrum, xy_eigvects, basis = singlet_erpa(hf_tpdm,
                                                    reduced_ham.two_body_tensor)
    assert np.isclose(pos_spectrum, 0.92926444)  # pyscf-rpa value
    assert isinstance(xy_eigvects, np.ndarray)
    assert isinstance(basis, dict)


def test_erpa_eom_ham_h2():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    rha_fermion = of.get_fermion_operator(reduced_ham)
    permuted_hijkl = np.einsum('ijlk', reduced_ham.two_body_tensor)
    opdm = np.diag([1] * molecule.n_electrons + [0] *
                   (molecule.n_qubits - molecule.n_electrons))
    tpdm = 2 * of.wedge(opdm, opdm, (1, 1), (1, 1))
    rdms = of.InteractionRDM(opdm, tpdm)
    dim = reduced_ham.one_body_tensor.shape[0] // 2
    full_basis = {}  # erpa basis.  A, B basis in RPA language
    cnt = 0
    for p, q in product(range(dim), repeat=2):
        if p < q:
            full_basis[(p, q)] = cnt
            full_basis[(q, p)] = cnt + dim * (dim - 1) // 2
            cnt += 1
    for rkey in full_basis.keys():
        p, q = rkey
        for ckey in full_basis.keys():
            r, s = ckey
            for sigma, tau in product([0, 1], repeat=2):
                test = erpa_eom_hamiltonian(permuted_hijkl, tpdm, 2 * q + sigma,
                                            2 * p + sigma, 2 * r + tau,
                                            2 * s + tau).real
                qp_op = of.FermionOperator(
                    ((2 * q + sigma, 1), (2 * p + sigma, 0)))
                rs_op = of.FermionOperator(((2 * r + tau, 1), (2 * s + tau, 0)))
                erpa_op = of.normal_ordered(
                    of.commutator(qp_op, of.commutator(rha_fermion, rs_op)))
                true = rdms.expectation(of.get_interaction_operator(erpa_op))
                assert np.isclose(true, test)


def test_erpa_eom_ham_lih():
    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)
    reduced_ham = make_reduced_hamiltonian(molecule.get_molecular_hamiltonian(),
                                           molecule.n_electrons)
    rha_fermion = of.get_fermion_operator(reduced_ham)
    permuted_hijkl = np.einsum('ijlk', reduced_ham.two_body_tensor)
    opdm = np.diag([1] * molecule.n_electrons + [0] *
                   (molecule.n_qubits - molecule.n_electrons))
    tpdm = 2 * of.wedge(opdm, opdm, (1, 1), (1, 1))
    rdms = of.InteractionRDM(opdm, tpdm)
    dim = 3  # so we don't do the full basis.  This would make the test long
    full_basis = {}  # erpa basis.  A, B basis in RPA language
    cnt = 0
    # start from 1 to make test shorter
    for p, q in product(range(1, dim), repeat=2):
        if p < q:
            full_basis[(p, q)] = cnt
            full_basis[(q, p)] = cnt + dim * (dim - 1) // 2
            cnt += 1
    for rkey in full_basis.keys():
        p, q = rkey
        for ckey in full_basis.keys():
            r, s = ckey
            for sigma, tau in product([0, 1], repeat=2):
                test = erpa_eom_hamiltonian(permuted_hijkl, tpdm, 2 * q + sigma,
                                            2 * p + sigma, 2 * r + tau,
                                            2 * s + tau).real
                qp_op = of.FermionOperator(
                    ((2 * q + sigma, 1), (2 * p + sigma, 0)))
                rs_op = of.FermionOperator(((2 * r + tau, 1), (2 * s + tau, 0)))
                erpa_op = of.normal_ordered(
                    of.commutator(qp_op, of.commutator(rha_fermion, rs_op)))
                true = rdms.expectation(of.get_interaction_operator(erpa_op))
                assert np.isclose(true, test)
