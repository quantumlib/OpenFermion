from typing import Callable
from itertools import product
import os
import numpy as np
import pytest
import scipy as sp
from scipy.optimize.optimize import OptimizeResult
from openfermion.config import DATA_DIRECTORY
from openfermion import (MolecularData, general_basis_change,
                         InteractionOperator)
from openfermion.hamiltonians._hartree_fock import (
    get_matrix_of_eigs, HartreeFockFunctional, InputError, rhf_params_to_matrix,
    rhf_func_generator, generate_hamiltonian, rhf_minimization)


def test_get_matrix_of_eigs():
    lam_vals = np.random.randn(4) + 1j * np.random.randn(4)
    lam_vals[0] = lam_vals[1]
    mat_eigs = np.zeros((lam_vals.shape[0], lam_vals.shape[0]),
                        dtype=np.complex128)
    for i, j in product(range(lam_vals.shape[0]), repeat=2):
        if np.isclose(abs(lam_vals[i] - lam_vals[j]), 0):
            mat_eigs[i, j] = 1
        else:
            mat_eigs[i, j] = (np.exp(1j * (lam_vals[i] - lam_vals[j])) -
                              1) / (1j * (lam_vals[i] - lam_vals[j]))

    test_mat_eigs = get_matrix_of_eigs(lam_vals)
    assert np.allclose(test_mat_eigs, mat_eigs)


def test_hffunctional_setup():
    fake_obi = np.zeros((8, 8))
    fake_tbi = np.zeros((8, 8, 8, 8))

    def fake_orbital_func(x, y, z):
        return None

    hff = HartreeFockFunctional(one_body_integrals=fake_obi,
                                two_body_integrals=fake_tbi,
                                overlap=fake_obi,
                                n_electrons=6,
                                model='rhf',
                                initial_orbitals=fake_orbital_func)
    assert hff.occ == list(range(3))
    assert hff.virt == list(range(3, 8))
    assert hff.nocc == 3
    assert hff.nvirt == 5

    hff = HartreeFockFunctional(one_body_integrals=fake_obi,
                                two_body_integrals=fake_tbi,
                                overlap=fake_obi,
                                n_electrons=6,
                                model='uhf',
                                initial_orbitals=fake_orbital_func)
    assert hff.occ == list(range(6))
    assert hff.virt == list(range(6, 16))
    assert hff.nocc == 6
    assert hff.nvirt == 10

    hff = HartreeFockFunctional(one_body_integrals=fake_obi,
                                two_body_integrals=fake_tbi,
                                overlap=fake_obi,
                                n_electrons=6,
                                model='ghf',
                                initial_orbitals=fake_orbital_func)
    assert hff.occ == list(range(6))
    assert hff.virt == list(range(6, 16))
    assert hff.nocc == 6
    assert hff.nvirt == 10

    with pytest.raises(InputError):
        hff = HartreeFockFunctional(one_body_integrals=fake_obi,
                                    two_body_integrals=fake_tbi,
                                    overlap=fake_obi,
                                    n_electrons=6,
                                    model='abc',
                                    initial_orbitals=fake_orbital_func)


def test_gradient():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)

    overlap = molecule.overlap_integrals
    mo_obi = molecule.one_body_integrals
    mo_tbi = molecule.two_body_integrals
    rotation_mat = molecule.canonical_orbitals.T.dot(overlap)
    obi = general_basis_change(mo_obi, rotation_mat, (1, 0))
    tbi = general_basis_change(mo_tbi, rotation_mat, (1, 1, 0, 0))
    hff = HartreeFockFunctional(one_body_integrals=obi,
                                two_body_integrals=tbi,
                                overlap=overlap,
                                n_electrons=molecule.n_electrons,
                                model='rhf',
                                nuclear_repulsion=molecule.nuclear_repulsion)

    params = np.random.randn(hff.nocc * hff.nvirt)
    u = sp.linalg.expm(
        rhf_params_to_matrix(params,
                             hff.num_orbitals,
                             occ=hff.occ,
                             virt=hff.virt))
    initial_opdm = np.diag([1] * hff.nocc + [0] * hff.nvirt)
    final_opdm = u.dot(initial_opdm).dot(u.conj().T)
    grad = hff.rhf_global_gradient(params, final_opdm)
    grad_dim = grad.shape[0]

    # get finite difference gradient
    finite_diff_grad = np.zeros(grad_dim)
    epsilon = 0.0001
    for i in range(grad_dim):
        params_epsilon = params.copy()
        params_epsilon[i] += epsilon
        u = sp.linalg.expm(
            rhf_params_to_matrix(params_epsilon,
                                 hff.num_orbitals,
                                 occ=hff.occ,
                                 virt=hff.virt))
        tfinal_opdm = u.dot(initial_opdm).dot(u.conj().T)
        energy_plus_epsilon = hff.energy_from_rhf_opdm(tfinal_opdm)

        params_epsilon[i] -= 2 * epsilon
        u = sp.linalg.expm(
            rhf_params_to_matrix(params_epsilon,
                                 hff.num_orbitals,
                                 occ=hff.occ,
                                 virt=hff.virt))
        tfinal_opdm = u.dot(initial_opdm).dot(u.conj().T)
        energy_minus_epsilon = hff.energy_from_rhf_opdm(tfinal_opdm)

        finite_diff_grad[i] = (energy_plus_epsilon -
                               energy_minus_epsilon) / (2 * epsilon)

    assert np.allclose(finite_diff_grad, grad, atol=epsilon)


def test_gradient_lih():
    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)

    overlap = molecule.overlap_integrals
    mo_obi = molecule.one_body_integrals
    mo_tbi = molecule.two_body_integrals
    rotation_mat = molecule.canonical_orbitals.T.dot(overlap)
    obi = general_basis_change(mo_obi, rotation_mat, (1, 0))
    tbi = general_basis_change(mo_tbi, rotation_mat, (1, 1, 0, 0))

    hff = HartreeFockFunctional(one_body_integrals=obi,
                                two_body_integrals=tbi,
                                overlap=overlap,
                                n_electrons=molecule.n_electrons,
                                model='rhf',
                                nuclear_repulsion=molecule.nuclear_repulsion)

    params = np.random.randn(hff.nocc * hff.nvirt)
    u = sp.linalg.expm(
        rhf_params_to_matrix(params,
                             hff.num_orbitals,
                             occ=hff.occ,
                             virt=hff.virt))
    grad_dim = hff.nocc * hff.nvirt
    initial_opdm = np.diag([1] * hff.nocc + [0] * hff.nvirt)
    final_opdm = u.dot(initial_opdm).dot(u.conj().T)
    grad = hff.rhf_global_gradient(params, final_opdm)

    # get finite difference gradient
    finite_diff_grad = np.zeros(grad_dim)
    epsilon = 0.0001
    for i in range(grad_dim):
        params_epsilon = params.copy()
        params_epsilon[i] += epsilon
        u = sp.linalg.expm(
            rhf_params_to_matrix(params_epsilon,
                                 hff.num_orbitals,
                                 occ=hff.occ,
                                 virt=hff.virt))
        tfinal_opdm = u.dot(initial_opdm).dot(u.conj().T)
        energy_plus_epsilon = hff.energy_from_rhf_opdm(tfinal_opdm)

        params_epsilon[i] -= 2 * epsilon
        u = sp.linalg.expm(
            rhf_params_to_matrix(params_epsilon,
                                 hff.num_orbitals,
                                 occ=hff.occ,
                                 virt=hff.virt))
        tfinal_opdm = u.dot(initial_opdm).dot(u.conj().T)
        energy_minus_epsilon = hff.energy_from_rhf_opdm(tfinal_opdm)

        finite_diff_grad[i] = (energy_plus_epsilon -
                               energy_minus_epsilon) / (2 * epsilon)

    assert np.allclose(finite_diff_grad, grad, atol=epsilon)


def test_rhf_func_generator():
    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)

    overlap = molecule.overlap_integrals
    mo_obi = molecule.one_body_integrals
    mo_tbi = molecule.two_body_integrals
    rotation_mat = molecule.canonical_orbitals.T.dot(overlap)
    obi = general_basis_change(mo_obi, rotation_mat, (1, 0))
    tbi = general_basis_change(mo_tbi, rotation_mat, (1, 1, 0, 0))

    hff = HartreeFockFunctional(one_body_integrals=obi,
                                two_body_integrals=tbi,
                                overlap=overlap,
                                n_electrons=molecule.n_electrons,
                                model='rhf',
                                nuclear_repulsion=molecule.nuclear_repulsion)
    unitary, energy, gradient = rhf_func_generator(hff)
    assert isinstance(unitary, Callable)
    assert isinstance(energy, Callable)
    assert isinstance(gradient, Callable)

    params = np.random.randn(hff.nocc * hff.nvirt)
    u = unitary(params)
    assert np.allclose(u.conj().T.dot(u), np.eye(hff.num_orbitals))
    assert isinstance(energy(params), float)
    assert isinstance(gradient(params), np.ndarray)

    _, _, _, opdm_func = rhf_func_generator(hff, get_opdm_func=True)
    assert isinstance(opdm_func, Callable)
    assert isinstance(opdm_func(params), np.ndarray)
    assert np.isclose(opdm_func(params).shape[0], hff.num_orbitals)

    _, energy, _ = rhf_func_generator(hff,
                                      init_occ_vec=np.array([1, 1, 1, 1, 0, 0]))
    assert isinstance(energy(params), float)


def test_rhf_params_to_matrix():
    params = np.random.randn(4)
    true_kappa = np.zeros((4, 4))
    true_kappa[0, 2], true_kappa[2, 0] = -params[0], params[0]
    true_kappa[1, 2], true_kappa[2, 1] = -params[1], params[1]
    true_kappa[0, 3], true_kappa[3, 0] = -params[2], params[2]
    true_kappa[1, 3], true_kappa[3, 1] = -params[3], params[3]
    test_kappa = rhf_params_to_matrix(params, 4)
    assert np.allclose(test_kappa, true_kappa)

    test_kappa = rhf_params_to_matrix(params, 4, occ=list(range(2)))
    assert np.allclose(test_kappa, true_kappa)

    test_kappa = rhf_params_to_matrix(params, 4, virt=list(range(2, 4)))
    assert np.allclose(test_kappa, true_kappa)

    with pytest.raises(ValueError):
        rhf_params_to_matrix(params + 1j, 4)


def test_generate_hamiltonian():
    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)
    mo_obi = molecule.one_body_integrals
    mo_tbi = molecule.two_body_integrals
    mol_ham = generate_hamiltonian(mo_obi, mo_tbi, constant=0)
    assert isinstance(mol_ham, InteractionOperator)
    assert np.allclose(mol_ham.one_body_tensor[::2, ::2], mo_obi)
    assert np.allclose(mol_ham.one_body_tensor[1::2, 1::2], mo_obi)
    assert np.allclose(mol_ham.two_body_tensor[::2, ::2, ::2, ::2],
                       0.5 * mo_tbi)
    assert np.allclose(mol_ham.two_body_tensor[1::2, 1::2, 1::2, 1::2],
                       0.5 * mo_tbi)


def test_rhf_min():
    filename = os.path.join(DATA_DIRECTORY, "H2_sto-3g_singlet_0.7414.hdf5")
    molecule = MolecularData(filename=filename)

    overlap = molecule.overlap_integrals
    mo_obi = molecule.one_body_integrals
    mo_tbi = molecule.two_body_integrals
    rotation_mat = molecule.canonical_orbitals.T.dot(overlap)
    obi = general_basis_change(mo_obi, rotation_mat, (1, 0))
    tbi = general_basis_change(mo_tbi, rotation_mat, (1, 1, 0, 0))
    hff = HartreeFockFunctional(one_body_integrals=obi,
                                two_body_integrals=tbi,
                                overlap=overlap,
                                n_electrons=molecule.n_electrons,
                                model='rhf',
                                nuclear_repulsion=molecule.nuclear_repulsion)
    result = rhf_minimization(hff)
    assert isinstance(result, OptimizeResult)

    result2 = rhf_minimization(hff,
                               initial_guess=np.array([0]),
                               sp_options={
                                   'maxiter': 100,
                                   'disp': False
                               })
    assert isinstance(result2, OptimizeResult)
    assert np.isclose(result2.fun, result.fun)
