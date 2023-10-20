# coverage: ignore
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
import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    import jax
    import jax.numpy as jnp
    from pyscf.pbc import gto, mp, scf

    from openfermion.resource_estimates.pbc.hamiltonian import (
        build_momentum_transfer_mapping,
        cholesky_from_df_ints,
    )
    from openfermion.resource_estimates.pbc.thc.factorizations.isdf import solve_kmeans_kpisdf
    from openfermion.resource_estimates.pbc.thc.factorizations.thc_jax import (
        adagrad_opt_kpthc_batched,
        get_zeta_size,
        kpoint_thc_via_isdf,
        lbfgsb_opt_kpthc_l2reg,
        lbfgsb_opt_kpthc_l2reg_batched,
        make_contiguous_cholesky,
        pack_thc_factors,
        prepare_batched_data_indx_arrays,
        thc_objective_regularized,
        thc_objective_regularized_batched,
        unpack_thc_factors,
    )
    from openfermion.resource_estimates.thc.utils.thc_factorization import lbfgsb_opt_thc_l2reg
    from openfermion.resource_estimates.thc.utils.thc_factorization import (
        thc_objective_regularized as thc_obj_mol,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_kpoint_thc_reg_gamma():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.mesh = [11] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    num_mo = mf.mo_coeff[0].shape[-1]
    num_interp_points = 10 * mf.mo_coeff[0].shape[-1]
    kpt_thc = solve_kmeans_kpisdf(mf, num_interp_points, single_translation=False, verbose=False)
    chi, zeta, g_mapping = kpt_thc.chi, kpt_thc.zeta, kpt_thc.g_mapping
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    buffer = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, buffer)
    num_G_per_Q = [z.shape[0] for z in zeta]
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        buffer, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    assert np.allclose(chi_unpacked, chi)
    for iq in range(num_kpts):
        assert np.allclose(zeta[iq], zeta_unpacked[iq])
    # force contiguous
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    rsmf.verbose = 0
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    Luv_cont = make_contiguous_cholesky(Luv)
    eri = np.einsum("npq,nrs->pqrs", Luv_cont[0, 0], Luv_cont[0, 0]).real
    buffer = np.zeros((chi.size + get_zeta_size(zeta)), dtype=np.float64)
    # transposed in openfermion
    buffer[: chi.size] = chi.T.real.ravel()
    buffer[chi.size :] = zeta[iq].real.ravel()
    np.random.seed(7)
    opt_param = lbfgsb_opt_thc_l2reg(
        eri, num_interp_points, maxiter=10, initial_guess=buffer, penalty_param=None
    )
    chi_unpacked_mol = opt_param[: chi.size].reshape((num_interp_points, num_mo)).T
    zeta_unpacked_mol = opt_param[chi.size :].reshape(zeta[0].shape)
    opt_param, _ = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        g_mapping,
        jnp.array(Luv_cont),
        maxiter=10,
        penalty_param=None,
        disp_freq=-1,
    )
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        opt_param, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    assert np.allclose(chi_unpacked[0], chi_unpacked_mol)
    assert np.allclose(zeta_unpacked[0], zeta_unpacked_mol)
    mol_obj = thc_obj_mol(buffer, num_mo, num_interp_points, eri, 1e-3)
    buffer = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, buffer)
    gam_obj = thc_objective_regularized(
        buffer, num_mo, num_interp_points, momentum_map, g_mapping, Luv_cont, 1e-3
    )
    assert mol_obj - gam_obj < 1e-12


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_kpoint_thc_reg_batched():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.mesh = [11] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    num_interp_points = 10 * cell.nao
    kpt_thc = solve_kmeans_kpisdf(mf, num_interp_points, single_translation=False, verbose=False)
    chi, zeta, g_mapping = kpt_thc.chi, kpt_thc.zeta, kpt_thc.g_mapping
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    Luv_cont = make_contiguous_cholesky(Luv)
    buffer = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    # Pack THC factors into flat array
    pack_thc_factors(chi, zeta, buffer)
    num_G_per_Q = [z.shape[0] for z in zeta]
    num_mo = mf.mo_coeff[0].shape[-1]
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        buffer, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    # Test packing/unpacking operation
    assert np.allclose(chi_unpacked, chi)
    for iq in range(num_kpts):
        assert np.allclose(zeta[iq], zeta_unpacked[iq])
    # Test objective is the same batched/non-batched
    penalty = 1e-3
    obj_ref = thc_objective_regularized(
        buffer, num_mo, num_interp_points, momentum_map, g_mapping, Luv_cont, penalty
    )
    # # Test gradient is the same
    indx_arrays = prepare_batched_data_indx_arrays(momentum_map, g_mapping)
    batch_size = num_kpts**2
    obj_batched = thc_objective_regularized_batched(
        buffer,
        num_mo,
        num_interp_points,
        momentum_map,
        g_mapping,
        Luv_cont,
        indx_arrays,
        batch_size,
        penalty_param=penalty,
    )
    assert abs(obj_ref - obj_batched) < 1e-12
    grad_ref_fun = jax.grad(thc_objective_regularized)
    grad_ref = grad_ref_fun(
        buffer, num_mo, num_interp_points, momentum_map, g_mapping, Luv_cont, penalty
    )
    # Test gradient is the same
    grad_batched_fun = jax.grad(thc_objective_regularized_batched)
    grad_batched = grad_batched_fun(
        buffer,
        num_mo,
        num_interp_points,
        momentum_map,
        g_mapping,
        Luv_cont,
        indx_arrays,
        batch_size,
        penalty,
    )
    assert np.allclose(grad_batched, grad_ref)
    opt_param, _ = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        g_mapping,
        jnp.array(Luv_cont),
        maxiter=2,
        penalty_param=1e-3,
        disp_freq=-1,
    )
    opt_param_batched, _ = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        g_mapping,
        jnp.array(Luv_cont),
        maxiter=2,
        penalty_param=1e-3,
        disp_freq=-1,
    )
    assert np.allclose(opt_param, opt_param_batched)
    batch_size = 7
    opt_param_batched_diff_batch, _ = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        g_mapping,
        jnp.array(Luv_cont),
        batch_size=batch_size,
        maxiter=2,
        penalty_param=1e-3,
        disp_freq=-1,
    )
    assert np.allclose(opt_param_batched, opt_param_batched_diff_batch)
    ada_param, _ = adagrad_opt_kpthc_batched(
        chi, zeta, momentum_map, g_mapping, jnp.array(Luv_cont), maxiter=2, batch_size=1
    )
    assert np.allclose(opt_param, opt_param_batched)
    batch_size = 7
    ada_param_diff_batch, _ = adagrad_opt_kpthc_batched(
        chi, zeta, momentum_map, g_mapping, jnp.array(Luv_cont), batch_size=batch_size, maxiter=2
    )
    assert np.allclose(ada_param, ada_param_diff_batch)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_kpoint_thc_helper():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.mesh = [11] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    rsmf.verbose = 0
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    cthc = 5
    num_mo = mf.mo_coeff[0].shape[-1]
    # Just testing function runs
    kpt_thc, _ = kpoint_thc_via_isdf(
        mf, Luv, cthc * num_mo, perform_adagrad_opt=False, perform_bfgs_opt=False
    )
    kpt_thc_bfgs, _ = kpoint_thc_via_isdf(
        mf,
        Luv,
        cthc * num_mo,
        perform_adagrad_opt=False,
        perform_bfgs_opt=True,
        bfgs_maxiter=10,
        initial_guess=kpt_thc,
    )
    kpoint_thc_via_isdf(
        mf,
        Luv,
        cthc * num_mo,
        perform_adagrad_opt=True,
        perform_bfgs_opt=True,
        bfgs_maxiter=10,
        adagrad_maxiter=10,
        initial_guess=kpt_thc_bfgs,
    )
