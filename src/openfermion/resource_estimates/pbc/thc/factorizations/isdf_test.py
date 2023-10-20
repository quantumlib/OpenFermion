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
    from ase.build import bulk
    from pyscf.pbc import gto, scf
    from pyscf.pbc.dft import numint
    from pyscf.pbc.lib.kpts_helper import get_kconserv, member, unique
    from pyscf.pbc.tools import pyscf_ase

    from openfermion.resource_estimates.pbc.hamiltonian import build_momentum_transfer_mapping
    from openfermion.resource_estimates.pbc.thc.factorizations.isdf import (
        build_eri_isdf_double_translation,
        build_eri_isdf_single_translation,
        build_g_vector_mappings_double_translation,
        build_g_vector_mappings_single_translation,
        build_g_vectors,
        build_kpoint_zeta,
        build_minus_q_g_mapping,
        get_miller,
        inverse_g_map_double_translation,
        solve_kmeans_kpisdf,
        solve_qrcp_isdf,
        supercell_isdf,
    )
    from openfermion.resource_estimates.pbc.thc.factorizations.kmeans import KMeansCVT


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_supercell_isdf_gamma():
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
    cell.verbose = 0
    cell.mesh = [11] * 3
    cell.build(parse_arg=False)
    # kpts = cell.make_kpts(kmesh, scaled_center=[0.2, 0.3, 0.5])

    mf = scf.RHF(cell)
    mf.kernel()

    from pyscf.pbc.dft import gen_grid

    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    orbitals = numint.eval_ao(cell, grid_points)
    orbitals_mo = np.einsum("Rp,pi->Ri", orbitals, mf.mo_coeff, optimize=True)
    num_mo = mf.mo_coeff.shape[1]
    num_interp_points = np.prod(cell.mesh)
    interp_indx = np.arange(np.prod(cell.mesh))

    chi, zeta, Theta = supercell_isdf(
        mf.with_df, interp_indx, orbitals=orbitals_mo, grid_points=grid_points
    )
    assert Theta.shape == (len(grid_points), num_interp_points)
    # Check overlap
    # Evaluate overlap from orbitals. Do it integral way to ensure
    # discretization error is the same from using coarse FFT grid for testing
    # speed
    ovlp_ao = np.einsum("mp,mq,m->pq", orbitals.conj(), orbitals, grid_inst.weights, optimize=True)
    ovlp_mo = np.einsum("pi,pq,qj->ij", mf.mo_coeff.conj(), ovlp_ao, mf.mo_coeff, optimize=True)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights, optimize=True)
    orbitals_mo_interp = orbitals_mo[interp_indx]
    ovlp_isdf = np.einsum(
        "mi,mj,m->ij", orbitals_mo_interp.conj(), orbitals_mo_interp, ovlp_mu, optimize=True
    )
    assert np.allclose(ovlp_mo, ovlp_isdf)
    # Check ERIs.
    eri_ref = mf.with_df.ao2mo(mf.mo_coeff)
    from pyscf import ao2mo

    eri_ref = ao2mo.restore(1, eri_ref, num_mo)
    # THC eris
    Lijn = np.einsum("mi,mj,mn->ijn", chi.conj(), chi, zeta, optimize=True)
    eri_thc = np.einsum("ijn,nk,nl->ijkl", Lijn, chi.conj(), chi, optimize=True)
    assert np.allclose(eri_thc, eri_ref)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_supercell_isdf_complex():
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
    # kpts = cell.make_kpts(kmesh, scaled_center=[0.2, 0.3, 0.5])

    mf = scf.RHF(cell, kpt=np.array([0.1, -0.001, 0.022]))
    mf.kernel()
    assert np.max(np.abs(mf.mo_coeff.imag)) > 1e-2

    from pyscf.pbc.dft import gen_grid

    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    orbitals = numint.eval_ao(cell, grid_points, kpt=mf.kpt)
    orbitals_mo = np.einsum("Rp,pi->Ri", orbitals, mf.mo_coeff, optimize=True)
    num_mo = mf.mo_coeff.shape[1]
    num_interp_points = 10 * num_mo
    nocc = cell.nelec[0]
    density = np.einsum(
        "Ri,Ri->R", orbitals_mo[:, :nocc].conj(), orbitals_mo[:, :nocc], optimize=True
    )
    kmeans = KMeansCVT(grid_points, max_iteration=500)
    interp_indx = kmeans.find_interpolating_points(num_interp_points, density.real, verbose=False)

    chi, zeta, Theta = supercell_isdf(
        mf.with_df, interp_indx, orbitals=orbitals_mo, grid_points=grid_points
    )
    assert Theta.shape == (len(grid_points), num_interp_points)
    # Check overlap
    # Evaluate overlap from orbitals. Do it integral way to ensure
    # discretization error is the same from using coarse FFT grid for testing
    # speed
    ovlp_ao = np.einsum("mp,mq,m->pq", orbitals.conj(), orbitals, grid_inst.weights, optimize=True)
    ovlp_mo = np.einsum("pi,pq,qj->ij", mf.mo_coeff.conj(), ovlp_ao, mf.mo_coeff, optimize=True)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights, optimize=True)
    orbitals_mo_interp = orbitals_mo[interp_indx]
    ovlp_isdf = np.einsum(
        "mi,mj,m->ij", orbitals_mo_interp.conj(), orbitals_mo_interp, ovlp_mu, optimize=True
    )
    assert np.allclose(ovlp_mo, ovlp_isdf)
    # Check ERIs.
    eri_ref = mf.with_df.ao2mo(mf.mo_coeff, kpts=mf.kpt.reshape((1, -1)))
    # Check there is a complex component
    assert np.max(np.abs(eri_ref.imag)) > 1e-3
    # for complex integrals ao2mo will yield num_mo^4 elements.
    eri_ref = eri_ref.reshape((num_mo,) * 4)
    # THC eris
    Lijn = np.einsum("mi,mj,mn->ijn", chi.conj(), chi, zeta, optimize=True)
    eri_thc = np.einsum("ijn,nk,nl->ijkl", Lijn, chi.conj(), chi, optimize=True)
    assert np.allclose(eri_thc, eri_ref)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_G_vector_mapping_double_translation():
    ase_atom = bulk("AlN", "wurtzite", a=3.11, c=4.98)
    cell = gto.Cell()
    cell.exp_to_discard = 0.1
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:].copy()
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.verbose = 0
    cell.build(parse_arg=False)

    nk = 3
    kmesh = [nk, nk, nk]
    kpts = cell.make_kpts(kmesh)

    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    (G_vecs, G_map, G_unique, _) = build_g_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    num_kpts = len(kpts)
    for iq in range(num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            q = kpts[ikp] - kpts[ikq]
            G_shift = G_vecs[G_map[iq, ikp]]
            assert np.allclose(q, kpts[iq] + G_shift)
    for iq in range(num_kpts):
        unique_G = np.unique(G_map[iq])
        for i, G in enumerate(G_map[iq]):
            assert unique_G[G_unique[iq][i]] == G

    inv_G_map = inverse_g_map_double_translation(cell, kpts, momentum_map)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ix_G_qk = G_map[iq, ik]
            assert ik in inv_G_map[iq, ix_G_qk]


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_G_vector_mapping_single_translation():
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
    cell.verbose = 0
    cell.build(parse_arg=False)

    nk = 3
    kmesh = [nk, nk, nk]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)

    kpts_pq = np.array([(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)])

    kpts_pq_indx = np.array([(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)])
    transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
    assert len(transfers) == (nk**3) ** 2
    _, unique_indx, _ = unique(transfers)
    (_, _, G_unique, delta_Gs) = build_g_vector_mappings_single_translation(
        cell, kpts, kpts_pq_indx[unique_indx]
    )
    kconserv = get_kconserv(cell, kpts)
    for ikp in range(num_kpts):
        for ikq in range(num_kpts):
            for ikr in range(num_kpts):
                iks = kconserv[ikp, ikq, ikr]
                delta_G_expected = kpts[ikp] - kpts[ikq] + kpts[ikr] - kpts[iks]
                q = kpts[ikp] - kpts[ikq]
                qindx = member(q, transfers[unique_indx])[0]
                # print(q, len(transfers[unique_indx]), len(transfers))
                dG_indx = G_unique[qindx, ikr]
                # print(qindx, dG_indx, delta_Gs[qindx].shape)
                # print(qindx)
                # print(len(delta_Gs[qindx]), dG_indx)
                delta_G = delta_Gs[qindx][dG_indx]
                assert np.allclose(delta_G_expected, delta_G)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_kpoint_isdf_double_translation():
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
    cell.verbose = 0
    cell.mesh = [11] * 3  # set to coarse value to just check ERIS numerically.
    cell.build(parse_arg=False)

    kmesh = [1, 2, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    num_mo = mf.mo_coeff[0].shape[-1]
    num_thc = np.prod(cell.mesh)
    kpt_thc = solve_kmeans_kpisdf(
        mf, num_thc, use_density_guess=True, verbose=False, single_translation=False
    )
    num_kpts = len(momentum_map)
    for iq in range(1, num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            for iks in range(num_kpts):
                ikr = momentum_map[iq, iks]
                _ = kpt_thc.g_mapping[iq, iks]
                kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
                mos_pqrs = [mf.mo_coeff[ikp], mf.mo_coeff[ikq], mf.mo_coeff[ikr], mf.mo_coeff[iks]]
                eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
                    (num_mo,) * 4
                )
                eri_pqrs_isdf = build_eri_isdf_double_translation(
                    kpt_thc.chi, kpt_thc.zeta, iq, [ikp, ikq, ikr, iks], kpt_thc.g_mapping
                )
                assert np.allclose(eri_pqrs, eri_pqrs_isdf)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_kpoint_isdf_single_translation():
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
    cell.verbose = 0
    cell.mesh = [11] * 3  # set to coarse value to just check ERIS numerically.
    cell.build(parse_arg=False)

    kmesh = [1, 2, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    num_thc = np.prod(cell.mesh)
    num_kpts = len(kpts)
    kpt_thc = solve_kmeans_kpisdf(
        mf, num_thc, use_density_guess=True, verbose=False, single_translation=True
    )
    kpts_pq = np.array([(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)])
    transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
    _, unique_indx, _ = unique(transfers)
    kconserv = get_kconserv(cell, kpts)
    num_mo = mf.mo_coeff[0].shape[-1]
    for ikp in range(num_kpts):
        for ikq in range(num_kpts):
            for ikr in range(num_kpts):
                iks = kconserv[ikp, ikq, ikr]
                kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
                mos_pqrs = [mf.mo_coeff[ikp], mf.mo_coeff[ikq], mf.mo_coeff[ikr], mf.mo_coeff[iks]]
                eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
                    (num_mo,) * 4
                )
                q = kpts[ikp] - kpts[ikq]
                qindx = member(q, transfers[unique_indx])[0]
                eri_pqrs_isdf = build_eri_isdf_single_translation(
                    kpt_thc.chi, kpt_thc.zeta, qindx, [ikp, ikq, ikr, iks], kpt_thc.g_mapping
                )
                assert np.allclose(eri_pqrs, eri_pqrs_isdf)


def get_complement(miller_indx, kmesh):
    complement = ~miller_indx
    complement[np.where(np.array(kmesh) == 1)] = 0
    return complement


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_kpoint_isdf_symmetries():
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

    kmesh = [1, 2, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    num_thc = np.prod(cell.mesh)  # should be no THC error when selecting all the grid points.
    kpt_thc = solve_kmeans_kpisdf(
        mf, num_thc, use_density_guess=True, verbose=False, single_translation=False
    )
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    (_, _, G_unique, delta_Gs) = build_g_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    _, minus_Q_G_map_unique = build_minus_q_g_mapping(cell, kpts, momentum_map)
    num_kpts = len(kpts)
    # Test symmetries from Appendix D of https://arxiv.org/pdf/2302.05531.pdf
    # Test LHS for sanity too (need to uncomment)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    from pyscf.pbc.lib.kpts_helper import conj_mapping

    minus_k_map = conj_mapping(cell, kpts)
    iq, ik, ik_prime = np.random.randint(0, num_kpts, 3)
    # Get -Q index
    minus_iq = minus_k_map[iq]
    ik_minus_q = momentum_map[iq, ik]
    iGpq = G_unique[iq, ik]
    iGsr = G_unique[iq, ik_prime]
    ik_prime_minus_q = momentum_map[iq, ik_prime]
    # Sanity check G mappings
    assert np.allclose(kpts[ik] - kpts[ik_minus_q] - kpts[iq], delta_Gs[iq][iGpq])
    assert np.allclose(kpts[ik_prime] - kpts[ik_prime_minus_q] - kpts[iq], delta_Gs[iq][iGsr])
    # (pk qk-Q | rk'-Q sk') = (q k-Q p k | sk' rk'-Q)*
    ik_prime_minus_q = momentum_map[iq, ik_prime]
    # uncomment to check normal eris
    # kpt_pqrs = [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
    # eri_pqrs = build_eri(mf, kpt_pqrs)
    # kpt_pqrs = [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
    # kpt_pqrs = [ik_minus_q, ik, ik_prime, ik_prime_minus_q]
    # eri_qpsr = build_eri(mf, kpt_pqrs).transpose((1, 0, 3, 2))
    # Sanity check relationship
    # assert np.allclose(eri_pqrs, eri_qpsr.conj())
    # Now check how to index into correct G when Q is conjugated
    # We want to find (-Q) + G_pq_comp + (Q + Gpq) = 0,
    # Q + Gpq = kp - kq = q
    # so G_pq_comp = -((-Q) + (Q+Gpq))
    iGpq_comp = minus_Q_G_map_unique[minus_iq, ik]
    iGsr_comp = minus_Q_G_map_unique[minus_iq, ik_prime]
    # Check zeta symmetry: expect zeta[Q,G1,G2,m,n] =
    # zeta[-Q,G1_comp,G2_comp,m, n].conj()
    # Build refernce point zeta[Q,G1,G2,m,n]
    zeta_ref = kpt_thc.zeta[iq][iGpq, iGsr]
    zeta_test = kpt_thc.zeta[minus_iq][iGpq_comp, iGsr_comp]
    # F31 (pk qk-Q | rk'-Q sk') = (rk'-Q s k'| pk qk-Q)
    assert np.allclose(zeta_ref, zeta_test.conj())
    # Sanity check do literal minus signs (should be complex
    # conjugate)
    zeta_test = build_kpoint_zeta(
        mf.with_df, -kpts[iq], -delta_Gs[iq][iGpq], -delta_Gs[iq][iGsr], grid_points, kpt_thc.xi
    )
    assert np.allclose(zeta_ref, zeta_test.conj())
    # (pk qk-Q | rk'-Q sk') = (rk'-Q s k'| pk qk-Q)
    # uncomment to check normal eris
    # kpt_pqrs = [ik_prime_minus_q, ik_prime, ik, ik_minus_q]
    # eri_rspq = build_eri(mf, kpt_pqrs).transpose((2, 3, 0, 1))
    # assert np.allclose(eri_pqrs, eri_rspq)
    # Check zeta symmetry: expect zeta[Q,G1,G2,m,n] =
    # zeta[-Q,G2_comp,G1_comp,m, n]
    zeta_test = kpt_thc.zeta[minus_iq][iGsr_comp, iGpq_comp]
    assert np.allclose(zeta_ref, zeta_test.T)
    # (pk qk-Q | rk'-Q sk') = (sk' r k'-Q| qk-Q pk)
    # uncomment to check normal eris
    # kpt_pqrs = [ik_prime, ik_prime_minus_q, ik_minus_q, ik]
    # eri_srqp = build_eri(mf, kpt_pqrs).transpose((3, 2, 1, 0))
    # assert np.allclose(eri_pqrs, eri_srqp.conj())
    # Check zeta symmetry: expect zeta[Q,G1,G2,m,n]
    # = zeta[Q,G2,G1,n, m].conj()
    zeta_test = kpt_thc.zeta[iq][iGsr, iGpq]
    assert np.allclose(zeta_ref, zeta_test.conj().T)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_symmetry_of_G_maps():
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
    cell.verbose = 0
    cell.build(parse_arg=False)

    kmesh = [3, 3, 3]
    kpts = cell.make_kpts(kmesh)
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    (G_vecs, G_map, _, delta_Gs) = build_g_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    G_dict, _ = build_g_vectors(cell)
    num_kpts = len(kpts)
    lattice_vectors = cell.lattice_vectors()
    from pyscf.pbc.lib.kpts_helper import conj_mapping

    minus_k_map = conj_mapping(cell, kpts)
    # k1 - k2 = Q + G
    for iq in range(1, num_kpts):
        minus_iq = minus_k_map[iq]
        for ik in range(num_kpts):
            Gpq = G_vecs[G_map[iq, ik]]
            Gpq_comp = -(kpts[minus_iq] + kpts[iq] + Gpq)
            iGpq_comp = G_dict[tuple(get_miller(lattice_vectors, Gpq_comp))]
            G_indx_unique = [
                G_dict[tuple(get_miller(lattice_vectors, G))] for G in delta_Gs[minus_iq]
            ]
            if iq == 1:
                pass
            assert iGpq_comp in G_indx_unique
            for _ in range(num_kpts):
                # Check complement(miller_Gpq) = miller_Gpq_comp
                # Get indx of "complement" G in original set of 27
                iGsr_comp = G_dict[tuple(get_miller(lattice_vectors, Gpq_comp))]
                # Get index of unique Gs in original set of 27
                # Check complement is in set corresponding to zeta[-Q]
                assert iGsr_comp in G_indx_unique

    # Check minus Q mapping
    minus_Q_G_map, minus_Q_G_map_unique = build_minus_q_g_mapping(cell, kpts, momentum_map)
    for iq in range(1, num_kpts):
        minus_iq = minus_k_map[iq]
        for ik in range(num_kpts):
            Gpq = G_vecs[G_map[iq, ik]]
            Gpq_comp = -(kpts[minus_iq] + kpts[iq] + Gpq)
            iGpq_comp = G_dict[tuple(get_miller(lattice_vectors, Gpq_comp))]
            assert iGpq_comp == minus_Q_G_map[minus_iq, ik]
            indx_in_unique_set = minus_Q_G_map_unique[minus_iq, ik]
            Gpq_comp_from_map = delta_Gs[iq][indx_in_unique_set]
            assert np.allclose(Gpq_comp, Gpq_comp_from_map)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_isdf_qrcp():
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
    cell.mesh = [11] * 3
    cell.unit = "B"
    cell.verbose = 0
    cell.build()

    kmesh = [1, 2, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    num_mo = mf.mo_coeff[0].shape[-1]
    num_thc = np.prod(cell.mesh)
    kpt_thc = solve_qrcp_isdf(mf, num_thc, single_translation=False)
    num_kpts = len(kpts)
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    for iq in range(1, num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            for iks in range(num_kpts):
                ikr = momentum_map[iq, iks]
                kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
                mos_pqrs = [mf.mo_coeff[ikp], mf.mo_coeff[ikq], mf.mo_coeff[ikr], mf.mo_coeff[iks]]
                eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
                    (num_mo,) * 4
                )
                eri_pqrs_isdf = build_eri_isdf_double_translation(
                    kpt_thc.chi, kpt_thc.zeta, iq, [ikp, ikq, ikr, iks], kpt_thc.g_mapping
                )
                assert np.allclose(eri_pqrs, eri_pqrs_isdf)
