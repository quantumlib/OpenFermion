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
import itertools

import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from pyscf.pbc import cc, mp

    from openfermion.resource_estimates.pbc.hamiltonian import (
        build_hamiltonian,
        cholesky_from_df_ints,
    )
    from openfermion.resource_estimates.pbc.testing import make_diamond_113_szv


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_build_hamiltonian():
    mf = make_diamond_113_szv()
    nmo = mf.mo_coeff[0].shape[-1]
    naux = 108
    hcore, chol = build_hamiltonian(mf)
    nkpts = len(mf.mo_coeff)
    assert hcore.shape == (nkpts, nmo, nmo)
    assert chol.shape == (nkpts, nkpts)
    assert chol[0, 0].shape == (naux, nmo, nmo)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_pyscf_chol_from_df():
    mf = make_diamond_113_szv()
    mymp = mp.KMP2(mf)
    nmo = mymp.nmo
    nocc = mymp.nocc
    nvir = nmo - nocc
    Luv = cholesky_from_df_ints(mymp)

    # 1. Test that the DF integrals give the correct SCF energy (oo block)
    mf.exxdiv = None  # exclude ewald exchange correction
    Eref = mf.energy_elec()[1]
    Eout = 0.0j
    nkpts = len(mf.mo_coeff)
    for ik, jk in itertools.product(range(nkpts), repeat=2):
        Lii = Luv[ik, ik][:, :nocc, :nocc]
        Ljj = Luv[jk, jk][:, :nocc, :nocc]
        Lij = Luv[ik, jk][:, :nocc, :nocc]
        Lji = Luv[jk, ik][:, :nocc, :nocc]
        oooo_d = np.einsum("Lij,Lkl->ijkl", Lii, Ljj) / nkpts
        oooo_x = np.einsum("Lij,Lkl->ijkl", Lij, Lji) / nkpts
        Eout += 2.0 * np.einsum("iijj->", oooo_d)
        Eout -= np.einsum("ijji->", oooo_x)
    assert abs(Eout / nkpts - Eref) < 1e-12

    # 2. Test that the DF integrals agree with those from MP2 (ov block)
    from pyscf.pbc.mp.kmp2 import _init_mp_df_eris

    Ltest = _init_mp_df_eris(mymp)
    for ik, jk in itertools.product(range(nkpts), repeat=2):
        assert np.allclose(Luv[ik, jk][:, :nocc, nocc:], Ltest[ik, jk], atol=1e-12)

    # 3. Test that the DF integrals have correct vvvv block (vv)
    Ivvvv = np.zeros((nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), dtype=np.complex128)
    for ik, jk, kk in itertools.product(range(nkpts), repeat=3):
        lk = mymp.khelper.kconserv[ik, jk, kk]
        Lij = Luv[ik, jk][:, nocc:, nocc:]
        Lkl = Luv[kk, lk][:, nocc:, nocc:]
        Imo = np.einsum("Lij,Lkl->ijkl", Lij, Lkl)
        Ivvvv[ik, jk, kk] = Imo / nkpts

    mycc = cc.KRCCSD(mf)
    eris = mycc.ao2mo()
    assert np.allclose(eris.vvvv, Ivvvv.transpose(0, 2, 1, 3, 5, 4, 6), atol=1e-12)
