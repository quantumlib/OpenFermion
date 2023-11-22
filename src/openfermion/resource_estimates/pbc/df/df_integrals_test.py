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
    from pyscf.pbc import mp

    from openfermion.resource_estimates.pbc.testing import make_diamond_113_szv
    from openfermion.resource_estimates.pbc.df.df_integrals import DFABKpointIntegrals
    from openfermion.resource_estimates.pbc.hamiltonian import cholesky_from_df_ints


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_df_amat_bmat():
    mf = make_diamond_113_szv()
    mymp = mp.KMP2(mf)
    nmo = mymp.nmo

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    dfk_inst = DFABKpointIntegrals(Luv.copy(), mf)
    naux = dfk_inst.naux

    dfk_inst.double_factorize()

    nkpts = len(mf.kpts)
    for qidx, kidx in itertools.product(range(nkpts), repeat=2):
        Amats, Bmats = dfk_inst.build_A_B_n_q_k_from_chol(qidx, kidx)
        # check if Amats and Bmats have the correct size
        assert Amats.shape == (naux, 2 * nmo, 2 * nmo)
        assert Bmats.shape == (naux, 2 * nmo, 2 * nmo)

        # check if Amats and Bmats have the correct symmetry--Hermitian
        assert np.allclose(Amats, Amats.conj().transpose(0, 2, 1))
        assert np.allclose(Bmats, Bmats.conj().transpose(0, 2, 1))

        # check if we can recover the Cholesky vector from Amat
        k_minus_q_idx = dfk_inst.k_transfer_map[qidx, kidx]
        test_chol = dfk_inst.build_chol_part_from_A_B(kidx, qidx, Amats, Bmats)
        assert np.allclose(test_chol, dfk_inst.chol[kidx, k_minus_q_idx])

        # check if factorized is working numerically exact case
        assert np.allclose(dfk_inst.amat_n_mats[kidx, qidx], Amats)
        assert np.allclose(dfk_inst.bmat_n_mats[kidx, qidx], Bmats)

        for nn in range(Amats.shape[0]):
            w, v = np.linalg.eigh(Amats[nn, :, :])
            non_zero_idx = np.where(w > 1.0e-4)[0]
            w = w[non_zero_idx]
            v = v[:, non_zero_idx]
            assert len(w) <= 2 * nmo

    for qidx in range(nkpts):
        for nn in range(naux):
            for kidx in range(nkpts):
                eigs_a_fixed_n_q = dfk_inst.amat_lambda_vecs[kidx, qidx, nn]
                eigs_b_fixed_n_q = dfk_inst.bmat_lambda_vecs[kidx, qidx, nn]
                assert len(eigs_a_fixed_n_q) <= 2 * nmo
                assert len(eigs_b_fixed_n_q) <= 2 * nmo

    for kidx in range(nkpts):
        for kpidx in range(nkpts):
            for qidx in range(nkpts):
                kmq_idx = dfk_inst.k_transfer_map[qidx, kidx]
                kpmq_idx = dfk_inst.k_transfer_map[qidx, kpidx]
                exact_eri_block = dfk_inst.get_eri_exact([kidx, kmq_idx, kpmq_idx, kpidx])
                test_eri_block = dfk_inst.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                assert np.allclose(exact_eri_block, test_eri_block)
