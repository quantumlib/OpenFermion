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
from functools import reduce
import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from pyscf.pbc import mp

    from openfermion.resource_estimates.pbc.df.compute_lambda_df import compute_lambda
    from openfermion.resource_estimates.pbc.df.df_integrals import DFABKpointIntegrals
    from openfermion.resource_estimates.pbc.hamiltonian import cholesky_from_df_ints
    from openfermion.resource_estimates.pbc.testing import make_diamond_113_szv


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_lambda_calc():
    mf = make_diamond_113_szv()
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABKpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0e-13)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)]
    )

    lambda_data = compute_lambda(hcore_mo, helper)
    assert np.isclose(lambda_data.lambda_total, 179.62240330857406)

    lambda_two_body = 0
    lambda_two_body_v2 = 0
    nkpts = len(mf.kpts)
    for qidx in range(nkpts):
        aval_to_square = np.zeros((helper.naux), dtype=np.complex128)
        bval_to_square = np.zeros((helper.naux), dtype=np.complex128)

        aval_to_square_v2 = np.zeros((helper.naux), dtype=np.complex128)
        bval_to_square_v2 = np.zeros((helper.naux), dtype=np.complex128)

        for kidx in range(nkpts):
            Amats, Bmats = helper.build_A_B_n_q_k_from_chol(qidx, kidx)
            Amats /= np.sqrt(nkpts)
            Bmats /= np.sqrt(nkpts)
            wa, _ = np.linalg.eigh(Amats)
            wb, _ = np.linalg.eigh(Bmats)
            aval_to_square += np.einsum("npq->n", np.abs(Amats) ** 2)
            bval_to_square += np.einsum("npq->n", np.abs(Bmats) ** 2)

            aval_to_square_v2 += np.sum(np.abs(wa) ** 2, axis=-1)
            bval_to_square_v2 += np.sum(np.abs(wb) ** 2, axis=-1)
            assert np.allclose(
                np.sum(np.abs(wa) ** 2, axis=-1), np.einsum("npq->n", np.abs(Amats) ** 2)
            )

        lambda_two_body += np.sum(aval_to_square)
        lambda_two_body += np.sum(bval_to_square)

        lambda_two_body_v2 += np.sum(aval_to_square_v2)
        lambda_two_body_v2 += np.sum(bval_to_square_v2)

    assert np.isclose(lambda_two_body, lambda_two_body_v2)
