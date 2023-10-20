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
    from pyscf.pbc import gto, mp, scf
    from openfermion.resource_estimates.pbc.hamiltonian import cholesky_from_df_ints
    from openfermion.resource_estimates.pbc.thc.factorizations.thc_jax import kpoint_thc_via_isdf
    from openfermion.resource_estimates.pbc.thc.compute_lambda_thc import compute_lambda
    from openfermion.resource_estimates.pbc.thc.thc_integrals import KPTHCDoubleTranslation


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_kpoint_thc_lambda():
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
    cell.build()

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    #
    # Build kpoint THC eris
    #
    #
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    # Force same MOs as FFTDF at least
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    cthc = 4
    num_thc = cthc * mf.mo_coeff[0].shape[-1]
    np.random.seed(7)
    kpt_thc, _ = kpoint_thc_via_isdf(
        mf,
        Luv,
        num_thc,
        perform_adagrad_opt=False,
        perform_bfgs_opt=True,
        bfgs_maxiter=10,
        verbose=False,
    )
    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)]
    )
    helper = KPTHCDoubleTranslation(kpt_thc.chi, kpt_thc.zeta, mf)
    lambda_data = compute_lambda(hcore_mo, helper)
    assert np.isclose(lambda_data.lambda_total, 93.84613761765415)
