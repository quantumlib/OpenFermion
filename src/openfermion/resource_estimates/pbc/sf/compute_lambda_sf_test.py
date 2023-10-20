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
    from ase.build import bulk

    from pyscf.pbc import gto, scf, mp
    from pyscf.pbc.tools import pyscf_ase

    from openfermion.resource_estimates.pbc.sf.compute_lambda_sf import compute_lambda
    from openfermion.resource_estimates.pbc.sf.sf_integrals import SingleFactorization
    from openfermion.resource_estimates.pbc.hamiltonian import (
        build_hamiltonian,
        cholesky_from_df_ints,
    )
    from openfermion.resource_estimates.pbc.testing import make_diamond_113_szv


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_lambda_calc():
    mf = make_diamond_113_szv()
    hcore, Luv = build_hamiltonian(mf)
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf)
    lambda_data = compute_lambda(hcore, helper)
    assert np.isclose(lambda_data.lambda_total, 2123.4342903006627)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_padding():
    ase_atom = bulk("H", "bcc", a=2.0, cubic=True)
    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:].copy()
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.verbose = 0
    cell.build()

    kmesh = [1, 2, 2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    from pyscf.pbc.mp.kmp2 import _add_padding

    mymp = mp.KMP2(mf)
    Luv_padded = cholesky_from_df_ints(mymp)
    mo_coeff_padded = _add_padding(mymp, mymp.mo_coeff, mymp.mo_energy)[0]
    helper = SingleFactorization(cholesky_factor=Luv_padded, kmf=mf)
    assert mf.mo_coeff[0].shape[-1] != mo_coeff_padded[0].shape[-1]

    hcore_ao = mf.get_hcore()
    hcore_no_padding = np.asarray(
        [reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)]
    )
    hcore_padded = np.asarray(
        [reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mo_coeff_padded)]
    )
    assert hcore_no_padding[0].shape != hcore_padded[0].shape
    assert np.isclose(np.sum(hcore_no_padding), np.sum(hcore_padded))
    Luv_no_padding = cholesky_from_df_ints(mymp, pad_mos_with_zeros=False)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            assert np.isclose(np.sum(Luv_padded[k1, k2]), np.sum(Luv_no_padding[k1, k2]))

    helper_no_padding = SingleFactorization(cholesky_factor=Luv_no_padding, kmf=mf)
    lambda_data_pad = compute_lambda(hcore_no_padding, helper_no_padding)
    helper = SingleFactorization(cholesky_factor=Luv_padded, kmf=mf)
    lambda_data_no_pad = compute_lambda(hcore_padded, helper)
    assert np.isclose(lambda_data_pad.lambda_total, lambda_data_no_pad.lambda_total)
