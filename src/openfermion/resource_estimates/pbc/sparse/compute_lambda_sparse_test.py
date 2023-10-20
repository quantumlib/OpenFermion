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
    from openfermion.resource_estimates.pbc.sparse.compute_lambda_sparse import compute_lambda
    from openfermion.resource_estimates.pbc.sparse.sparse_integrals import SparseFactorization
    from openfermion.resource_estimates.pbc.hamiltonian import cholesky_from_df_ints
    from openfermion.resource_estimates.pbc.testing import make_diamond_113_szv


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_lambda_sparse():
    mf = make_diamond_113_szv()
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = SparseFactorization(cholesky_factor=Luv, kmf=mf)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)]
    )
    compute_lambda(hcore_mo, helper)
