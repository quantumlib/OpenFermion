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
    from openfermion.resource_estimates.pbc.df.generate_costing_table_df import (
        generate_costing_table,
    )
    from openfermion.resource_estimates.pbc.testing import make_diamond_113_szv


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_generate_costing_table_df():
    mf = make_diamond_113_szv()
    thresh = np.array([0.1, 1e-2, 1e-14])  # Eigenvalue threshold for second factorization.
    table = generate_costing_table(mf, cutoffs=thresh, chi=10, beta=22, dE_for_qpe=1e-3)
    assert np.allclose(table.dE, 1e-3)
    assert np.allclose(table.chi, 10)
    assert np.allclose(table.beta, 22)
    assert np.allclose(table.cutoff, thresh)
    assert np.allclose(table.num_aux, [648] * 3)
    assert np.isclose(table.approx_energy.values[2], table.exact_energy.values[0])
