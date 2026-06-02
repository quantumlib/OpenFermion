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
    from openfermion.resource_estimates.pbc.sf.compute_sf_resources import (
        _compute_cost,
        compute_cost,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_estimate():
    n = 152
    lam = 3071.8
    L = 275
    dE = 0.001
    chi = 10

    res = _compute_cost(n, lam, L, dE, chi, 20_000, 3, 3, 3)
    # 1663687, 8027577592851, 438447}
    assert np.isclose(res[0], 1663687)
    assert np.isclose(res[1], 8027577592851)
    assert np.isclose(res[2], 438447)
    res = _compute_cost(n, lam, L, dE, chi, res[0], 3, 3, 3)
    assert np.isclose(res[0], 1663707)
    assert np.isclose(res[1], 8027674096311)
    assert np.isclose(res[2], 438452)

    res = _compute_cost(n, lam, L, dE, chi, 20_000, 3, 5, 1)
    # 907828, 4380427154244, 219526
    assert np.isclose(res[0], 907828)
    assert np.isclose(res[1], 4380427154244)
    assert np.isclose(res[2], 219526)
    res = _compute_cost(n, lam, L, dE, chi, res[0], 3, 5, 1)
    assert np.isclose(res[0], 907828)
    assert np.isclose(res[1], 4380427154244)
    assert np.isclose(res[2], 219526)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_estimate_helper():
    n = 152
    lam = 3071.8
    L = 275
    dE = 0.001
    chi = 10

    res = compute_cost(
        num_spin_orbs=n, lambda_tot=lam, num_aux=L, kmesh=[3, 3, 3], dE_for_qpe=dE, chi=chi
    )
    assert np.isclose(res.toffolis_per_step, 1663707)
    assert np.isclose(res.total_toffolis, 8027674096311)
    assert np.isclose(res.logical_qubits, 438452)

    res = compute_cost(
        num_spin_orbs=n, lambda_tot=lam, num_aux=L, kmesh=[3, 5, 1], dE_for_qpe=dE, chi=chi
    )
    # 1663687, 8027577592851, 438447}
    assert np.isclose(res.toffolis_per_step, 907828)
    assert np.isclose(res.total_toffolis, 4380427154244)
    assert np.isclose(res.logical_qubits, 219526)
