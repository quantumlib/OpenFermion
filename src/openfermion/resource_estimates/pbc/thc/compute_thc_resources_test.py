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
    from openfermion.resource_estimates.pbc.thc.compute_thc_resources import (
        _compute_cost,
        compute_cost,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_thc_resources():
    lam = 307.68
    dE = 0.001
    n = 108
    chi = 10
    beta = 16
    M = 350

    res = _compute_cost(n, lam, dE, chi, beta, M, 1, 1, 1, 20_000)
    # print(res) # 26205, 12664955115, 2069
    print(res)  # (80098, 38711603694, 17630)
    assert np.isclose(res[0], 80098)
    assert np.isclose(res[1], 38711603694)
    assert np.isclose(res[2], 17630)

    res = _compute_cost(n, lam, dE, chi, beta, M, 3, 3, 3, 20_000)
    # print(res)  # {205788, 99457957764, 78813
    print(res)  # (270394, 130682231382, 78815)
    assert np.isclose(res[0], 270394)
    assert np.isclose(res[1], 130682231382)
    assert np.isclose(res[2], 78815)

    res = _compute_cost(n, lam, dE, chi, beta, M, 3, 5, 1, 20_000)
    # print(res)  # 151622, 73279367466, 39628
    print(res)  #  (202209, 97728216327, 77517)
    assert np.isclose(res[0], 202209)
    assert np.isclose(res[1], 97728216327)
    assert np.isclose(res[2], 77517)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_thc_resources_helper():
    lam = 307.68
    dE = 0.001
    n = 108
    chi = 10
    beta = 16
    M = 350

    res = compute_cost(
        num_spin_orbs=n,
        lambda_tot=lam,
        thc_dim=M,
        kmesh=[1, 1, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=beta,
    )
    assert np.isclose(res.toffolis_per_step, 80098)
    assert np.isclose(res.total_toffolis, 38711603694)
    assert np.isclose(res.logical_qubits, 17630)

    res = compute_cost(
        num_spin_orbs=n,
        lambda_tot=lam,
        thc_dim=M,
        kmesh=[3, 3, 3],
        dE_for_qpe=dE,
        chi=chi,
        beta=beta,
    )
    assert np.isclose(res.toffolis_per_step, 270394)
    assert np.isclose(res.total_toffolis, 130682231382)
    assert np.isclose(res.logical_qubits, 78815)

    res = compute_cost(
        num_spin_orbs=n,
        lambda_tot=lam,
        thc_dim=M,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=beta,
    )
    assert np.isclose(res.toffolis_per_step, 202209)
    assert np.isclose(res.total_toffolis, 97728216327)
    assert np.isclose(res.logical_qubits, 77517)
