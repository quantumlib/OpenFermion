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
    from openfermion.resource_estimates.pbc.df.compute_df_resources import (
        _compute_cost,
        compute_cost,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_costing():
    nRe = 108
    lamRe = 294.8
    dE = 0.001
    LRe = 360
    LxiRe = 13031
    chi = 10
    betaRe = 16

    # (*The Li et al orbitals.*)
    nLi = 152
    lamLi = 1171.2
    LLi = 394
    LxiLi = 20115
    betaLi = 20

    res = _compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 2, 2, 2, 20_000)
    res = _compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 2, 2, 2, res[0])
    # 48250, 22343175750, 8174
    assert np.isclose(res[0], 48250)
    assert np.isclose(res[1], 22343175750)
    assert np.isclose(res[2], 8174)

    res = _compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 3, 5, 1, 20_000)
    res = _compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 3, 5, 1, res[0])
    # 53146, 24610371366, 8945
    assert np.isclose(res[0], 53146)
    assert np.isclose(res[1], 24610371366)
    assert np.isclose(res[2], 8945)

    res = _compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 2, 2, 2, 20_000)
    res = _compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 2, 2, 2, res[0])
    # print(res) # 79212, 145727663004, 13873
    assert np.isclose(res[0], 79212)
    assert np.isclose(res[1], 145727663004)
    assert np.isclose(res[2], 13873)
    res = _compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 3, 5, 1, res[0])
    # print(res) # 86042, 158292930114, 14952
    assert np.isclose(res[0], 86042)
    assert np.isclose(res[1], 158292930114)
    assert np.isclose(res[2], 14952)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_costing_helper():
    nRe = 108
    lamRe = 294.8
    dE = 0.001
    LRe = 360
    LxiRe = 13031
    chi = 10
    betaRe = 16

    # (*The Li et al orbitals.*)
    nLi = 152
    lamLi = 1171.2
    LLi = 394
    LxiLi = 20115
    betaLi = 20

    res = compute_cost(
        num_spin_orbs=nRe,
        lambda_tot=lamRe,
        num_aux=LRe,
        num_eig=LxiRe,
        kmesh=[2, 2, 2],
        dE_for_qpe=dE,
        chi=chi,
        beta=betaRe,
    )
    # 48250, 22343175750, 8174
    assert np.isclose(res.toffolis_per_step, 48250)
    assert np.isclose(res.total_toffolis, 22343175750)
    assert np.isclose(res.logical_qubits, 8174)

    res = compute_cost(
        num_spin_orbs=nRe,
        lambda_tot=lamRe,
        num_aux=LRe,
        num_eig=LxiRe,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=betaRe,
    )
    # 53146, 24610371366, 8945
    assert np.isclose(res.toffolis_per_step, 53146)
    assert np.isclose(res.total_toffolis, 24610371366)
    assert np.isclose(res.logical_qubits, 8945)

    res = compute_cost(
        num_spin_orbs=nLi,
        lambda_tot=lamLi,
        num_aux=LLi,
        num_eig=LxiLi,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=betaLi,
    )
    # print(res) # 79212, 145727663004, 13873
    assert np.isclose(res.toffolis_per_step, 86042)
    assert np.isclose(res.total_toffolis, 158292930114)
    assert np.isclose(res.logical_qubits, 14952)


if __name__ == "__main__":
    test_costing()
