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

import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from openfermion.resource_estimates.pbc.sparse.compute_sparse_resources import (
        _compute_cost,
        compute_cost,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_compute_cost():
    nRe = 108
    lam_re = 2135.3
    dRe = 705831
    dE = 0.001
    chi = 10

    nLi = 152
    lam_Li = 1547.3
    dLi = 440501
    dE = 0.001
    chi = 10

    Nkx = 2
    Nky = 2
    Nkz = 2

    res = _compute_cost(nRe, lam_re, dRe, dE, chi, 20_000, Nkx, Nky, Nkz)
    # res = _compute_cost(nRe, lam_re, dRe, dE, chi, res[0], Nkx, Nky, Nkz)
    assert res[0] == 22962
    assert res[1] == 77017349364
    assert res[2] == 11335

    res = _compute_cost(nRe, lam_re, dRe, dE, chi, 20_000, 3, 5, 1)
    assert res[0] == 29004
    assert res[1] == 97282954488
    assert res[2] == 8060

    res = _compute_cost(nLi, lam_Li, dLi, dE, chi, 20_000, Nkx, Nky, Nkz)
    res = _compute_cost(nLi, lam_Li, dLi, dE, chi, res[0], Nkx, Nky, Nkz)
    assert res[0] == 21426
    assert res[1] == 52075764444
    assert res[2] == 7015

    res = _compute_cost(nLi, lam_Li, dLi, dE, chi, 20_000, 3, 5, 1)
    assert res[0] == 28986
    assert res[1] == 70450299084
    assert res[2] == 9231


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_compute_cost_helper():
    dE = 0.001
    chi = 10

    nLi = 152
    lam_Li = 1547.3
    dLi = 440501
    dE = 0.001
    chi = 10

    Nkx = 2
    Nky = 2
    Nkz = 2

    res = compute_cost(
        num_spin_orbs=nLi,
        lambda_tot=lam_Li,
        num_sym_unique=dLi,
        kmesh=[Nkx, Nky, Nkz],
        dE_for_qpe=dE,
        chi=chi,
    )
    assert res.toffolis_per_step == 21426
    assert res.total_toffolis == 52075764444
    assert res.logical_qubits == 7015


if __name__ == "__main__":
    test_compute_cost()
