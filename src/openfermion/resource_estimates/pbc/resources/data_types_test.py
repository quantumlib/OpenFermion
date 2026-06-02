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
    from openfermion.resource_estimates.pbc.resources.data_types import (
        PBCResources,
        ResourceEstimates,
    )
    from openfermion.resource_estimates.pbc.hamiltonian import HamiltonianProperties


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_pbc_resources():
    nmo = 10
    np.random.seed(7)
    pbc_resources = PBCResources(
        "pbc", num_spin_orbitals=nmo, num_kpts=12, dE=1e-7, chi=13, exact_energy=-13.3
    )
    for cutoff in np.logspace(-1, -3, 5):
        lv = np.random.random(3)
        lambdas = HamiltonianProperties(
            lambda_total=lv[0], lambda_one_body=lv[1], lambda_two_body=lv[2]
        )
        resource = ResourceEstimates(
            toffolis_per_step=np.random.randint(0, 1000),
            total_toffolis=np.random.randint(0, 1000),
            logical_qubits=13,
        )
        pbc_resources.add_resources(
            ham_properties=lambdas, resource_estimates=resource, approx_energy=-12, cutoff=cutoff
        )
    df = pbc_resources.to_dataframe()
    assert np.allclose(
        df.lambda_total.values, [0.07630829, 0.97798951, 0.26843898, 0.38094113, 0.21338535]
    )
    assert (df.toffolis_per_step.values == [919, 366, 895, 787, 949]).all()
    assert (df.total_toffolis.values == [615, 554, 391, 444, 112]).all()
