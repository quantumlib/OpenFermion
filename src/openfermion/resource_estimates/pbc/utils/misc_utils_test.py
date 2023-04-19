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

from openfermion.resource_estimates.pbc.utils.resource_utils import (
    PBCResources,
    ResourceEstimates,
)
from openfermion.resource_estimates.pbc.utils.hamiltonian_utils import (
    HamiltonianProperties,
    build_momentum_transfer_mapping,
)


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
            ham_properties=lambdas,
            resource_estimates=resource,
            approx_energy=-12,
            cutoff=cutoff,
        )
    df = pbc_resources.to_dataframe()
    assert np.allclose(
        df.lambda_total.values,
        [0.07630829, 0.97798951, 0.26843898, 0.38094113, 0.21338535],
    )
    assert (df.toffolis_per_step.values == [919, 366, 895, 787, 949]).all()
    assert (df.total_toffolis.values == [615, 554, 391, 444, 112]).all()


def test_momentum_transfer_map():
    from pyscf.pbc import gto

    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 0
    cell.build(parse_arg=False)
    kpts = cell.make_kpts([2, 2, 1], scaled_center=[0.1, 0.2, 0.3])
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    for i, Q in enumerate(kpts):
        for j, k1 in enumerate(kpts):
            k2 = kpts[mom_map[i, j]]
            test = Q - k1 + k2
            assert np.amin(np.abs(test[None, :] - cell.Gv - kpts[0][None, :])) < 1e-15
