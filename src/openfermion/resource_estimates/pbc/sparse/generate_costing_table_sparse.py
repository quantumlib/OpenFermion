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
import pandas as pd

import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from openfermion.resource_estimates.pbc.resources import PBCResources
from openfermion.resource_estimates.pbc.sparse.sparse_integrals import SparseFactorization
from openfermion.resource_estimates.pbc.hamiltonian import build_hamiltonian
from openfermion.resource_estimates.pbc.hamiltonian.cc_extensions import (
    build_approximate_eris,
    build_cc_inst,
    build_approximate_eris_rohf,
)
from openfermion.resource_estimates.pbc.sparse.compute_lambda_sparse import compute_lambda
from openfermion.resource_estimates.pbc.sparse.compute_sparse_resources import compute_cost


def generate_costing_table(
    pyscf_mf: scf.HF,
    name="pbc",
    chi: int = 10,
    thresholds: np.ndarray = np.logspace(-1, -5, 6),
    dE_for_qpe=0.0016,
    energy_method="MP2",
) -> pd.DataFrame:
    """Generate resource estimate costing table given a set of cutoffs for
        sparse Hamiltonian.

    Arguments:
        pyscf_mf: k-point pyscf mean-field object
        name: Optional descriptive name for simulation.
        chi: the number of bits for the representation of the coefficients
        thresholds: Array of sparse thresholds to generate the table for.
        dE_for_qpe: Phase estimation epsilon.

    Returns
        resources: Table of resource estimates.
    """
    kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)
    cc_inst = build_cc_inst(pyscf_mf)
    exact_eris = cc_inst.ao2mo()
    if energy_method == "MP2":
        energy_function = lambda x: cc_inst.init_amps(x)
        reference_energy, _, _ = energy_function(exact_eris)
    elif energy_method == "CCSD":
        energy_function = lambda x: cc_inst.kernel(eris=x)
        reference_energy, _, _ = energy_function(exact_eris)
    else:
        raise ValueError(f"Unknown value for energy_method: {energy_method}")

    hcore, chol = build_hamiltonian(pyscf_mf)
    num_spin_orbs = 2 * hcore[0].shape[-1]

    num_kpts = np.prod(kmesh)

    sparse_resource_obj = PBCResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        energy_method=energy_method,
        exact_energy=np.real(reference_energy),
    )
    approx_eris = exact_eris
    for thresh in thresholds:
        sparse_helper = SparseFactorization(cholesky_factor=chol, kmf=pyscf_mf, threshold=thresh)
        if pyscf_mf.cell.spin == 0:
            approx_eris = build_approximate_eris(cc_inst, sparse_helper, eris=approx_eris)
        else:
            approx_eris = build_approximate_eris_rohf(cc_inst, sparse_helper, eris=approx_eris)
        approx_energy, _, _ = energy_function(approx_eris)

        sparse_data = compute_lambda(hcore, sparse_helper)

        sparse_res_cost = compute_cost(
            num_spin_orbs,
            sparse_data.lambda_total,
            sparse_data.num_sym_unique,
            list(kmesh),
            dE_for_qpe=dE_for_qpe,
            chi=chi,
        )
        sparse_resource_obj.add_resources(
            ham_properties=sparse_data,
            resource_estimates=sparse_res_cost,
            cutoff=thresh,
            approx_energy=np.real(approx_energy),
        )

    return sparse_resource_obj.to_dataframe()
