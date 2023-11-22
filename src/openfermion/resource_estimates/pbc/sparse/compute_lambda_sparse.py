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
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from openfermion.resource_estimates.pbc.hamiltonian import HamiltonianProperties
from openfermion.resource_estimates.pbc.sparse.sparse_integrals import SparseFactorization


@dataclass
class SparseHamiltonianProperties(HamiltonianProperties):
    """Store for return values of compute_lambda function

    Extension of HamiltonianProperties dataclass to also hold the number of
    retained matrix elements (num_sym_unique).
    """

    num_sym_unique: int


def compute_lambda(
    hcore: npt.NDArray, sparse_int_obj: SparseFactorization
) -> SparseHamiltonianProperties:
    """Compute lambda value for sparse method

    Arguments:
        hcore: array of hcore(k) by kpoint. k-point order
            is pyscf order generated for this problem.
        sparse_int_obj: The sparse integral object that is used
            to compute eris and the number of unique
            terms.

    Returns:
        ham_props: A SparseHamiltonianProperties instance containing Lambda
            values for the sparse hamiltonian.
    """
    kpts = sparse_int_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.0

    import time

    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
            eri_kqqk_pqrs = sparse_int_obj.get_eri_exact([kidx, qidx, qidx, kidx])
            h1_neg -= np.einsum("prrq->pq", eri_kqqk_pqrs, optimize=True) / nkpts
            # + sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = sparse_int_obj.get_eri_exact([kidx, kidx, qidx, qidx])

            h1_pos += np.einsum("pqrr->pq", eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg + h1_pos
        lambda_one_body += np.sum(np.abs(one_body_mat[kidx].real)) + np.sum(
            np.abs(one_body_mat[kidx].imag)
        )

    lambda_two_body = 0.0
    nkpts = len(kpts)
    # recall (k, k-q|k'-q, k')
    for kidx in range(nkpts):
        for kpidx in range(nkpts):
            for qidx in range(nkpts):
                kmq_idx = sparse_int_obj.k_transfer_map[qidx, kidx]
                kpmq_idx = sparse_int_obj.k_transfer_map[qidx, kpidx]
                test_eri_block = sparse_int_obj.get_eri([kidx, kmq_idx, kpmq_idx, kpidx]) / nkpts
                lambda_two_body += np.sum(np.abs(test_eri_block.real)) + np.sum(
                    np.abs(test_eri_block.imag)
                )

    lambda_tot = lambda_one_body + lambda_two_body
    sparse_data = SparseHamiltonianProperties(
        lambda_total=lambda_tot,
        lambda_one_body=lambda_one_body,
        lambda_two_body=lambda_two_body,
        num_sym_unique=sparse_int_obj.get_total_unique_terms_above_thresh(return_nk_counter=False),
    )
    return sparse_data
