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

from openfermion.resource_estimates.pbc.df.df_integrals import DFABKpointIntegrals
from openfermion.resource_estimates.pbc.hamiltonian import HamiltonianProperties


@dataclass
class DFHamiltonianProperties(HamiltonianProperties):
    """Store for return values of compute_lambda function

    Extension of HamiltonianProperties dataclass to also hold the number of
    retained eigenvalues (num_eig).
    """

    num_eig: int


def compute_lambda(hcore: npt.NDArray, df_obj: DFABKpointIntegrals) -> DFHamiltonianProperties:
    """Compute lambda for double-factorized Hamiltonian.

    one-body term h_pq(k) = hcore_{pq}(k)
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
                            + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
    The first term is the kinetic energy + pseudopotential (or electron-nuclear)
    second term is from rearranging two-body operator into chemist charge-charge
    type notation, and the third is from the one body term obtained when
    squaring the two-body A and B operators.

    Arguments:
        hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
        df_obj: DFABKpointIntegrals integral helper.

    Returns:
        ham_props: A HamiltonianProperties instance containing Lambda values for
            DF hamiltonian.
    """
    kpts = df_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.0

    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
            eri_kqqk_pqrs = df_obj.get_eri_exact([kidx, qidx, qidx, kidx])
            h1_neg -= np.einsum("prrq->pq", eri_kqqk_pqrs, optimize=True) / nkpts
            # + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = df_obj.get_eri_exact([kidx, kidx, qidx, qidx])
            h1_pos += np.einsum("pqrr->pq", eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg + h1_pos
        one_eigs, _ = np.linalg.eigh(one_body_mat[kidx])
        lambda_one_body += np.sum(np.abs(one_eigs))

    lambda_two_body = 0.0
    num_eigs = 0
    for qidx in range(len(kpts)):
        for nn in range(df_obj.naux):
            first_number_to_square = 0
            second_number_to_square = 0
            # sum up p,k eigenvalues
            for kidx in range(len(kpts)):
                # A and B are W
                if df_obj.amat_lambda_vecs[kidx, qidx, nn] is None:
                    continue
                eigs_a_fixed_n_q = df_obj.amat_lambda_vecs[kidx, qidx, nn] / np.sqrt(nkpts)
                eigs_b_fixed_n_q = df_obj.bmat_lambda_vecs[kidx, qidx, nn] / np.sqrt(nkpts)
                first_number_to_square += np.sum(np.abs(eigs_a_fixed_n_q))
                num_eigs += len(eigs_a_fixed_n_q)
                if eigs_b_fixed_n_q is not None:
                    second_number_to_square += np.sum(np.abs(eigs_b_fixed_n_q))
                    num_eigs += len(eigs_b_fixed_n_q)

            lambda_two_body += first_number_to_square**2
            lambda_two_body += second_number_to_square**2

    lambda_two_body *= 0.25

    lambda_tot = lambda_one_body + lambda_two_body
    df_data = DFHamiltonianProperties(
        lambda_total=lambda_tot,
        lambda_one_body=lambda_one_body,
        lambda_two_body=lambda_two_body,
        num_eig=num_eigs,
    )
    return df_data
