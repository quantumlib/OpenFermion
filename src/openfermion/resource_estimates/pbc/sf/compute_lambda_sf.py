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

from openfermion.resource_estimates.pbc.sf.sf_integrals import SingleFactorization


@dataclass
class SFHamiltonianProperties(HamiltonianProperties):
    """Store for return values of compute_lambda function

    Extension of HamiltonianProperties dataclass to also hold the number of
    retained cholesky vectors (num_aux).
    """

    num_aux: int


def compute_lambda(hcore: npt.NDArray, sf_obj: SingleFactorization) -> SFHamiltonianProperties:
    """Lambda for single-factorized Hamiltonian.

    Compute one-body and two-body lambda for qubitization of
    single-factorized Hamiltonian.

    one-body term h_pq(k) = hcore_{pq}(k)
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
                            + sum_{Q}sum_{r}(pkqk|rQrQ)
    The first term is the kinetic energy + pseudopotential
    (or electron-nuclear), second term is from rearranging two-body operator
    into chemist charge-charge type notation, and the third is from the one body
    term obtained when squaring the two-body A and B operators.

    two-body term V = 0.5 sum_{Q}sum_{n}(A_{n}(Q)^2 +_ B_{n}(Q)^2)
    or V = 0.5 sum_{Q}sum_{n'}W_{n}(Q)^{2} where n' is twice the range of n.
    lambda is 0.5sum_{Q}sum_{n'}(sum_{p,q}^{N_{k}N/2}|Re[W_{p,q}(Q)^{n}]| +
    |Im[W_{pq}(Q)^{n}]|)^{2}

    Args:
        hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
        sf_obj: SingleFactorization integral helper object.

    Returns:
        ham_props: A HamiltonianProperties instance containing Lambda values for
            SF hamiltonian.
    """
    kpts = sf_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.0

    old_naux = sf_obj.naux  # need to reset naux for one-body computation
    sf_obj.naux = sf_obj.chol[0, 0].shape[0]

    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
            eri_kqqk_pqrs = sf_obj.get_eri([kidx, qidx, qidx, kidx])
            h1_neg -= np.einsum("prrq->pq", eri_kqqk_pqrs, optimize=True) / nkpts
            # + sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = sf_obj.get_eri([kidx, kidx, qidx, qidx])
            h1_pos += np.einsum("pqrr->pq", eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg + h1_pos
        lambda_one_body += np.sum(np.abs(one_body_mat[kidx].real) + np.abs(one_body_mat[kidx].imag))

    ############################################################################
    #
    # \lambda_{V} = \frac 12 \sum_{\Q}\sum_{n}^{M}\left
    # (\sum_{\K,pq}(|\Rea[L_{p\K,q\K-\Q,n}]| +
    #               |\Ima[L_{p\K,q\K-\Q,n}]|)\right)^{2}
    #
    # chol = [nkpts, nkpts, naux, nao, nao]
    #
    ############################################################################
    sf_obj.naux = old_naux  # reset naux to original value
    # this part needs to change
    lambda_two_body = 0.0
    for qidx in range(len(kpts)):
        # A and B are W
        A, B = sf_obj.build_AB_from_chol(qidx)  # [naux, nao * nk, nao * nk]
        A /= np.sqrt(nkpts)
        B /= np.sqrt(nkpts)
        # sum_q sum_n (sum_{pq} |Re{A_{pq}^n}| + |Im{A_{pq}^n|)^2
        lambda_two_body += np.sum(np.einsum("npq->n", np.abs(A.real) + np.abs(A.imag)) ** 2)
        lambda_two_body += np.sum(np.einsum("npq->n", np.abs(B.real) + np.abs(B.imag)) ** 2)

    lambda_two_body *= 0.5

    lambda_tot = lambda_one_body + lambda_two_body
    sf_data = SFHamiltonianProperties(
        lambda_total=lambda_tot,
        lambda_one_body=lambda_one_body,
        lambda_two_body=lambda_two_body,
        num_aux=sf_obj.naux,
    )
    return sf_data
