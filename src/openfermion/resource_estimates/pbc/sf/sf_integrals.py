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
import itertools
import numpy as np
import numpy.typing as npt

from pyscf.pbc import scf

from openfermion.resource_estimates.pbc.hamiltonian import build_momentum_transfer_mapping


# Single-Factorization
class SingleFactorization:
    def __init__(self, cholesky_factor: npt.NDArray, kmf: scf.HF, naux: int = None):
        """Class defining single-factorized ERIs.

        Args:
            cholesky_factor: Cholesky factor tensor that is
                [nkpts, nkpts, naux, nao, nao].
                To see how to generate this see cholesky_from_df_ints
            kmf: pyscf k-object.  Currently only used to obtain the number of
                k-points.  Must have an attribute kpts which len(self.kmf.kpts)
                returns number of kpts.
        """
        self.chol = cholesky_factor
        self.kmf = kmf
        self.nk = len(self.kmf.kpts)
        if naux is None:
            naux = cholesky_factor[0, 0].shape[0]
        self.naux = naux
        self.nao = cholesky_factor[0, 0].shape[-1]
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map

    def build_AB_from_chol(self, qidx: int):
        """Construct A and B matrices given Q-kpt idx.

        This constructs all matrices association with n-chol

        Args:
            qidx: index for momentum mode Q.

        Returns:
            A:  A matrix of size [naux, nmo * kpts, nmk * kpts]
            B:  B matrix of size [naux, nmo * kpts, nmk * kpts]
        """
        nmo = self.nao  # number of gaussians used
        naux = self.naux

        # now form A and B
        # First set up matrices to store. We will loop over Q index and
        # construct entire set of matrices index by n-aux.
        rho = np.zeros((naux, nmo * self.nk, nmo * self.nk), dtype=np.complex128)

        for kidx in range(self.nk):
            k_minus_q_idx = self.k_transfer_map[qidx, kidx]
            for p, q in itertools.product(range(nmo), repeat=2):
                P = int(kidx * nmo + p)  # a^_{pK}
                Q = int(k_minus_q_idx * nmo + q)  # a_{q(K-Q)}
                rho[:, P, Q] += self.chol[kidx, k_minus_q_idx][
                    :naux, p, q
                ]  # L_{pK, q(K-Q)}a^_{pK}a_{q(K-Q)}

        A = 0.5 * (rho + rho.transpose((0, 2, 1)).conj())
        B = 0.5j * (rho - rho.transpose((0, 2, 1)).conj())

        return A, B

    def get_eri(self, ikpts: list, check_eq=False):
        r"""Construct (pkp qkq| rkr sks) via cholesky factors.

        Note: 3-tensor L_{sks, rkr} = L_{rkr, sks}^{*}

        Args:
            ikpts: list of four integers representing the index of the kpoint
                in self.kmf.kpts
            check_eq: optional value to confirm a symmetry in the Cholesky
                vectors (Default value = False)

        Returns:
            eri: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, ikr, iks = ikpts
        n = self.naux
        naux_pq = self.chol[ikp, ikq].shape[0]
        if n > naux_pq:
            print(f"WARNING: specified naux ({n}) is too large!")
            n = naux_pq
        if check_eq:
            assert np.allclose(
                np.einsum(
                    "npq,nsr->pqrs",
                    self.chol[ikp, ikq][:n],
                    self.chol[iks, ikr][:n].conj(),
                    optimize=True,
                ),
                np.einsum(
                    "npq,nrs->pqrs", self.chol[ikp, ikq][:n], self.chol[ikr, iks][:n], optimize=True
                ),
            )
        return np.einsum(
            "npq,nsr->pqrs", self.chol[ikp, ikq][:n], self.chol[iks, ikr][:n].conj(), optimize=True
        )
