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
from typing import Union, Tuple
import itertools
import numpy as np
import numpy.typing as npt

from pyscf.pbc import scf
from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk

from openfermion.resource_estimates.pbc.hamiltonian import build_momentum_transfer_mapping


def _symmetric_two_body_terms(quad: Tuple[int, ...], complex_valued: bool):
    p, q, r, s = quad
    yield p, q, r, s
    yield q, p, s, r
    yield s, r, q, p
    yield r, s, p, q
    if not complex_valued:
        yield p, s, r, q
        yield q, r, s, p
        yield s, p, q, r
        yield r, q, p, s


def unique_iter(nmo: int):
    """Iterate over unique pqrs indices"""
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_symmetric_two_body_terms(quad, True))
            yield tuple(quad)


def _pq_rs_two_body_terms(quad):
    """Symmetry inequivalent indices when kp = kq and kr = ks.

    A subset of the four-fold symmetry can be applied
    (pkp, qkp|rkr, skr) = (qkp,pkp|skr,rkr) by complex conjucation
    """
    p, q, r, s = quad
    yield p, q, r, s
    yield q, p, s, r


def unique_iter_pq_rs(nmo: int):
    """Iterate over unique symmetry pqrs indices"""
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_pq_rs_two_body_terms(quad))
            yield tuple(quad)


def _ps_qr_two_body_terms(quad):
    """Symmetry inequivalent indices when kp = ks and kq = kr.

    A subset of the four-fold symmetry can be applied (pkp,qkq|rkq,skp) ->
    (skp,rkq|qkq,pkp) by complex conj and dummy index exchange
    """
    p, q, r, s = quad
    yield p, q, r, s
    yield s, r, q, p


def unique_iter_ps_qr(nmo):
    """Iterate over unique ps qr indices"""
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_ps_qr_two_body_terms(quad))
            yield tuple(quad)


def _pr_qs_two_body_terms(quad):
    """Symmetry inequivalent indices when kp = kr and kq = ks.

    A subset of the four-fold symmetry can be applied (pkp,qkq|rkp,skq) ->
    (rkp,skq|pkp,rkq) by dummy index exchange
    """
    p, q, r, s = quad
    yield p, q, r, s
    yield r, s, p, q


def unique_iter_pr_qs(nmo):
    """Iterate over unique pr qs indices"""
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_pr_qs_two_body_terms(quad))
            yield tuple(quad)


class SparseFactorization:
    def __init__(self, cholesky_factor: np.ndarray, kmf: scf.HF, threshold=1.0e-14):
        """Initialize a ERI object for CCSD from sparse integrals.

        Args:
            cholesky_factor: Cholesky factor tensor that is
                [nkpts, nkpts, naux, nao, nao].
                To see how to generate this see pyscf_chol_from_df
            kmf: pyscf k-object.  Currently only used to obtain the number of
                k-points.  Must have an attribute kpts which len(self.kmf.kpts)
                returns number of kpts.
            threshold: Default 1.0E-8 is the value at which to ignore the
                integral
        """
        self.chol = cholesky_factor
        self.kmf = kmf
        self.nk = len(self.kmf.kpts)
        self.nao = cholesky_factor[0, 0].shape[-1]
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map
        self.threshold = threshold

    def get_total_unique_terms_above_thresh(
        self, return_nk_counter=False
    ) -> Union[int, Tuple[int, int]]:
        """Determine all unique (pkp, qkq|rkr, sks).

        Accounts for momentum conservation and four fold symmetry.

        Args:
            return_nk_counter: Return number of visited k-points.

        Returns:
            num_sym_unique: Number of symmetry unique terms above the threshold
        """
        kpts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts)
        nkpts = len(self.kmf.kpts)
        completed = np.zeros((nkpts, nkpts, nkpts), dtype=bool)
        counter = 0
        nk_counter = 0
        for kvals in loop_kkk(nkpts):
            kp, kq, kr = kvals
            ks = kpts_helper.kconserv[kp, kq, kr]
            if not completed[kp, kq, kr]:
                nk_counter += 1
                eri_block = self.get_eri([kp, kq, kr, ks])
                if kp == kq == kr == ks:
                    completed[kp, kq, kr] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dd = np.zeros((n,), dtype=eri_block.dtype)
                    for p in range(n):
                        Dd[p] = eri_block[p, p, p, p]
                        eri_block[p, p, p, p] = 0.0
                    Dp = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dp[p, q] = eri_block[p, q, p, q]
                        eri_block[p, q, p, q] = 0.0
                    Dc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, r in itertools.product(range(n), repeat=2):
                        Dc[p, r] = eri_block[p, p, r, r]
                        eri_block[p, p, r, r] = 0.0
                    Dpc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dpc[p, q] = eri_block[p, q, q, p]
                        eri_block[p, q, q, p] = 0.0
                    counter += np.count_nonzero(Dd)
                    counter += np.count_nonzero(Dp) // 2
                    counter += np.count_nonzero(Dc) // 2
                    counter += np.count_nonzero(Dpc) // 2
                    counter += np.count_nonzero(eri_block) // 4
                elif kp == kq and kr == ks:
                    completed[kp, kq, kr] = True
                    completed[kr, ks, kp] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, r in itertools.product(range(n), repeat=2):
                        Dc[p, r] = eri_block[p, p, r, r]
                        eri_block[p, p, r, r] = 0.0
                    counter += np.count_nonzero(Dc)
                    counter += np.count_nonzero(eri_block) // 2
                elif kp == ks and kq == kr:
                    completed[kp, kq, kr] = True
                    completed[kr, ks, kp] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dpc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dpc[p, q] = eri_block[p, q, q, p]
                        eri_block[p, q, q, p] = 0.0
                    counter += np.count_nonzero(Dpc)
                    counter += np.count_nonzero(eri_block) // 2
                elif kp == kr and kq == ks:
                    completed[kp, kq, kr] = True
                    completed[kq, kp, ks] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dp = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dp[p, q] = eri_block[p, q, p, q]
                        eri_block[p, q, p, q] = 0.0
                    counter += np.count_nonzero(Dp)
                    counter += np.count_nonzero(eri_block) // 2
                else:
                    counter += np.count_nonzero(eri_block)
                    completed[kp, kq, kr] = True
                    completed[kr, ks, kp] = True
                    completed[kq, kp, ks] = True
                    completed[ks, kr, kq] = True
        if return_nk_counter:
            return counter, nk_counter
        return counter

    def get_eri(self, ikpts, check_eq=False) -> npt.NDArray:
        """Construct (pkp qkq| rkr sks) via  cholesky factors.

        Note: 3-tensor L_{sks, rkr} = L_{rkr, sks}^{*}

        Args:
            ikpts: list of four integers representing the index of the kpoint in
                self.kmf.kpts
            check_eq: optional value to confirm a symmetry in the Cholesky
                vectors. (Default value = False)

        Returns:
            eris: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, ikr, iks = ikpts
        if check_eq:
            assert np.allclose(
                np.einsum(
                    "npq,nsr->pqrs", self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True
                ),
                np.einsum("npq,nrs->pqrs", self.chol[ikp, ikq], self.chol[ikr, iks], optimize=True),
            )
        eri = np.einsum(
            "npq,nsr->pqrs", self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True
        )
        zero_mask = np.abs(eri) < self.threshold
        eri[zero_mask] = 0
        return eri

    def get_eri_exact(self, kpts) -> npt.NDArray:
        """Construct (pkp qkq| rkr sks) exactly from mean-field object.

        This is for constructing the J and K like terms needed for the
        one-body component lambda.

        Args:
            kpts: list of four integers representing the index of the kpoint
                in self.kmf.kpts

        Returns:
            eris: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, ikr, iks = kpts
        return np.einsum(
            "npq,nsr->pqrs", self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True
        )
