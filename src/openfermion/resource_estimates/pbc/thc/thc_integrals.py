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
from typing import Union
import numpy as np
import numpy.typing as npt

from pyscf.pbc import scf
from pyscf.pbc.lib.kpts_helper import unique, member

from openfermion.resource_estimates.pbc.hamiltonian import build_momentum_transfer_mapping

from openfermion.resource_estimates.pbc.thc.factorizations.isdf import (
    build_g_vector_mappings_double_translation,
    build_g_vector_mappings_single_translation,
    build_eri_isdf_double_translation,
    build_eri_isdf_single_translation,
)


class KPTHCDoubleTranslation:
    def __init__(
        self,
        chi: npt.NDArray,
        zeta: npt.NDArray,
        kmf: scf.HF,
        chol: Union[npt.NDArray, None] = None,
    ):
        """Class for constructing THC factorized ERIs.

        Assumes 2 G vectors to index THC factors (i.e. `double translation.')

        Arguments:
            chi: array of interpolating orbitals of shape
                [num_kpts, num_mo, num_interp_points]
            zeta: central tensor of dimension
                [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
            kmf: pyscf k-object.  Currently only used to obtain the number of
                k-points.  must have an attribute kpts which len(self.kmf.kpts)
                returns number of kpts.
            cholesky_factor: Cholesky object for computing exact integrals
        """
        self.chi = chi
        self.zeta = zeta
        self.kmf = kmf
        self.nk = len(self.kmf.kpts)
        self.kpts = self.kmf.kpts
        self.k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.reverse_k_transfer_map = np.zeros_like(self.k_transfer_map)  # [kidx, kmq_idx] = qidx
        for kidx in range(self.nk):
            for qidx in range(self.nk):
                kmq_idx = self.k_transfer_map[qidx, kidx]
                self.reverse_k_transfer_map[kidx, kmq_idx] = qidx
        # Two-translation ISDF zeta[iq, dG, dG']
        _, _, g_map_unique, _ = build_g_vector_mappings_double_translation(
            self.kmf.cell, self.kmf.kpts, self.k_transfer_map
        )
        self.g_mapping = g_map_unique
        self.chol = chol

    def get_eri(self, ikpts: list) -> npt.NDArray:
        r"""Construct ERIs given kpt indices.

        .. math::

            (pkp qkq| rkr sks) = \\sum_{mu nu} zeta[iq, dG, dG', mu, nu]
            chi[kp,p,mu]* chi[kq,q,mu] chi[kp,p,nu]* chi[ks,s,nu]

        Arguments:
            ikpts: list of four integers representing the index of the kpoint in
                self.kmf.kpts

        Returns:
            eris: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, _, _ = ikpts
        q_indx = self.reverse_k_transfer_map[ikp, ikq]
        return build_eri_isdf_double_translation(self.chi, self.zeta, q_indx, ikpts, self.g_mapping)

    def get_eri_exact(self, kpts: list) -> npt.NDArray:
        """Construct (pkp qkq| rkr sks) exactly from cholesky factors.

        Arguments:
            kpts: list of four integers representing the index of the kpoint in
                self.kmf.kpts

        Returns:
            eris: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, ikr, iks = kpts
        if self.chol is not None:
            return np.einsum(
                "npq,nsr->pqrs", self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True
            )
        else:
            eri_kpt = self.kmf.with_df.ao2mo(
                [self.kmf.mo_coeff[i] for i in (ikp, ikq, ikr, iks)],
                [self.kmf.kpts[i] for i in (ikp, ikq, ikr, iks)],
                compact=False,
            )
            shape_pqrs = [self.kmf.mo_coeff[i].shape[-1] for i in (ikp, ikq, ikr, iks)]
            eri_kpt = eri_kpt.reshape(shape_pqrs)
        return eri_kpt


class KPTHCSingleTranslation(KPTHCDoubleTranslation):
    def __init__(self, chi: npt.NDArray, zeta: npt.NDArray, kmf: scf.HF):
        """Class for constructing THC factorized ERIs.

        Assumes one delta G (i.e. a single translation vector.)

        Arguments:
            chi: array of interpolating orbitals of shape
                [num_kpts, num_mo, num_interp_points]
            zeta: central tensor of dimension
                [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
            kmf: pyscf k-object.  Currently only used to obtain the number of
                k-points. must have an attribute kpts which len(self.kmf.kpts)
                returns number of kpts.
            cholesky_factor: Cholesky object for computing exact integrals
        """
        super().__init__(chi, zeta, kmf)
        # one-translation ISDF zeta[iq, dG]
        num_kpts = len(self.kmf.kpts)
        kpts = self.kmf.kpts
        kpts_pq = np.array(
            [(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
        )
        kpts_pq_indx = np.array(
            [(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
        )
        transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
        _, unique_indx, _ = unique(transfers)
        (_, _, g_map_unique, _) = build_g_vector_mappings_single_translation(
            kmf.cell, kpts, kpts_pq_indx[unique_indx]
        )
        self.g_mapping = g_map_unique
        self.momentum_transfers = transfers[unique_indx]

    def get_eri(self, ikpts):
        """Construct ERIs from kpt indices

        Evaluated via

        .. math::

            (pkp qkq| rkr sks) = \\sum_{mu nu} zeta[iq, dG, mu, nu]
                chi[kp,p,mu]* chi[kq,q,mu] chi[kp,p,nu]* chi[ks,s,nu]

        Arguments:
            kpts: list of four integers representing the index of the kpoint in
                self.kmf.kpts

        Returns:
            eris: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, _, _ = ikpts
        mom_transfer = self.kpts[ikp] - self.kpts[ikq]
        q_indx = member(mom_transfer, self.momentum_transfers)[0]
        return build_eri_isdf_single_translation(self.chi, self.zeta, q_indx, ikpts, self.g_mapping)

    def get_eri_exact(self, kpts):
        """Construct (pkp qkq| rkr sks) exactly from cholesky factors.

        Arguments:
            kpts: list of four integers representing the index of the kpoint in
                self.kmf.kpts

        Returns:
            eris: ([pkp][qkq]|[rkr][sks])
        """
        ikp, ikq, ikr, iks = kpts
        if self.chol is not None:
            return np.einsum(
                "npq,nsr->pqrs", self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True
            )
        else:
            eri_kpt = self.kmf.with_df.ao2mo(
                [self.kmf.mo_coeff[i] for i in (ikp, ikq, ikr, iks)],
                [self.kmf.kpts[i] for i in (ikp, ikq, ikr, iks)],
                compact=False,
            )
            shape_pqrs = [self.kmf.mo_coeff[i].shape[-1] for i in (ikp, ikq, ikr, iks)]
            eri_kpt = eri_kpt.reshape(shape_pqrs)
        return eri_kpt
