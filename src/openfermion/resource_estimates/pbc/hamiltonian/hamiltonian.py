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
from dataclasses import dataclass, asdict
from typing import Tuple
import h5py
import numpy as np
import numpy.typing as npt

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc.df import df
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc import mp, scf, gto


@dataclass
class HamiltonianProperties:
    """Lighweight descriptive data class to hold return values from
    compute_lambda functions.

    Attributes:
        lambda_total: Total lambda value (norm) of Hamiltonian.
        lambda_one_body: One-body lambda value (norm) of Hamiltonian.
        lambda_two_body: Two-body lambda value (norm) of Hamiltonian.
    """

    lambda_total: float
    lambda_one_body: float
    lambda_two_body: float

    dict = asdict


def build_hamiltonian(mf: "scf.KRHF") -> Tuple[npt.NDArray, npt.NDArray]:
    """Utility function to build one- and two-electron matrix elements from mean
    field object.

    Arguments:
        mf: pyscf KRHF object.

    Returns:
        hcore_mo: one-body Hamiltonian in MO basis.
        chol: 3-index RSGDF density fitted integrals.
    """
    # Build temporary mp2 object so MO coeffs can potentially be padded if mean
    # field solution yields different number of MOs per k-point.
    tmp_mp2 = mp.KMP2(mf)
    mo_coeff_padded = _add_padding(tmp_mp2, tmp_mp2.mo_coeff, tmp_mp2.mo_energy)[0]
    hcore_mo = np.asarray([C.conj().T @ hk @ C for (C, hk) in zip(mo_coeff_padded, mf.get_hcore())])
    chol = cholesky_from_df_ints(tmp_mp2)
    return hcore_mo, chol


def cholesky_from_df_ints(mp2_inst, pad_mos_with_zeros=True) -> npt.NDArray:
    """Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and
    virtual canonical crystalline orbitals. Note that `o` and `v` contain kpt
    indices `ko` and `kv`, and the third kpt index `kL` is determined by
    the conservation of momentum.

    Note that if the number of mos differs at each k-point then this function
    will pad MOs with zeros to ensure contiguity.

    Args:
        mp2_inst: pyscf KMP2 instance.

    Returns:
        Lchol: 3-center DF ints, with shape (nkpts, nkpts, naux, nmo, nmo)
    """

    log = logger.Logger(mp2_inst.stdout, mp2_inst.verbose)

    if mp2_inst._scf.with_df._cderi is None:
        mp2_inst._scf.with_df.build()

    cell = mp2_inst._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    # nvir = nmo - nocc
    nao = cell.nao_nr()

    mo_coeff = mp2_inst._scf.mo_coeff
    kpts = mp2_inst.kpts
    if pad_mos_with_zeros:
        mo_coeff = _add_padding(mp2_inst, mp2_inst.mo_coeff, mp2_inst.mo_energy)[0]
        nmo = mp2_inst.nmo
    else:
        nmo = nao
        num_mo_per_kpt = np.array([C.shape[-1] for C in mo_coeff])
        if not (num_mo_per_kpt == nmo).all():
            log.info(
                "Number of MOs differs at each k-point or is not the same " "as the number of AOs."
            )
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    Lchol = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())

    bra_start = 0
    bra_end = nmo
    ket_start = nmo
    ket_end = 2 * nmo
    with h5py.File(mp2_inst._scf.with_df._cderi, "r") as f:
        kptij_lst = f["j3c-kptij"][:]
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq_ao = np.asarray(df._getitem(f, "j3c", kpti_kptj, kptij_lst))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order="F")
                if dtype == np.double:
                    out = _ao2mo.nr_e2(
                        Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym="s2"
                    )
                else:
                    # Note: Lpq.shape[0] != naux if linear dependency is found
                    # in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(
                        Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc
                    )
                Lchol[ki, kj] = out.reshape(-1, nmo, nmo)

    log.timer_debug1("transforming DF-AO integrals to MO", *cput0)

    return Lchol


def build_momentum_transfer_mapping(cell: gto.Cell, kpoints: np.ndarray) -> np.ndarray:
    # Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    # k1 - k2 + G = Q.
    a = cell.lattice_vectors() / (2 * np.pi)
    delta_k1_k2_Q = (
        kpoints[:, None, None, :] - kpoints[None, :, None, :] - kpoints[None, None, :, :]
    )
    delta_k1_k2_Q += kpoints[0][None, None, None, :]  # shift to center
    delta_dot_a = np.einsum("wx,kpQx->kpQw", a, delta_k1_k2_Q)
    int_delta_dot_a = np.rint(delta_dot_a)
    # Should be zero if transfer is statisfied (2*pi*n)
    mapping = np.where(np.sum(np.abs(delta_dot_a - int_delta_dot_a), axis=3) < 1e-10)
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,) * 2, dtype=np.int32)
    # Note index flip due to Q being first index in map but broadcasted last..
    momentum_transfer_map[mapping[1], mapping[0]] = mapping[2]

    return momentum_transfer_map
