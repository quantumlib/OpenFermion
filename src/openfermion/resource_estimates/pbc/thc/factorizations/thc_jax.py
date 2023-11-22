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
"""Reoptimize THC factors using a combination of BFGS and adaagrad including a
penalty parameter to regularize the lambda value as described in
https://arxiv.org/abs/2302.05531

The entrypoint for a user should be the function kpoint_thc_via_isdf which can
be used as

>>> krhf_inst = scf.KRHF(cell, kpts)
>>> krmp_inst = scf.KMP2(krhf_inst)
>>> Lchol = pyscf_chol_from_df(krmp_inst)
>>> nthc = cthc * nmo
>>> thc_factors = kpoint_thc_via_isdf(krhf_inst, Lchol, nthc)
"""

# pylint: disable=wrong-import-position
import math
import time
from typing import Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from pyscf.pbc import scf
from scipy.optimize import minimize

from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.typing as jnpt

from openfermion.resource_estimates.thc.utils import adagrad
from openfermion.resource_estimates.pbc.thc.factorizations.isdf import (
    KPointTHC,
    solve_kmeans_kpisdf,
)
from openfermion.resource_estimates.pbc.hamiltonian import build_momentum_transfer_mapping


def load_thc_factors(chkfile_name: str) -> KPointTHC:
    """Load THC factors from a checkpoint file

    Args:
        chkfile_name: Filename containing THC factors.

    Returns:
        kthc: KPointISDF object built from chkfile_name.
    """
    xi = None
    with h5py.File(chkfile_name, "r") as fh5:
        chi = fh5["chi"][:]
        g_mapping = fh5["g_mapping"][:]
        num_kpts = g_mapping.shape[0]
        zeta = np.zeros((num_kpts,), dtype=object)
        if "xi" in list(fh5.keys()):
            xi = fh5["xi"][:]
        else:
            xi = None
        for iq in range(g_mapping.shape[0]):
            zeta[iq] = fh5[f"zeta_{iq}"][:]
    return KPointTHC(xi=xi, zeta=zeta, g_mapping=g_mapping, chi=chi)


def save_thc_factors(
    chkfile_name: str,
    chi: npt.NDArray,
    zeta: npt.NDArray,
    gpq_map: npt.NDArray,
    xi: Union[npt.NDArray, None] = None,
) -> None:
    """Write THC factors to file

    Args:
        chkfile_name: Filename to write to.
        chi: THC leaf tensor.
        zeta: THC central tensor.
        gpq_map: maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        xi: Interpolating vectors (optional, Default None).
    """
    num_kpts = chi.shape[0]
    with h5py.File(chkfile_name, "w") as fh5:
        fh5["chi"] = chi
        fh5["g_mapping"] = gpq_map
        if xi is not None:
            fh5["xi"] = xi
        for iq in range(num_kpts):
            fh5[f"zeta_{iq}"] = zeta[iq]


def get_zeta_size(zeta: npt.NDArray) -> int:
    """zeta (THC central tensor) is not contiguous so this helper function
    returns its size

    Args:
        zeta: THC central tensor

    Returns:
        zeta_size: Number of elements in zeta
    """
    return sum([z.size for z in zeta])


def unpack_thc_factors(
    xcur: npt.NDArray, num_thc: int, num_orb: int, num_kpts: int, num_g_per_q: list
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Unpack THC factors from flattened array used for reoptimization.

    Args:
        xcur: Flattened array containing k-point THC factors.
        num_thc: THC rank.
        num_orb: Number of orbitals.
        num_kpts: Number of kpoints.
        num_g_per_q: number of g vectors per q vector.

    Returns:
        chi: THC leaf tensor.
        zeta: THC central tensor.
    """
    # leaf tensor (num_kpts, num_orb, num_thc)
    chi_size = num_kpts * num_orb * num_thc
    chi_real = xcur[:chi_size].reshape(num_kpts, num_orb, num_thc)
    chi_imag = xcur[chi_size : 2 * chi_size].reshape(num_kpts, num_orb, num_thc)
    chi = chi_real + 1j * chi_imag
    zeta_packed = xcur[2 * chi_size :]
    zeta = []
    start = 0
    for iq in range(num_kpts):
        num_g = num_g_per_q[iq]
        size = num_g * num_g * num_thc * num_thc
        zeta_real = zeta_packed[start : start + size].reshape((num_g, num_g, num_thc, num_thc))
        zeta_imag = zeta_packed[start + size : start + 2 * size].reshape(
            (num_g, num_g, num_thc, num_thc)
        )
        zeta.append(zeta_real + 1j * zeta_imag)
        start += 2 * size
    return chi, zeta


def pack_thc_factors(chi: npt.NDArray, zeta: npt.NDArray, buffer: npt.NDArray) -> None:
    """Pack THC factors into flattened array used for reoptimization.

    Args:
        chi: THC leaf tensor.
        zeta: THC central tensor.
        buffer: Flattened array containing k-point THC factors. Modified inplace
    """
    assert len(chi.shape) == 3
    buffer[: chi.size] = chi.real.ravel()
    buffer[chi.size : 2 * chi.size] = chi.imag.ravel()
    start = 2 * chi.size
    num_kpts = len(zeta)
    for iq in range(num_kpts):
        size = zeta[iq].size
        buffer[start : start + size] = zeta[iq].real.ravel()
        buffer[start + size : start + 2 * size] = zeta[iq].imag.ravel()
        start += 2 * size


@jax.jit
def compute_objective_batched(
    chis: Tuple[jnpt.ArrayLike, jnpt.ArrayLike, jnpt.ArrayLike, jnpt.ArrayLike],
    zetas: jnpt.ArrayLike,
    chols: Tuple[jnpt.ArrayLike, jnpt.ArrayLike],
    norm_factors: Tuple[jnpt.ArrayLike, jnpt.ArrayLike, jnpt.ArrayLike, jnpt.ArrayLike],
    num_kpts: int,
    penalty_param: float = 0.0,
) -> float:
    """Compute THC objective function.

    Batches evaluation over kpts.

    Args:
        chis: THC leaf tensor.
        zetas: THC central tensor.
        chols: Cholesky factors definining 'exact' eris.
        norm_factors: THC normalization factors.
        num_kpts: Number of k-points.
        penalty_param: Penalty parameter.

    Returns:
        objective: THC objective function
    """
    eri_thc = jnp.einsum(
        "Jpm,Jqm,Jmn,Jrn,Jsn->Jpqrs",
        chis[0].conj(),
        chis[1],
        zetas,
        chis[2].conj(),
        chis[3],
        optimize=True,
    )
    eri_ref = jnp.einsum("Jnpq,Jnrs->Jpqrs", chols[0], chols[1], optimize=True)
    deri = (eri_thc - eri_ref) / num_kpts
    norm_left = norm_factors[0] * norm_factors[1]
    norm_right = norm_factors[2] * norm_factors[3]
    mpq_normalized = (
        jnp.einsum("JP,JPQ,JQ->JPQ", norm_left, zetas, norm_right, optimize=True) / num_kpts
    )

    lambda_z = jnp.sum(jnp.einsum("jpq->j", 0.5 * jnp.abs(mpq_normalized)) ** 2.0)

    res = 0.5 * jnp.sum((jnp.abs(deri)) ** 2) + penalty_param * lambda_z
    return res


def prepare_batched_data_indx_arrays(
    momentum_map: npt.NDArray, gpq_map: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    r"""Create arrays to batch over.

    Flatten sum_q sum_{k,k_prime} -> sum_q \sum_{indx} and pack momentum
    conserving indices and central tensors so we can sum over indx efficiently.

    Args:
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.

    Returns:
        indx_pqrs: momentum conserving k-point indices.
        zetas: Batches Central tensors.
    """
    num_kpts = momentum_map.shape[0]
    indx_pqrs = np.zeros((num_kpts, num_kpts**2, 4), dtype=jnp.int32)
    zetas = np.zeros((num_kpts, num_kpts**2, 2), dtype=jnp.int32)
    for iq in range(num_kpts):
        indx = 0
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            gpq = gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                gsr = gpq_map[iq, ik_prime]
                indx_pqrs[iq, indx] = [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
                zetas[iq, indx] = [gpq, gsr]
                indx += 1
    return indx_pqrs, zetas


@jax.jit
def get_batched_data_1indx(array: jnpt.ArrayLike, indx: jnpt.ArrayLike) -> jnpt.ArrayLike:
    """Helper function to extract entries of array given another array.

    Args:
        array: Array to index
        indx: Indexing array

    Retuns:
        indexed_array: i.e. array[indx]
    """
    return array[indx]


@jax.jit
def get_batched_data_2indx(
    array: jnpt.ArrayLike, indxa: jnpt.ArrayLike, indxb: jnpt.ArrayLike
) -> jnpt.ArrayLike:
    """Helper function to extract entries of 2D array given another array

    Args:
        array: Array to index
        indxa: Indexing array
        indxb: Indexing array

    Retuns:
        indexed_array: i.e. array[indxa, indxb]
    """
    return array[indxa, indxb]


def thc_objective_regularized_batched(
    xcur: jnpt.ArrayLike,
    num_orb: int,
    num_thc: int,
    momentum_map: npt.NDArray,
    gpq_map: npt.NDArray,
    chol: jnpt.ArrayLike,
    indx_arrays: Tuple[jnpt.ArrayLike, jnpt.ArrayLike],
    batch_size: int,
    penalty_param=0.0,
) -> float:
    """Compute THC objective function.

    Here we batch over multiple k-point indices for improved GPU efficiency.

    Args:
        xcur: Flattened array containing k-point THC factors.
        num_orb: Number of orbitals.
        num_thc: THC rank.
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: Maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        chol: Cholesky factors definining 'exact' eris.
        indx_arrays: Batched index arrays (see prepare_batched_data_indx_arrays)
        batch_size: Size of each batch of data. Should be in range
        [1, num_kpts**2]. penalty_param: Penalty param if computing regularized
        cost function.

    Returns:
        objective: THC objective function
    """
    num_kpts = momentum_map.shape[0]
    num_g_per_q = [len(np.unique(gq)) for gq in gpq_map]
    chi, zeta = unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_g_per_q)
    # Normalization factor, no factor of sqrt as there are 4 chis in total when
    # building ERI.
    norm_kp = jnp.einsum("kpP,kpP->kP", chi.conj(), chi, optimize=True) ** 0.5
    num_batches = math.ceil(num_kpts**2 / batch_size)

    indx_pqrs, indx_zeta = indx_arrays
    objective = 0.0
    for iq in range(num_kpts):
        for ibatch in range(num_batches):
            start = ibatch * batch_size
            end = (ibatch + 1) * batch_size
            chi_p = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 0])
            chi_q = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 1])
            chi_r = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 2])
            chi_s = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 3])
            norm_k1 = get_batched_data_1indx(norm_kp, indx_pqrs[iq, start:end, 0])
            norm_k2 = get_batched_data_1indx(norm_kp, indx_pqrs[iq, start:end, 1])
            norm_k3 = get_batched_data_1indx(norm_kp, indx_pqrs[iq, start:end, 2])
            norm_k4 = get_batched_data_1indx(norm_kp, indx_pqrs[iq, start:end, 3])
            zeta_batch = get_batched_data_2indx(
                zeta[iq], indx_zeta[iq, start:end, 0], indx_zeta[iq, start:end, 1]
            )
            chol_batch_pq = get_batched_data_2indx(
                chol, indx_pqrs[iq, start:end, 0], indx_pqrs[iq, start:end, 1]
            )
            chol_batch_rs = get_batched_data_2indx(
                chol, indx_pqrs[iq, start:end, 2], indx_pqrs[iq, start:end, 3]
            )
            objective += compute_objective_batched(
                (chi_p, chi_q, chi_r, chi_s),
                zeta_batch,
                (chol_batch_pq, chol_batch_rs),
                (norm_k1, norm_k2, norm_k3, norm_k4),
                num_kpts,
                penalty_param=penalty_param,
            )
    return objective


def thc_objective_regularized(
    xcur, num_orb, num_thc, momentum_map, gpq_map, chol, penalty_param=0.0
):
    """Compute THC objective function. Non-batched version.

    Args:
        xcur: Flattened array containing k-point THC factors.
        num_orb: Number of orbitals.
        num_thc: THC rank.
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: Maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        chol: Cholesky factors definining 'exact' eris.
        penalty_param: Penalty param if computing regularized cost function.

    Returns:
        objective: THC objective function
    """
    res = 0.0
    num_kpts = momentum_map.shape[0]
    num_g_per_q = [len(np.unique(GQ)) for GQ in gpq_map]
    chi, zeta = unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_g_per_q)
    num_kpts = momentum_map.shape[0]
    norm_kP = jnp.einsum("kpP,kpP->kP", chi.conj(), chi, optimize=True) ** 0.5
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                Gsr = gpq_map[iq, ik_prime]
                eri_thc = jnp.einsum(
                    "pm,qm,mn,rn,sn->pqrs",
                    chi[ik].conj(),
                    chi[ik_minus_q],
                    zeta[iq][Gpq, Gsr],
                    chi[ik_prime_minus_q].conj(),
                    chi[ik_prime],
                )
                eri_ref = jnp.einsum(
                    "npq,nsr->pqrs", chol[ik, ik_minus_q], chol[ik_prime, ik_prime_minus_q].conj()
                )
                deri = (eri_thc - eri_ref) / num_kpts
                norm_left = norm_kP[ik] * norm_kP[ik_minus_q]
                norm_right = norm_kP[ik_prime_minus_q] * norm_kP[ik_prime]
                MPQ_normalized = (
                    jnp.einsum("P,PQ,Q->PQ", norm_left, zeta[iq][Gpq, Gsr], norm_right) / num_kpts
                )

                lambda_z = 0.5 * jnp.sum(jnp.abs(MPQ_normalized))
                res += 0.5 * jnp.sum((jnp.abs(deri)) ** 2) + penalty_param * (lambda_z**2)

    return res


def lbfgsb_opt_kpthc_l2reg(
    chi: npt.NDArray,
    zeta: npt.NDArray,
    momentum_map: npt.NDArray,
    gpq_map: npt.NDArray,
    chol: npt.NDArray,
    chkfile_name: Union[str, None] = None,
    maxiter: int = 150_000,
    disp_freq: int = 98,
    penalty_param: Union[float, None] = None,
) -> Tuple[npt.NDArray, float]:
    """Least-squares fit of two-electron integral tensors with  L-BFGS-B

    Uses l2-regularization of lambda penalty function.

    Args:
        chi: THC leaf tensor.
        zeta: THC central tensor.
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: Maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        chol: Cholesky factors definining 'exact' eris.
        batch_size: Size of each batch of data. Should be in range
            [1, num_kpts**2].
        penalty_param: Penalty param if computing regularized cost function.
        chkfile_name: Filename to store intermediate state of optimization to.
        maxiter: Max L-BFGS-B iteration.
        disp_freq: L-BFGS-B disp_freq.
        penalty_param: Paramter to penalize optimization by one-norm of
            Hamiltonian. If None it is determined automatically through trying
            to balance the two terms in the objective function.

    Returns:
        objective: THC objective function
    """
    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    print(initial_guess)
    loss = thc_objective_regularized(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        penalty_param=0.0,
    )
    reg_loss = thc_objective_regularized(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        penalty_param=1.0,
    )
    # set penalty
    lambda_z = (reg_loss - loss) ** 0.5
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        penalty_param = loss / lambda_z
    print("loss {}".format(loss))
    print("lambda_z {}".format(lambda_z))
    print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    thc_grad = jax.grad(thc_objective_regularized, argnums=[0])
    print("Initial Grad")
    print(
        thc_grad(
            jnp.array(initial_guess),
            num_orb,
            num_thc,
            momentum_map,
            gpq_map,
            jnp.array(chol),
            penalty_param,
        )
    )
    res = minimize(
        thc_objective_regularized,
        initial_guess,
        args=(num_orb, num_thc, momentum_map, gpq_map, jnp.array(chol), penalty_param),
        jac=thc_grad,
        method="L-BFGS-B",
        options={"disp": disp_freq > 0, "iprint": disp_freq, "maxiter": maxiter},
    )

    params = np.array(res.x)
    loss = thc_objective_regularized(
        jnp.array(res.x),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        penalty_param=0.0,
    )
    if chkfile_name is not None:
        num_g_per_q = [len(np.unique(GQ)) for GQ in gpq_map]
        chi, zeta = unpack_thc_factors(params, num_thc, num_orb, num_kpts, num_g_per_q)
        save_thc_factors(chkfile_name, chi, zeta, gpq_map)
    return np.array(params), loss


def lbfgsb_opt_kpthc_l2reg_batched(
    chi: npt.NDArray,
    zeta: npt.NDArray,
    momentum_map: npt.NDArray,
    gpq_map: npt.NDArray,
    chol: npt.NDArray,
    chkfile_name: Union[str, None] = None,
    maxiter: int = 150_000,
    disp_freq: int = 98,
    penalty_param: Union[float, None] = None,
    batch_size: Union[int, None] = None,
) -> Tuple[npt.NDArray, float]:
    """Least-squares fit of two-electron integral tensors with  L-BFGS-B.

    Uses l2-regularization of lambda. This version batches over multiple
    k-points which may be faster on GPUs.

    Args:
        chi: THC leaf tensor.
        zeta: THC central tensor.
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: Maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        chol: Cholesky factors definining 'exact' eris.
        batch_size: Size of each batch of data. Should be in range
            [1, num_kpts**2].
        penalty_param: Penalty param if computing regularized cost function.
        chkfile_name: Filename to store intermediate state of optimization to.
        maxiter: Max L-BFGS-B iteration.
        disp_freq: L-BFGS-B disp_freq.
        penalty_param: Paramter to penalize optimization by one-norm of
            Hamiltonian. If None it is determined automatically through trying
            to balance the two terms in the objective function.
        batch_size: Number of k-points-pairs to batch over. Default num_kpts**2.

    Returns:
        objective: THC objective function
    """
    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    if batch_size is None:
        batch_size = num_kpts**2
    indx_arrays = prepare_batched_data_indx_arrays(momentum_map, gpq_map)
    data_amount = batch_size * (4 * num_orb * num_thc + num_thc * num_thc)  # chi[p,m] + zeta[m,n]
    data_size_gb = (data_amount * 16) / (1024**3)
    print(f"Batch size in GB: {data_size_gb}")
    loss = thc_objective_regularized_batched(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        indx_arrays,
        batch_size,
        penalty_param=0.0,
    )
    start = time.time()
    reg_loss = thc_objective_regularized_batched(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        indx_arrays,
        batch_size,
        penalty_param=1.0,
    )
    print("Time to evaluate loss function : {:.4f}".format(time.time() - start))
    print("loss {}".format(loss))
    # set penalty
    lambda_z = (reg_loss - loss) ** 0.5
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        penalty_param = loss / lambda_z
    print("lambda_z {}".format(lambda_z))
    print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    thc_grad = jax.grad(thc_objective_regularized_batched, argnums=[0])
    print("Initial Grad")
    start = time.time()
    print(
        thc_grad(
            jnp.array(initial_guess),
            num_orb,
            num_thc,
            momentum_map,
            gpq_map,
            jnp.array(chol),
            indx_arrays,
            batch_size,
            penalty_param,
        )
    )
    print("# Time to evaluate gradient: {:.4f}".format(time.time() - start))
    res = minimize(
        thc_objective_regularized_batched,
        initial_guess,
        args=(
            num_orb,
            num_thc,
            momentum_map,
            gpq_map,
            jnp.array(chol),
            indx_arrays,
            batch_size,
            penalty_param,
        ),
        jac=thc_grad,
        method="L-BFGS-B",
        options={"disp": disp_freq > 0, "iprint": disp_freq, "maxiter": maxiter},
    )
    loss = thc_objective_regularized_batched(
        jnp.array(res.x),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        indx_arrays,
        batch_size,
        penalty_param=0.0,
    )

    params = np.array(res.x)
    if chkfile_name is not None:
        num_g_per_q = [len(np.unique(gq)) for gq in gpq_map]
        chi, zeta = unpack_thc_factors(params, num_thc, num_orb, num_kpts, num_g_per_q)
        save_thc_factors(chkfile_name, chi, zeta, gpq_map)
    return np.array(params), loss


def adagrad_opt_kpthc_batched(
    chi,
    zeta,
    momentum_map,
    gpq_map,
    chol,
    batch_size=None,
    chkfile_name=None,
    stepsize=0.01,
    momentum=0.9,
    maxiter=50_000,
    gtol=1.0e-5,
) -> Tuple[npt.NDArray, float]:
    """Adagrad optimization of THC factors.

    THC optimization usually starts with BFGS and then is completed with Adagrad
    or other first order solver.

    Args:
        chi: THC leaf tensor.
        zeta: THC central tensor.
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: Maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        chol: Cholesky factors definining 'exact' eris.
        batch_size: Size of each batch of data. Should be in range
            [1, num_kpts**2].
        chkfile_name: Filename to store intermediate state of optimization to.
        maxiter: Max L-BFGS-B iteration.
        disp_freq: L-BFGS-B disp_freq.
        penalty_param: Paramter to penalize optimization by one-norm of
            Hamiltonian. If None it is determined automatically through trying
            to balance the two terms in the objective function.
        batch_size: Number of k-points-pairs to batch over. Default num_kpts**2.

    Returns:
        objective: THC objective function
    """
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    # set initial guess
    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    opt_init, opt_update, get_params = adagrad(step_size=stepsize, momentum=momentum)
    opt_state = opt_init(initial_guess)
    thc_grad = jax.grad(thc_objective_regularized_batched, argnums=[0])

    if batch_size is None:
        batch_size = num_kpts**2
    indx_arrays = prepare_batched_data_indx_arrays(momentum_map, gpq_map)

    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = thc_grad(
            params, num_orb, num_thc, momentum_map, gpq_map, chol, indx_arrays, batch_size
        )
        grad_norm_l1 = np.linalg.norm(gradient[0], ord=1)
        return opt_update(i, gradient[0], opt_state), grad_norm_l1

    for t in range(maxiter):
        opt_state, grad_l1 = update(t, opt_state)
        params = get_params(opt_state)
        if t % 500 == 0:
            fval = thc_objective_regularized_batched(
                params, num_orb, num_thc, momentum_map, gpq_map, chol, indx_arrays, batch_size
            )
            outline = "Objective val {: 5.15f}".format(fval)
            outline += "\tGrad L1-norm {: 5.15f}".format(grad_l1)
            print(outline)
        if grad_l1 <= gtol:
            # break out of loop
            # which sends to save
            break
    else:
        print("Maximum number of iterations reached")
    # save results before returning
    x = np.array(params)
    loss = thc_objective_regularized_batched(
        params, num_orb, num_thc, momentum_map, gpq_map, chol, indx_arrays, batch_size
    )
    if chkfile_name is not None:
        num_g_per_q = [len(np.unique(GQ)) for GQ in gpq_map]
        chi, zeta = unpack_thc_factors(x, num_thc, num_orb, num_kpts, num_g_per_q)
        save_thc_factors(chkfile_name, chi, zeta, gpq_map)
    return params, loss


def make_contiguous_cholesky(cholesky: npt.NDArray) -> npt.NDArray:
    """It is convenient for optimization to make the cholesky array contiguous.
    This function truncates and auxiliary index that is greater than the minimum
    number of auxiliary vectors.

    Args:
        cholesky: Cholesky vectors

    Returns:
        cholesk_contg: Contiguous array of cholesky vectors.
    """
    num_kpts = len(cholesky)
    num_mo = cholesky[0, 0].shape[-1]
    if cholesky.dtype == object:
        # Jax requires contiguous arrays so just truncate naux if it's not
        # uniform hopefully shouldn't affect results dramatically as from
        # experience the naux amount only varies by a few % per k-point
        # Alternatively todo: padd with zeros
        min_naux = min([cholesky[k1, k1].shape[0] for k1 in range(num_kpts)])
        cholesky_contiguous = np.zeros(
            (num_kpts, num_kpts, min_naux, num_mo, num_mo), dtype=np.complex128
        )
        for ik1 in range(num_kpts):
            for ik2 in range(num_kpts):
                cholesky_contiguous[ik1, ik2] = cholesky[ik1, ik2][:min_naux]
    else:
        cholesky_contiguous = cholesky

    return cholesky_contiguous


def compute_isdf_loss(chi, zeta, momentum_map, gpq_map, chol):
    """Helper function to compute ISDF loss.

    Args:
        chi: THC leaf tensor.
        zeta: THC central tensor.
        momentum_map: momentum transfer mapping. map[iQ, ik_p] -> ik_q;
            (kpts[ikp] - kpts[ikq])%G = kpts[iQ].
        gpq_map: Maps momentum conserving tuples of kpoints to reciprocal
            lattice vectors in THC central tensor.
        chol: Cholesky factors definining 'exact' eris.

    Returns:
        loss: ISDF loss in ERIs.
    """
    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    loss = thc_objective_regularized(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        gpq_map,
        jnp.array(chol),
        penalty_param=0.0,
    )
    return loss


def kpoint_thc_via_isdf(
    kmf: scf.RHF,
    cholesky: npt.NDArray,
    num_thc: int,
    perform_bfgs_opt: bool = True,
    perform_adagrad_opt: bool = True,
    bfgs_maxiter: int = 3000,
    adagrad_maxiter: int = 3000,
    checkpoint_basename: str = "thc",
    save_checkpoints: bool = False,
    use_batched_algos: bool = True,
    penalty_param: Union[None, float] = None,
    batch_size: Union[None, bool] = None,
    max_kmeans_iteration: int = 500,
    verbose: bool = False,
    initial_guess: Union[None, KPointTHC] = None,
    isdf_density_guess: bool = False,
) -> Tuple[KPointTHC, dict]:
    """
    Solve k-point THC using ISDF as an initial guess.

    Arguments:
        kmf: instance of pyscf.pbc.KRHF object. Must be using FFTDF density
            fitting for integrals.
        cholesky: 3-index cholesky integrals needed for BFGS optimization.
        num_thc: THC dimensions (int), usually nthc = c_thc * n, where n is the
            number of spatial orbitals in the unit cell and c_thc is some
            poisiitve integer (typically <= 15).
        perform_bfgs_opt: Perform subsequent BFGS optimization of THC
            factors?
        perform_adagrad_opt: Perform subsequent adagrad optimization of THC
            factors? This is performed after BFGD if perform_bfgs_opt is True.
        bfgs_maxiter: Maximum iteration for adagrad optimization.
        adagrad_maxiter: Maximum iteration for adagrad optimization.
        save_checkpoints: Whether to save checkpoint files or not (which will
            store THC factors. For each step we store checkpoints as
            {checkpoint_basename}_{thc_method}.h5, where thc_method is one of
            the strings (isdf, bfgs or adagrad).
        checkpoint_basename: Base name for checkpoint files. string,
            default "thc".
        use_batched_algos: Whether to use batched algorithms which may be
            faster but have higher memory cost. Bool. Default True.
        penalty_param: Penalty parameter for l2 regularization. Float.
            Default None.
        max_kmeans_iteration: Maximum number of iterations for KMeansCVT
            step. int. Default 500.
        verbose: Print information? Bool, default False.

    Returns:
        kthc: k-point THC factors
        info: dictionary of losses for each stage of factorization.
    """
    # Perform initial ISDF calculation of THC factors
    info = {}
    start = time.time()
    if initial_guess is not None:
        kpt_thc = initial_guess
    else:
        kpt_thc = solve_kmeans_kpisdf(
            kmf,
            num_thc,
            single_translation=False,
            verbose=verbose,
            max_kmeans_iteration=max_kmeans_iteration,
            use_density_guess=isdf_density_guess,
        )
    if verbose:
        print("Time for generating initial guess {:.4f}".format(time.time() - start))
    num_mo = kmf.mo_coeff[0].shape[-1]
    num_kpts = len(kmf.kpts)
    chi, zeta, g_mapping = kpt_thc.chi, kpt_thc.zeta, kpt_thc.g_mapping
    if save_checkpoints:
        chkfile_name = f"{checkpoint_basename}_isdf.h5"
        save_thc_factors(chkfile_name, chi, zeta, g_mapping, kpt_thc.xi)
    momentum_map = build_momentum_transfer_mapping(kmf.cell, kmf.kpts)
    if cholesky is not None:
        cholesky_contiguous = make_contiguous_cholesky(cholesky)
        info["loss_isdf"] = compute_isdf_loss(
            chi, zeta, momentum_map, g_mapping, cholesky_contiguous
        )
    start = time.time()
    if perform_bfgs_opt:
        if save_checkpoints:
            chkfile_name = f"{checkpoint_basename}_bfgs.h5"
        else:
            chkfile_name = None
        if use_batched_algos:
            opt_params, loss_bfgs = lbfgsb_opt_kpthc_l2reg_batched(
                chi,
                zeta,
                momentum_map,
                g_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
                batch_size=batch_size,
                disp_freq=(98 if verbose else -1),
            )
            info["loss_bfgs"] = loss_bfgs
        else:
            opt_params, loss_bfgs = lbfgsb_opt_kpthc_l2reg(
                chi,
                zeta,
                momentum_map,
                g_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
                disp_freq=(98 if verbose else -1),
            )
            info["loss_bfgs"] = loss_bfgs
        num_g_per_q = [len(np.unique(GQ)) for GQ in g_mapping]
        chi, zeta = unpack_thc_factors(opt_params, num_thc, num_mo, num_kpts, num_g_per_q)
    if verbose:
        print("Time for BFGS {:.4f}".format(time.time() - start))
    start = time.time()
    if perform_adagrad_opt:
        if save_checkpoints:
            chkfile_name = f"{checkpoint_basename}_adagrad.h5"
        else:
            chkfile_name = None
        if use_batched_algos:
            opt_params, loss_ada = adagrad_opt_kpthc_batched(
                chi,
                zeta,
                momentum_map,
                g_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=adagrad_maxiter,
                batch_size=batch_size,
            )
            info["loss_adagrad"] = loss_ada
        num_g_per_q = [len(np.unique(GQ)) for GQ in g_mapping]
        chi, zeta = unpack_thc_factors(opt_params, num_thc, num_mo, num_kpts, num_g_per_q)
    if verbose:
        print("Time for ADAGRAD {:.4f}".format(time.time() - start))
    result = KPointTHC(chi=chi, zeta=zeta, g_mapping=g_mapping, xi=kpt_thc.xi)
    return result, info
