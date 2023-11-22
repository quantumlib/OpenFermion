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
"""Module for performing ISDF THC factorization of k-point dependent integrals.

The ISDF (interpolative separable density fitting) implementation currently
provides a tensor hypercontraction (THC)-like factorization of the two electron
integrals which should converge to the exact representation (fast fourier
transform density fitting (FFTDF)) of the ERIs in the limit of large THC rank.
This differs from the assumption of using range-separated gaussian density
fitting (RSGDF) throughout the rest of the resource estimation scripts. However,
we typically are only interested in ISDF as an initial guess for the THC factors
which are then subsequently reoptimized to regularize lambda. The assumption
here is that FFTDF / ISDF is a good enough approximation to the RSGDF ERIs and
thus serves as a good initial guess.
"""

import itertools
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.linalg

from pyscf.pbc import df, gto, scf, tools
from pyscf.pbc.dft import numint
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.lib.kpts_helper import conj_mapping, get_kconserv, unique

from openfermion.resource_estimates.pbc.thc.factorizations.kmeans import KMeansCVT
from openfermion.resource_estimates.pbc.hamiltonian import build_momentum_transfer_mapping


def check_isdf_solution(
    orbitals: npt.NDArray, interp_orbitals: npt.NDArray, xi: npt.NDArray
) -> float:
    r"""Check accuracy of isdf least squares solution.

    Very costly and should only be used for testing purposes.

    Args:
        orbitals: Orbitals on full real space grid. [num_grd, num_orb]
        interp_orbitals: interpolating orbitals (those orbitals evaluated on
            interpolating points.) [num_interp, num_orb]
        xi: interpolating vectors. [num_grid, num_interp]
    Returns:
        error: |phi_{ij}(r) - \sum_m xi_m(r) phi_{ij}(r_m)
        orbitals: npt.NDArray:
        interp_orbitals: npt.NDArray:
        xi: npt.NDArray:
    """

    lhs = np.einsum("Ri,Rj->Rij", orbitals.conj(), orbitals, optimize=True)
    rhs = np.einsum("mi,mj->mij", interp_orbitals.conj(), interp_orbitals, optimize=True)
    lhs_check = np.einsum("Rm,mij->Rij", xi, rhs, optimize=True)
    return np.linalg.norm(lhs - lhs_check)


def solve_isdf(orbitals: npt.NDArray, interp_indx: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Solve for interpolating vectors given interpolating points and orbitals.

    Used for both supercell and k-point ISDF factorizations.

    Args:
        orbitals: orbitals on a grid of shape [num_grid_points, num_orbitals]
        interp_indx: array indexing interpolating points (subset of grid
            points to use selected by K-Means algorithm. shape is
            [num_interp_points].
        orbitals: npt.NDArray:
        interp_indx: npt.NDArray:

    Returns:
        tuple: (Interpolang vectors, interpolating orbitals) (xi_mu(r),
            phi_i(r_mu)). Note xi_mu(r) is called theta[R, mu] in keeping with
            original ISDF notation.
    """
    interp_orbitals = orbitals[interp_indx]
    # Form pseudo-densities
    # P[R, r_mu] = \sum_{i} phi_{R,i}^* phi_{mu,i}
    pseudo_density = np.einsum("Ri,mi->Rm", orbitals.conj(), interp_orbitals, optimize=True)
    # [Z C^]_{J, mu} = (sum_i phi_{i, J}^* phi_{i, mu}) (sum_j phi_{j, J}
    # phi_{i, mu})
    zc_dag = np.einsum("Rm,Rm->Rm", pseudo_density, pseudo_density.conj(), optimize=True)
    # Just down sample from ZC_dag
    cc_dag = zc_dag[interp_indx].copy()
    # Solve ZC_dag = Theta CC_dag
    # Theta = ZC_dag @ CC_dag_inv
    # Solve ZC_dag = Theta CC_dag
    # -> ZC_dag^T = CC_dag^T Theta^T
    # rcond = None uses MACH_EPS * max(M,N) for least squares convergence.
    theta_dag, _, _, _ = np.linalg.lstsq(cc_dag.conj().T, zc_dag.conj().T, rcond=None)
    return theta_dag.conj().T, interp_orbitals


def supercell_isdf(
    mydf: df.FFTDF,
    interp_indx: npt.NDArray,
    orbitals: npt.NDArray,
    grid_points: npt.NDArray,
    kpoint=np.zeros(3),
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""
    Build ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    Args:
        mydf: instance of pyscf.pbc.df.FFTDF object.
        interp_indx: array indexing interpolating points determined through
            K-Means CVT procedure. Dimension [num_interp_points]
        orbitals: orbitals on a grid of shape [num_grid_points, num_orbitals]
        grid_points: Real space grid. Dimension [num_grid_points, num_dim],
            num_dim is 1, 2 or 3 for 1D, 2D, 3D.

    Returns:
        tuple: (chi, zeta, Theta): orbitals on interpolating
            points, zeta (central tensor), and matrix of interpolating vectors
            Theta of dimension [num_grid_points, num_interp_points] (also called
            xi_mu(r)), where num_grid_points is the number of real space grid
            points and num_interp_points is the number of interpolating points.
    """

    cell = mydf.cell
    num_grid_points = len(grid_points)

    theta, chi = solve_isdf(orbitals, interp_indx)

    # FFT Theta[R, mu] -> Theta[mu, G]
    # Transpose as fft expects contiguous.
    theta_g = tools.fft(theta.T, mydf.mesh)
    coulg = tools.get_coulG(cell, k=kpoint, mesh=mydf.mesh)
    weighted_coulg = coulg * cell.vol / num_grid_points**2.0

    # zeta_{mu,nu} = \sum_G 4pi/(omega * G^2) zeta_{mu,G} * (zeta_G*){nu, G}
    theta_g_tilde = np.einsum("iG,G->iG", theta_g, weighted_coulg)
    zeta = (theta_g_tilde) @ theta_g.conj().T
    return chi, zeta, theta


def build_kpoint_zeta(
    df_inst: df.FFTDF,
    q: int,
    delta_g: npt.NDArray,
    delta_g_prime: npt.NDArray,
    grid_points: npt.NDArray,
    xi_mu: npt.NDArray,
) -> npt.NDArray:
    """Build k-point THC zeta (central tensor).

    Built for for given q, delta_g, delta_g_prime.

    Args:
        mydf: instance of pyscf.pbc.df.FFTDF object.
        q: Momentum transfer (in 1BZ).
        delta_g: Reciprocal lattice vector satisfying Q - (Q-k) = delta_G
        delta_g_prime: Reciprocal lattice vector satisfying Q - (Q-k') = delta_G
        grid_points: Real space grid. Dimension [num_grid_points, num_dim],
            num_dim is 1, 2 or 3 for 1D, 2D, 3D.
        xi_mu: array containing interpolating vectors determined during ISDF
            procedure

    Returns:
        zeta: central tensor of dimension [num_interp_points, num_interp_points]

    """
    cell = df_inst.cell
    num_grid_points = grid_points.shape[0]
    # delta_G - delta_G_prime because we have Gpq and Gsr and Gsr = -Grs, phase
    # = Delta G = Gpq + Grs
    phase_factor = np.exp(
        -1j * (np.einsum("x,Rx->R", delta_g - delta_g_prime, grid_points, optimize=True))
    )
    # Minus sign again due to we use Q = kp - kq, but we should have
    # V(G + k_q - k_p)
    coulg = tools.get_coulG(cell, k=-(q + delta_g), mesh=df_inst.mesh)
    weighted_coulg = coulg * cell.vol / num_grid_points
    xi_mug = tools.fft(xi_mu.T, df_inst.mesh)
    xi_mug *= weighted_coulg
    vr = tools.ifft(xi_mug, df_inst.mesh)
    zeta = np.einsum("R,Rn,mR->mn", phase_factor, xi_mu, vr, optimize=True)
    return zeta


def build_kpoint_zeta_single_tranlsation(
    df_inst: df.FFTDF, q: int, delta_g: npt.NDArray, grid_points: npt.NDArray, xi_mu: npt.NDArray
) -> npt.NDArray:
    """Build k-point THC zeta (central tensor)

    Built for for given q, delta_g. Note single delta_g (single translation
    vector) in controast to the _double_translation variant above.

    Args:
      mydf: instance of pyscf.pbc.df.FFTDF object.
      q: Momentum transfer kp-kq.
      delta_G: Reciprocal lattice vector satisfying Q - (Q-k) = delta_G
      grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
      xi_mu: array containing interpolating vectors determined during ISDF
        procedure

    Returns:
      zeta: central tensor of dimension [num_interp_points, num_interp_points]

    """
    cell = df_inst.cell
    num_grid_points = grid_points.shape[0]
    # delta_G - delta_G_prime because we have Gpq and Gsr and Gsr = -Grs, phase
    # = Delta G = Gpq + Grs
    phase_factor = np.exp(-1j * (np.einsum("x,Rx->R", delta_g, grid_points, optimize=True)))
    # Minus sign again due to we use Q = kp - kq, but we should have
    # V(G + k_q - k_p)
    coulg = tools.get_coulG(cell, k=-q, mesh=df_inst.mesh)
    weighted_coulg = coulg * cell.vol / num_grid_points
    xi_mug = tools.fft(xi_mu.T, df_inst.mesh)
    xi_mug *= weighted_coulg
    vr = tools.ifft(xi_mug, df_inst.mesh)
    zeta = np.einsum("R,Rn,mR->mn", phase_factor, xi_mu, vr, optimize=True)
    return zeta


def build_g_vectors(cell: gto.Cell) -> npt.NDArray:
    """Build all 27 Gvectors

    Args:
        cell: pyscf.pbc.gto.Cell object.
        cell: gto.Cell:

    Returns:
        tuple: g_dict a dictionary mapping miller index to appropriate
            g_vector index and g_vectors array of 27 g_vectors shape [27, 3].

    """
    g_dict = {}
    g_vectors = np.zeros((27, 3), dtype=np.float64)
    lattice_vectors = cell.lattice_vectors()
    indx = 0
    for n1, n2, n3 in itertools.product(range(-1, 2), repeat=3):
        g_dict[(n1, n2, n3)] = indx
        g_vectors[indx] = np.einsum("n,ng->g", (n1, n2, n3), cell.reciprocal_vectors())
        miller_indx = np.rint(np.einsum("nx,x->n", lattice_vectors, g_vectors[indx]) / (2 * np.pi))
        assert (miller_indx == (n1, n2, n3)).all()
        indx += 1
    return g_dict, g_vectors


def find_unique_g_vectors(
    g_vectors: npt.NDArray, g_mapping: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Find all unique G-vectors and build mapping to original set.

    Args:
        g_vectors: array of 27 g-vectors.
        g_mapping: array of 27 g-vectors.

    Returns:
      unique_g_index in range [0,...,num_unique_gs[iq]], and delta_gs are the
        unique g-vectors of size [num_qpoints, num_unique_gs[iq]].

    """
    unique_mapping = np.zeros_like(g_mapping)
    num_qpoints = g_mapping.shape[0]
    delta_gs = np.zeros((num_qpoints,), dtype=object)
    for iq in range(num_qpoints):
        unique_g = np.unique(g_mapping[iq])
        delta_gs[iq] = g_vectors[unique_g]
        # build map to unique index
        unique_mapping[iq] = [ix for el in g_mapping[iq] for ix in np.where(unique_g == el)[0]]

    return unique_mapping, delta_gs


def build_g_vector_mappings_double_translation(
    cell: gto.Cell, kpts: npt.NDArray, momentum_map: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """build g-vector mappings that map k-point differences to 1bz.

    args:
      cell: pyscf.pbc.gto.cell object.
      kpts: array of kpoints.
      momentum_map: momentum mapping to satisfy q = (k_p - k_q) mod g.
        momentum_map[iq, ikp] = ikq.

    returns:
      tuple: (g_vectors, gpq_mapping, gpq_mapping_unique, delta_gs), g_vectors
        is a list of all 27
      g-vectors and gpq_mapping[iq, kp] = indx_gpq, where gpq = kpts[ikp] -
      kpts[ikq] - kpts[iq], i.e. returns index to g-vector (consistent with
      g_vectors) satisfying this condition. gpq_mapping_unique provides
      mapping to unique g_vector index. delta_gs provides compressed lists of
      unique g vectors.

    """
    g_dict, g_vectors = build_g_vectors(cell)
    lattice_vectors = cell.lattice_vectors()
    num_kpts = len(kpts)
    gpq_mapping = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    num_kpts = len(kpts)
    for iq in range(num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            delta_gpq = (kpts[ikp] - kpts[ikq]) - kpts[iq]
            miller_indx = np.rint(np.einsum("wx,x->w", lattice_vectors, delta_gpq) / (2 * np.pi))
            gpq_mapping[iq, ikp] = g_dict[tuple(miller_indx)]
    gpq_mapping_unique, delta_gs = find_unique_g_vectors(g_vectors, gpq_mapping)
    return g_vectors, gpq_mapping, gpq_mapping_unique, delta_gs


def get_miller(lattice_vectors: npt.NDArray, g: npt.NDArray) -> npt.NDArray:
    """convert g to miller indx.

    args:
      lattice_vectors: array of lattice vectors.
      g: reciprocal lattice vector.

    returns:
      miller_index: 3d array of miller indices.

    """
    miller_indx = np.rint(np.einsum("wx,x->w", lattice_vectors, g) / (2 * np.pi)).astype(np.int32)
    return miller_indx


def build_minus_q_g_mapping(
    cell: gto.Cell, kpts: npt.NDArray, momentum_map: npt.NDArray
) -> npt.NDArray:
    """Build conjugat g map

    Mapping satisfies (-Q) + G + (Q + Gpq) = 0 (*)

    Args:
        cell: pyscf.pbc.gto.Cell object.
        kpts: array of kpoints.
        momentum_map: momentum mapping to satisfy q = (k_p - k_q) mod g.
            momentum_map[iq, ikp] = ikq.

    Returns:
        minus_q_mapping: [indx_minus_q, k]-th element yields index for g
            satisfying (*) above, where indx_minus_q is given by indx_minus_q =
            minus_k[q], and minus_k = conj_mapping(cell, kpts).
        minus_q_mapping_unique: indexes the appropriate g vector given by
            delta_gs[indx_minus_q][indx] = g, where indx =
            minus_q_mapping_unique[indx_minus_q, k], and deltags is built by
            build_g_vector_mappings_double_translation.
    """
    (g_vecs, g_map, _, delta_gs) = build_g_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    g_dict, _ = build_g_vectors(cell)
    num_kpts = len(kpts)
    lattice_vectors = cell.lattice_vectors()
    minus_k_map = conj_mapping(cell, kpts)
    minus_q_mapping = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    minus_q_mapping_unique = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    for iq in range(num_kpts):
        minus_iq = minus_k_map[iq]
        for ik in range(num_kpts):
            gpq = g_vecs[g_map[iq, ik]]
            # complementary gpq (g in (*) in docstring)
            gpq_comp = -(kpts[minus_iq] + kpts[iq] + gpq)
            # find index in original set of 27
            igpq_comp = g_dict[tuple(get_miller(lattice_vectors, gpq_comp))]
            minus_q_mapping[minus_iq, ik] = igpq_comp
        indx_delta_gs = np.array(
            [g_dict[tuple(get_miller(lattice_vectors, g))] for g in delta_gs[minus_iq]]
        )
        minus_q_mapping_unique[minus_iq] = [
            ix for el in minus_q_mapping[minus_iq] for ix in np.where(el == indx_delta_gs)[0]
        ]

    return minus_q_mapping, minus_q_mapping_unique


def build_g_vector_mappings_single_translation(
    cell: gto.Cell, kpts: npt.NDArray, kpts_pq: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Build g-vector mappings that map k-point differences to 1BZ.

    Args:
      cell: pyscf.pbc.gto.Cell object.
      kpts: array of kpoints.
      kpts_pq: Unique list of kp - kq indices of shape [num_unique_pq, 2].

    Returns:
        G_vectors: list of all 27 G-vectors
        Gpqr_mapping: [iq, kr]-th element yields indx_Gpqr, where Gpqr =
            kpts[ikp] - kpts[ikq] + kpts[ikr] - kpts[iks], i.e. returns index to
            G_vectors (consistent with G_vectors) satisfying this condition.
        Gpqr_mapping_unique: provides mapping to unique G_vector index.
        Delta_gs: provides compressed lists of unique G vectors.

    """
    g_dict, g_vectors = build_g_vectors(cell)
    lattice_vectors = cell.lattice_vectors()
    num_kpts = len(kpts)
    gpqr_mapping = np.zeros((len(kpts_pq), num_kpts), dtype=np.int32)
    kconserv = get_kconserv(cell, kpts)
    for iq, (ikp, ikq) in enumerate(kpts_pq):
        q = kpts[ikp] - kpts[ikq]
        for ikr in range(num_kpts):
            iks = kconserv[ikp, ikq, ikr]
            delta_gpqr = q + kpts[ikr] - kpts[iks]
            miller_indx = np.rint(np.einsum("wx,x->w", lattice_vectors, delta_gpqr) / (2 * np.pi))
            gpqr_mapping[iq, ikr] = g_dict[tuple(miller_indx)]

    gpqr_mapping_unique, delta_gs = find_unique_g_vectors(g_vectors, gpqr_mapping)
    return g_vectors, gpqr_mapping, gpqr_mapping_unique, delta_gs


def inverse_g_map_double_translation(
    cell: gto.Cell, kpts: npt.NDArray, momentum_map: npt.NDArray
) -> npt.NDArray:
    """for given q and g figure out all k which satisfy q - k + g = 0

    Args:
        cell: pyscf.pbc.gto.Cell object.
        kpts: array of kpoints.
        momentum_map: momentum mapping to satisfy Q = (k_p - k_q) mod G.
            momentum_map[iq, ikp] = ikq.

    Returns:
        inverse_map: ragged numpy array. inverse_map[iq, iG] returns array
            of size in range(0, num_kpts) and lists all k-point indices that
            satisfy G_pq[iq, ik] = iG, i.e. an array of all ik.
    """
    g_dict, g_vectors = build_g_vectors(cell)
    lattice_vectors = cell.lattice_vectors()
    num_kpts = len(kpts)
    gpq_mapping = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    num_kpts = len(kpts)
    for iq in range(num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            delta_gpq = (kpts[ikp] - kpts[ikq]) - kpts[iq]
            miller_indx = np.rint(np.einsum("wx,x->w", lattice_vectors, delta_gpq) / (2 * np.pi))
            gpq_mapping[iq, ikp] = g_dict[tuple(miller_indx)]

    inverse_map = np.zeros((num_kpts, 27), dtype=object)
    for iq in range(num_kpts):
        for ig in range(len(g_vectors)):
            inverse_map[iq, ig] = np.array(
                [ik for ik in range(num_kpts) if gpq_mapping[iq, ik] == ig]
            )

    return inverse_map


def build_eri_isdf_double_translation(
    chi: npt.NDArray, zeta: npt.NDArray, q_indx: int, kpts_indx: list, g_mapping: npt.NDArray
) -> npt.NDArray:
    """Build (pkp qkq | rkr sks) from k-point ISDF factors.

    Args:
        chi: array of interpolating orbitals of shape
            [num_kpts, num_mo, num_interp_points]
        zeta: central tensor of dimension
            [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
        q_indx: Index of momentum transfer.
        kpts_indx: List of kpt indices corresponding to [kp, kq, kr, ks]
        G_mapping: array to map kpts to G vectors [q_indx, kp] = G_pq

    Returns:
      eri:  (pkp qkq | rkr sks)
    """
    ikp, ikq, ikr, iks = kpts_indx
    gpq = g_mapping[q_indx, ikp]
    gsr = g_mapping[q_indx, iks]
    eri = np.einsum(
        "pm,qm,mn,rn,sn->pqrs",
        chi[ikp].conj(),
        chi[ikq],
        zeta[q_indx][gpq, gsr],
        chi[ikr].conj(),
        chi[iks],
        optimize=True,
    )
    return eri


def build_eri_isdf_single_translation(
    chi: npt.NDArray, zeta: npt.NDArray, q_indx: int, kpts_indx: list, g_mapping: npt.NDArray
) -> npt.NDArray:
    """Build (pkp qkq | rkr sks) from k-point ISDF factors.

    Args:
        chi: array of interpolating orbitals of shape
            [num_kpts, num_mo, num_interp_points]
        zeta: central tensor of dimension
            [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
        q_indx: Index of momentum transfer.
        kpts_indx: List of kpt indices corresponding to [kp, kq, kr, ks]
        G_mapping: array to map kpts to G vectors [q_indx, kp] = G_pq

    Returns:
        eri:  (pkp qkq | rkr sks)

    """
    ikp, ikq, ikr, iks = kpts_indx
    delta_g_indx = g_mapping[q_indx, ikr]
    eri = np.einsum(
        "pm,qm,mn,rn,sn->pqrs",
        chi[ikp].conj(),
        chi[ikq],
        zeta[q_indx][delta_g_indx],
        chi[ikr].conj(),
        chi[iks],
        optimize=True,
    )
    return eri


def kpoint_isdf_double_translation(
    df_inst: df.FFTDF,
    interp_indx: npt.NDArray,
    kpts: npt.NDArray,
    orbitals: npt.NDArray,
    grid_points: npt.NDArray,
    only_unique_g: bool = True,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Build kpoint ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    For the double translation case we build zeta[Q, G, G'] for all possible G
    and G' that satisfy Q - (Q-k) = G. If only_unique_G is True we only build
    the unique G's which satisfiy this expression rather than all 27^2.

    Args:
        df_inst: instance of pyscf.pbc.df.FFTDF object.
        interp_indx: array indexing interpolating points determined through
            K-Means CVT procedure. Dimension [num_interp_points]
        kpts: Array of k-points.
        orbitals: orbitals on a grid of shape [num_grid_points,
            num_orbitals], note num_orbitals = N_k * m, where m is the number of
            orbitals in the unit cell and N_k is the number of k-points.
        grid_points: Real space grid. Dimension [num_grid_points, num_dim],
            num_dim is 1, 2 or 3 for 1D, 2D, 3D.
        only_unique_G: Only build central tensor for unique Gs which satisfy
            momentum conservation condition.

    Returns:
        chi: orbitals on interpolating points
        zeta: THC central tensor. Dimension is
            [num_kpts, 27, 27, num_interp_points, num_interp_points]
            if only_unique_G is False otherwise it is of shape [num_kpts,
            num_unique[q], num_unique[q], 27, num_interp_points,
            num_interp_points].
        theta: matrix of interpolating vectors of dimension [num_grid_points,
            num_interp_points] (also called xi_mu(r)), where num_grid_points is
            the number of real space grid points and num_interp_points is the
            number of interpolating points. zeta (the
        g_mapping: g_mapping maps k-points to the appropriate delta_g index, i.e
            g_mapping[iq, ik] = i_delta_g. the index will map to the appropriate
            index in the reduced set of g vectors.
    """
    num_grid_points = len(grid_points)
    assert orbitals.shape[0] == num_grid_points
    xi, chi = solve_isdf(orbitals, interp_indx)
    momentum_map = build_momentum_transfer_mapping(df_inst.cell, kpts)
    num_kpts = len(kpts)
    num_interp_points = xi.shape[1]
    assert xi.shape == (num_grid_points, num_interp_points)
    (
        g_vectors,
        g_mapping,
        g_mapping_unique,
        delta_gs_unique,
    ) = build_g_vector_mappings_double_translation(df_inst.cell, kpts, momentum_map)
    if only_unique_g:
        g_mapping = g_mapping_unique
        delta_gs = delta_gs_unique
    else:
        delta_gs = [g_vectors] * num_kpts
        g_mapping = g_mapping
    zeta = np.zeros((num_kpts,), dtype=object)
    for iq in range(num_kpts):
        num_g = len(delta_gs[iq])
        out_array = np.zeros(
            (num_g, num_g, num_interp_points, num_interp_points), dtype=np.complex128
        )
        for ig, delta_g in enumerate(delta_gs[iq]):
            for ig_prime, delta_g_prime in enumerate(delta_gs[iq]):
                zeta_indx = build_kpoint_zeta(
                    df_inst, kpts[iq], delta_g, delta_g_prime, grid_points, xi
                )
                out_array[ig, ig_prime] = zeta_indx
        zeta[iq] = out_array
    return chi, zeta, xi, g_mapping


def kpoint_isdf_single_translation(
    df_inst: df.FFTDF,
    interp_indx: npt.NDArray,
    kpts: npt.NDArray,
    orbitals: npt.NDArray,
    grid_points: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Build kpoint ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    For the double translation case we build zeta[Q, G, G'] for all possible G
    and G' that satisfy Q - (Q-k) = G. If only_unique_G is True we only build
    the unique G's which satisfiy this expression rather than all 27^2.

    Args:
        df_inst: instance of pyscf.pbc.df.FFTDF object.
        interp_indx: array indexing interpolating points determined through
            K-Means CVT procedure. Dimension [num_interp_points]
        kpts: Array of k-points.
        orbitals: orbitals on a grid of shape [num_grid_points,
            num_orbitals], note num_orbitals = N_k * m, where m is the number of
            orbitals in the unit cell and N_k is the number of k-points.
        grid_points: Real space grid. Dimension [num_grid_points, num_dim],
            num_dim is 1, 2 or 3 for 1D, 2D, 3D.

    Returns:
        chi: orbitals on interpolating points
        zeta: THC central tensor. Dimension is
            [num_kpts, 27, 27, num_interp_points, num_interp_points]
            if only_unique_G is False otherwise it is of shape [num_kpts,
            num_unique[Q], num_unique[Q], 27, num_interp_points,
            num_interp_points].
        Theta: Matrix of interpolating vectors of dimension [num_grid_points,
            num_interp_points] (also called xi_mu(r)), where num_grid_points is
            the number of real space grid points and num_interp_points is the
            number of interpolating points. Zeta (the
        G_mapping: G_mapping maps k-points to the appropriate delta_G index, i.e
            G_mapping[iq, ik] = i_delta_G. the index will map to the appropriate
            index in the reduced set of G vectors.
    """
    num_grid_points = len(grid_points)
    assert orbitals.shape[0] == num_grid_points
    xi, chi = solve_isdf(orbitals, interp_indx)
    num_kpts = len(kpts)
    num_interp_points = xi.shape[1]
    assert xi.shape == (num_grid_points, num_interp_points)
    kpts_pq = np.array([(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)])
    kpts_pq_indx = np.array([(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)])
    transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
    unique_q, unique_indx, _ = unique(transfers)
    _, _, g_map_unique, delta_gs = build_g_vector_mappings_single_translation(
        df_inst.cell, kpts, kpts_pq_indx[unique_indx]
    )
    num_q_vectors = len(unique_q)
    zeta = np.zeros((num_q_vectors,), dtype=object)
    for iq in range(len(unique_q)):
        num_g = len(delta_gs[iq])
        out_array = np.zeros((num_g, num_interp_points, num_interp_points), dtype=np.complex128)
        for ig, delta_g in enumerate(delta_gs[iq]):
            zeta_indx = build_kpoint_zeta_single_tranlsation(
                df_inst, unique_q[iq], delta_g, grid_points, xi
            )
            out_array[ig] = zeta_indx
        zeta[iq] = out_array
    return chi, zeta, xi, g_map_unique


def build_isdf_orbital_inputs(mf_inst: scf.RHF) -> npt.NDArray:
    """Build orbital product inputs from mean field object

    Args:
        mf_inst: pyscf pbc mean-field object.
        mf_inst: scf.RHF:

    Returns:
        cell_periodic_mo: cell periodic part of Bloch orbital on real space grid
            of shape [num_grid_points, num_kpts*num_mo]

    """
    cell = mf_inst.cell
    kpts = mf_inst.kpts
    num_kpts = len(kpts)
    grid_points = cell.gen_uniform_grids(mf_inst.with_df.mesh)
    num_mo = mf_inst.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_grid_points = grid_points.shape[0]
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf_inst.mo_coeff, optimize=True
    )
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    return cell_periodic_mo


def density_guess(
    density: npt.NDArray, grid_inst: UniformGrids, grid_points: npt.NDArray, num_interp_points: int
) -> npt.NDArray:
    """Select initial centroids based on electronic density.

    Args:
        density: Density on real space grid.
        grid_inst: pyscf UniformGrids object.
        grid_points: Real space grid points.
        num_interp_points: Number of interpolating points.

    Returns:
        grid_points: Grid points sampled using density as a weighting function.

    """
    norm_factor = np.einsum("R,R->", density, grid_inst.weights).real
    prob_dist = (density.real * grid_inst.weights) / norm_factor
    indx = np.random.choice(len(grid_points), num_interp_points, replace=False, p=prob_dist)
    return grid_points[indx]


def interp_indx_from_qrcp(
    Z: npt.NDArray, num_interp_pts: npt.NDArray, return_diagonal: bool = False
):
    """Find interpolating points via QRCP

    Z^T P = Q R, where R has diagonal elements that are in descending order of
    magnitude. Interpolating points are then chosen by the columns of the
    permuted Z^T.

    Args:
        Z: Orbital pair product matrix.
        num_interp_pts: integer corresponding to number of interpolating points
            to select from full real space grid.
        return_diagonal: bool:  (Default value = False)

    Returns:
        interp_indx: Index of interpolating points in full real space grid.
    """

    Q, R, P = scipy.linalg.qr(Z.T, pivoting=True)
    signs_R = np.diag(np.sign(R.diagonal()))
    # Force diagonal of R to be positive which isn't strictly enforced in QR
    # factorization.
    R = np.dot(signs_R, R)
    Q = np.dot(Q, signs_R)
    interp_indx = P[:num_interp_pts]
    if return_diagonal:
        return interp_indx, R.diagonal()
    else:
        return interp_indx


def setup_isdf(
    mf_inst: scf.RHF, verbose: bool = False
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Setup common data for ISDF solution.

    Args:
        mf_inst: pyscf pbc mean-field object.
        verbose: Whether to print some information.
        mf_inst: scf.RHF:
        verbose: bool:  (Default value = False)

    Returns:
        grid_points: Real space grid points.
        cell_periodic_mo: Cell periodic part of MOs on a grid.
            [num_grid, num_kpts * num_orb]
        bloch_orbitals_mo: MOs on a grid. [num_grid, num_kpts, num_orb]

    """
    assert isinstance(mf_inst.with_df, df.FFTDF), "mf object must use FFTDF"
    cell = mf_inst.cell
    kpts = mf_inst.kpts
    grid_points = cell.gen_uniform_grids(mf_inst.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    if verbose:
        print("Real space grid shape: ({}, {})".format(grid_points.shape[0], grid_points.shape[1]))
        print("Number of grid points: {}".format(num_grid_points))
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf_inst.mo_coeff, optimize=True
    )
    num_mo = mf_inst.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_kpts = len(kpts)
    # Cell periodic part
    # u = e^{-ik.r} phi(r)
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    return grid_points, cell_periodic_mo, bloch_orbitals_mo


@dataclass
class KPointTHC:
    """Light class to hold THC tensors.

    Atributes:
        chi: THC leaf tensor. shape = [num_kpts, num_mo, num_thc]
        zeta: THC central tensor.
          shape = [num_kpts, num_G, num_G, num_thc, num_thc]
        G_mapping: For a given Q and k index this array gives the G index for
          zeta.
        xi: ISDF interpolating vectors. May be none if this class holds
            reoptimized THC decomposition.

    Examples:
        The following pseudocode shows how you can build an ERI matrix block
        given elements of this class.

        >>> kthc = solve_kmeans_kpisdf(...)
        >>> ikp, ikq, ikr, iks = kpts_indx
        >>> Gpq = kthc.G_mapping[q_indx, ikp]
        >>> Gsr = kthc.G_mapping[q_indx, iks]
        >>> eri = np.einsum(
            "pm,qm,mn,rn,sn->pqrs",
            kthc.chi[ikp].conj(),
            kthc.chi[ikq],
            kthc.zeta[q_indx][Gpq, Gsr],
            kthc.chi[ikr].conj(),
            kthc.chi[iks],
            optimize=True,
        )
    """

    chi: npt.NDArray[np.complex128]
    zeta: npt.NDArray
    g_mapping: npt.NDArray
    xi: Union[npt.NDArray[np.complex128], None]

    @property
    def num_interp_points(self) -> int:
        """Number of interpolating points (THC dimension)"""
        return self.chi.shape[-1]

    @property
    def num_thc_factors(self) -> int:
        """Number of interpolating points (THC dimension)"""
        return self.chi.shape[-1]


def solve_kmeans_kpisdf(
    mf_inst: scf.RHF,
    num_interp_points: int,
    max_kmeans_iteration: int = 500,
    single_translation: bool = False,
    use_density_guess: bool = True,
    kmeans_weighting_function: str = "density",
    verbose: bool = True,
) -> KPointTHC:
    r"""Solve for k-point THC factors using k-means CVT ISDF procedure.

    Args:
        mf_inst: pyscf pbc mean-field object.
        num_interp_points: Number of interpolating points (THC rank M).
        max_kmeans_iteration: Max iteration for k-means CVT algorithm.
        single_translation: Build THC factors assuming single translation of kp
            - kq. If true we build zeta[Q, G], else zeta[Q, G, G'].
        use_density_guess: Select initial grid points according to electron
            density? Default True.
        kmeans_weighting_function: Weighting function to use in k-means CVT.
            One of ["density", "orbital_density", "sum_squares"].
        verbose: Whether to print some information.

    Returns:
        solution: THC factors held in KPointTHC object.
    """
    if verbose:
        print(f" Number of interpolating points: {num_interp_points}")
        print(f" Max K-Means iteration: {max_kmeans_iteration}")
        print(f" Use density initial guess: {density_guess}")
        print(f" K-Means weighting function: {kmeans_weighting_function}")
        print(f" Single Translation: {single_translation}")
    # Build real space grid, and orbitals on real space grid
    grid_points, cell_periodic_mo, bloch_orbitals_mo = setup_isdf(mf_inst)
    grid_inst = UniformGrids(mf_inst.cell)
    # Find interpolating points using K-Means CVT algorithm
    kmeans = KMeansCVT(grid_points, max_iteration=max_kmeans_iteration)
    nocc = mf_inst.cell.nelec[0]  # assuming same for each k-point
    density = np.einsum(
        "kRi,kRi->R",
        bloch_orbitals_mo[:, :, :nocc].conj(),
        bloch_orbitals_mo[:, :, :nocc],
        optimize=True,
    )
    if use_density_guess:
        initial_centroids = density_guess(density, grid_inst, grid_points, num_interp_points)
    else:
        initial_centroids = None
    weighting_function = None
    # Build weighting function for CVT algorithm.
    if kmeans_weighting_function == "density":
        weighting_function = density
    elif kmeans_weighting_function == "orbital_density":
        # w(r) = sum_{ij} |phi_i(r)| |phi_j(r)|
        tmp = np.einsum(
            "kRi,pRj->Rkipj", abs(bloch_orbitals_mo), abs(bloch_orbitals_mo), optimize=True
        )
        weighting_function = np.einsum("Rkipj->R", tmp)
    elif kmeans_weighting_function == "sum_squares":
        # w(r) = sum_{i} |phi_{ki}(r)|
        weighting_function = np.einsum(
            "kRi,kRi->R", bloch_orbitals_mo.conj(), bloch_orbitals_mo, optimize=True
        )
        weighting_function = weighting_function
    else:
        raise ValueError(f"Unknown value for weighting function {kmeans_weighting_function}")
    interp_indx = kmeans.find_interpolating_points(
        num_interp_points, weighting_function.real, verbose=verbose, centroids=initial_centroids
    )
    solution = solve_for_thc_factors(
        mf_inst,
        interp_indx,
        cell_periodic_mo,
        grid_points,
        single_translation=single_translation,
        verbose=verbose,
    )
    return solution


def solve_qrcp_isdf(
    mf_inst: scf.RHF, num_interp_points: int, single_translation: bool = True, verbose: bool = True
) -> KPointTHC:
    r"""Solve for k-point THC factors using QRCP ISDF procedure.

    Args:
        mf_inst: pyscf pbc mean-field object.
        num_interp_points: Number of interpolating points (THC rank M).
        single_translation: Build THC factors assuming single translation of kp
        single_translation: Build THC factors assuming single translation of kp
            - kq. If true we build zeta[Q, G], else zeta[Q, G, G'].
        verbose: Whether to print some information.

    Returns:
      solution: THC factors held in KPointTHC object.
    """
    # Build real space grid, and orbitals on real space grid
    grid_points, cell_periodic_mo, _ = setup_isdf(mf_inst)
    # Find interpolating points using K-Means CVT algorithm
    # Z_{R, (ki)(k'j)} = u_{ki}(r)* u_{k'j}(r)
    num_grid_points = len(grid_points)
    num_orbs = cell_periodic_mo.shape[1]
    pair_products = np.einsum(
        "Ri,Rj->Rij", cell_periodic_mo.conj(), cell_periodic_mo, optimize=True
    ).reshape((num_grid_points, num_orbs**2))
    interp_indx = interp_indx_from_qrcp(pair_products, num_interp_points)
    # Solve for THC factors.
    solution = solve_for_thc_factors(
        mf_inst,
        interp_indx,
        cell_periodic_mo,
        grid_points,
        single_translation=single_translation,
        verbose=verbose,
    )
    return solution


def solve_for_thc_factors(
    mf_inst,
    interp_points_index,
    cell_periodic_mo,
    grid_points,
    single_translation=True,
    verbose=True,
) -> KPointTHC:
    r"""Solve for k-point THC factors using interpolating points as input.

    Args:
        mf_inst: pyscf pbc mean-field object.
        interp_points_index: Indices of interpolating points found from k-means
            CVT or QRCP.
        cell_periodic_mo: cell periodic part of Bloch orbital on real space grid
            shape: [num_grid_points, num_kpts*num_mo]
        kmeans_weighting_function: Weighting function to use in k-means CVT.
            One of ["density", "orbital_density", "sum_squares"].
        grid_points: Real space grid. Dimension [num_grid_points, num_dim],
                num_dim is 1, 2 or 3 for 1D, 2D, 3D.
        single_translation: Build THC factors assuming single translation of kp
            - kq. If true we build zeta[Q, G], else zeta[Q, G, G'].
            (Default value = True)
        verbose: Whether to print some information. (Default value = True)

    Returns:
        solution: THC factors held in KPointTHC object.
    """
    assert isinstance(mf_inst.with_df, df.FFTDF), "mf object must use FFTDF"
    if single_translation:
        chi, zeta, xi, g_mapping = kpoint_isdf_single_translation(
            mf_inst.with_df, interp_points_index, mf_inst.kpts, cell_periodic_mo, grid_points
        )
    else:
        chi, zeta, xi, g_mapping = kpoint_isdf_double_translation(
            mf_inst.with_df, interp_points_index, mf_inst.kpts, cell_periodic_mo, grid_points
        )
    num_interp_points = len(interp_points_index)
    num_kpts = len(mf_inst.kpts)
    num_mo = mf_inst.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    assert chi.shape == (num_interp_points, num_kpts * num_mo)
    # go from Rki -> kiR
    chi = chi.reshape((num_interp_points, num_kpts, num_mo))
    chi = chi.transpose((1, 2, 0))

    solution = KPointTHC(chi=chi, zeta=zeta, xi=xi, g_mapping=g_mapping)

    return solution
