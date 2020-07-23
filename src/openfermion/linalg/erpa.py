"""Code to generate the eigenvalue problem for the ERPA equations"""
from typing import Dict, Tuple, Union
from itertools import product
import numpy
from numpy import einsum
import scipy
from openfermion.utils.rdm_mapping_functions import kronecker_delta as kdelta
from openfermion.utils.rdm_mapping_functions import map_two_pdm_to_one_pdm


def erpa_eom_hamiltonian(h_ijkl: numpy.ndarray, tpdm: numpy.ndarray, p: int,
                         q: int, r: int, s: int) -> Union[float, complex]:
    """
    Evaluate sum_{a,b,c,d}h_{a, b, d, c}<psi[p^ q, [a^ b^ c d, r^ s]]psi>

    Args:
        h_ijkl: two-body integral tensors of full reduced Hamiltonian
        tpdm: two-RDM
        p: left creation op index
        q: left annihilation op index
        r: right creation op index
        s: right annihilation op index
    Returns:
        float or complex number
    """
    h_mat = 0
    #  (   1.00000) h_ijkl(q,s,a,b) cre(p) cre(r) des(a) des(b)
    h_mat += 1.0 * einsum('ab,ab', h_ijkl[q, s, :, :], tpdm[p, r, :, :])

    #  (  -1.00000) h_ijkl(q,a,r,b) cre(p) cre(a) des(s) des(b)
    h_mat += -1.0 * einsum('ab,ab', h_ijkl[q, :, r, :], tpdm[p, :, s, :])

    #  (   1.00000) h_ijkl(q,a,b,r) cre(p) cre(a) des(s) des(b)
    h_mat += 1.0 * einsum('ab,ab', h_ijkl[q, :, :, r], tpdm[p, :, s, :])

    #  (  -1.00000) h_ijkl(s,q,a,b) cre(p) cre(r) des(a) des(b)
    h_mat += -1.0 * einsum('ab,ab', h_ijkl[s, q, :, :], tpdm[p, r, :, :])

    #  (  -1.00000) h_ijkl(s,a,p,b) cre(r) cre(a) des(q) des(b)
    h_mat += -1.0 * einsum('ab,ab', h_ijkl[s, :, p, :], tpdm[r, :, q, :])

    #  (   1.00000) h_ijkl(s,a,b,p) cre(r) cre(a) des(q) des(b)
    h_mat += 1.0 * einsum('ab,ab', h_ijkl[s, :, :, p], tpdm[r, :, q, :])

    #  (   1.00000) h_ijkl(a,q,r,b) cre(p) cre(a) des(s) des(b)
    h_mat += 1.0 * einsum('ab,ab', h_ijkl[:, q, r, :], tpdm[p, :, s, :])

    #  (  -1.00000) h_ijkl(a,q,b,r) cre(p) cre(a) des(s) des(b)
    h_mat += -1.0 * einsum('ab,ab', h_ijkl[:, q, :, r], tpdm[p, :, s, :])

    #  (   1.00000) h_ijkl(a,s,p,b) cre(r) cre(a) des(q) des(b)
    h_mat += 1.0 * einsum('ab,ab', h_ijkl[:, s, p, :], tpdm[r, :, q, :])

    #  (  -1.00000) h_ijkl(a,s,b,p) cre(r) cre(a) des(q) des(b)
    h_mat += -1.0 * einsum('ab,ab', h_ijkl[:, s, :, p], tpdm[r, :, q, :])

    #  (   1.00000) h_ijkl(a,b,p,r) cre(a) cre(b) des(q) des(s)
    h_mat += 1.0 * einsum('ab,ab', h_ijkl[:, :, p, r], tpdm[:, :, q, s])

    #  (  -1.00000) h_ijkl(a,b,r,p) cre(a) cre(b) des(q) des(s)
    h_mat += -1.0 * einsum('ab,ab', h_ijkl[:, :, r, p], tpdm[:, :, q, s])

    if q == r:
        #  (   1.00000) h_ijkl(s,a,b,c) kdelta(q,r) cre(p) cre(a) des(b) des(c)
        h_mat += 1.0 * einsum('abc,abc', h_ijkl[s, :, :, :], tpdm[p, :, :, :])

        #  (  -1.00000) h_ijkl(a,s,b,c) kdelta(q,r) cre(p) cre(a) des(b) des(c)
        h_mat += -1.0 * einsum('abc,abc', h_ijkl[:, s, :, :], tpdm[p, :, :, :])

    if p == s:
        #  (   1.00000) h_ijkl(a,b,r,c) kdelta(p,s) cre(a) cre(b) des(q) des(c)
        h_mat += 1.0 * einsum('abc,abc', h_ijkl[:, :, r, :], tpdm[:, :, q, :])

        #  (  -1.00000) h_ijkl(a,b,c,r) kdelta(p,s) cre(a) cre(b) des(q) des(c)
        h_mat += -1.0 * einsum('abc,abc', h_ijkl[:, :, :, r], tpdm[:, :, q, :])

    return h_mat


def singlet_erpa(tpdm: numpy.ndarray, h_ijkl: numpy.ndarray) \
        -> Tuple[numpy.ndarray, numpy.ndarray, Dict]:
    """
    Generate the singlet ERPA equations

    [ea + eb, [H, sa, sb]] = [ea, [H, sa]]

    The erpa equations are solved with scipy.linalg.eig which calls
    lapack's geev

    Args:
        tpdm: 2-RDM tensor normal OpenFermion format
        h_ijkl: reduced Hamiltonian tensor
    Returns:
        Tuple of the erpa system.
    """
    permuted_hijkl = numpy.einsum('ijlk', h_ijkl)
    roots = numpy.array(numpy.roots([1, -1, -einsum('ijji', tpdm)]))
    n_electrons = roots[numpy.where(roots > 0)[0]].real
    opdm = map_two_pdm_to_one_pdm(tpdm, n_electrons)
    dim = tpdm.shape[0] // 2  # dim = num spatial orbitals
    full_basis = {}  # erpa basis.  A, B basis in RPA language
    cnt = 0
    for p, q in product(range(dim), repeat=2):
        if p < q:
            full_basis[(p, q)] = cnt
            full_basis[(q, p)] = cnt + dim * (dim - 1) // 2
            cnt += 1

    erpa_mat = numpy.zeros((len(full_basis), len(full_basis)))
    metric_mat = numpy.zeros((len(full_basis), len(full_basis)))
    for rkey, ridx in full_basis.items():
        p, q = rkey
        for ckey, cidx in full_basis.items():
            r, s = ckey
            for sigma, tau in product([0, 1], repeat=2):
                erpa_mat[ridx, cidx] += 0.5 * erpa_eom_hamiltonian(
                    permuted_hijkl, tpdm, 2 * q + sigma, 2 * p + sigma,
                    2 * r + tau, 2 * s + tau).real
                metric_mat[ridx, cidx] += 0.5 * (
                    opdm[2 * q + sigma, 2 * s + tau] *
                    kdelta(2 * r + tau, 2 * p + sigma) -
                    opdm[2 * p + sigma, 2 * r + tau] *
                    kdelta(2 * q + sigma, 2 * s + tau)).real

    # The metric is hermetian and can be diagonalized
    # this allows us to project into the non-zero eigenvalue space
    ws, vs = numpy.linalg.eigh(metric_mat)
    non_zero_idx = numpy.where(numpy.abs(ws) > 1.0E-8)[0]
    left_mat = vs[:, non_zero_idx].T @ erpa_mat @ vs[:, non_zero_idx]
    right_mat = vs[:, non_zero_idx].T @ metric_mat @ vs[:, non_zero_idx]

    # solve the matrix pencil using ddgev in lapack
    w, v = scipy.linalg.eig(left_mat, right_mat)

    # the spectrum is symmetric (-w, w)
    # find the positive spectrum eigensystem and return
    real_eig_idx = numpy.where(numpy.abs(w.imag) < 1.0E-8)[0]
    real_eigs = w[real_eig_idx]
    real_eig_vecs = v[:, real_eig_idx]
    reverse_projected_eig_vecs = vs[:, non_zero_idx] @ real_eig_vecs
    pos_indices = numpy.where(real_eigs > 0)[0]
    pos_indices = pos_indices[numpy.argsort(real_eigs[pos_indices])]
    return real_eigs[pos_indices], reverse_projected_eig_vecs[:, pos_indices], \
           full_basis
