# coverage:ignore
""" Compute lambda for double low rank factoriz. method of von Burg, et al, with BLISS post-processing of DF fragments"""
import numpy as np
from openfermion.resource_estimates.molecule import pyscf_to_cas


def compute_lambda_df_with_bliss_post_processing(pyscf_mf, df_factors):
    """Compute lambda for Hamiltonian using DF method of von Burg, et al. where DF is post-processed with a low-rank-preserving BLISS operator

    Args:
        pyscf_mf - Pyscf mean field object
        df_factors (ndarray) - (N x N x rank) array of DF factors from ERI

    Returns:
        lambda_tot (float) - lambda value for the double factorized Hamiltonian
    """
    # grab tensors from pyscf_mf object
    h1, eri_full, _, num_alpha, num_beta = pyscf_to_cas(pyscf_mf)
    num_elec = num_alpha + num_beta
    num_orb = h1.shape[0]

    # two body contributions
    lambda_F = 0.0
    h1_correction_BLISS = np.zeros([num_orb, num_orb])
    for vector in range(df_factors.shape[2]):
        Lij = df_factors[:, :, vector]
        e = np.linalg.eigvalsh(Lij)
        s = np.median(e)
        lambda_F += 0.25 * np.sum(np.abs(e - s)) ** 2
        h1_correction_BLISS += s * (num_elec - num_orb) * Lij

    # one body contributions
    T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum("llij->ij", eri_full) + h1_correction_BLISS
    e, _ = np.linalg.eigh(T)
    s = np.median(e)
    lambda_T = np.sum(np.abs(e - s))

    lambda_tot = lambda_T + lambda_F

    return lambda_tot