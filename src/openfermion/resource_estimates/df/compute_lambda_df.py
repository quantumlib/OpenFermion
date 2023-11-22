# coverage:ignore
""" Compute lambda for double low rank factoriz. method of von Burg, et al """
import numpy as np
from openfermion.resource_estimates.molecule import pyscf_to_cas


def compute_lambda(pyscf_mf, df_factors):
    """Compute lambda for Hamiltonian using DF method of von Burg, et al.

    Args:
        pyscf_mf - Pyscf mean field object
        df_factors (ndarray) - (N x N x rank) array of DF factors from ERI

    Returns:
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """
    # grab tensors from pyscf_mf object
    h1, eri_full, _, _, _ = pyscf_to_cas(pyscf_mf)

    # one body contributions
    T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum("llij->ij", eri_full)
    e, _ = np.linalg.eigh(T)
    lambda_T = np.sum(np.abs(e))

    # two body contributions
    lambda_F = 0.0
    for vector in range(df_factors.shape[2]):
        Lij = df_factors[:, :, vector]
        # e, v = np.linalg.eigh(Lij)
        e = np.linalg.eigvalsh(Lij)  # just need eigenvals
        lambda_F += 0.25 * np.sum(np.abs(e)) ** 2

    lambda_tot = lambda_T + lambda_F

    return lambda_tot
