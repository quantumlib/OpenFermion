# coverage:ignore
""" Compute lambda for single low rank factorization method of Berry, et al """

import numpy as np
from openfermion.resource_estimates.molecule import pyscf_to_cas


def compute_lambda(pyscf_mf, sf_factors):
    """Compute lambda for Hamiltonian using SF method of Berry, et al.

    Args:
        pyscf_mf - PySCF mean field object
        sf_factors (ndarray) - (N x N x rank) array of SF factors from rank
            reduction of ERI

    Returns:
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """

    # grab tensors from pyscf_mf object
    h1, eri_full, _, _, _ = pyscf_to_cas(pyscf_mf)

    # Effective one electron operator contribution
    T = (
        h1
        - 0.5 * np.einsum("pqqs->ps", eri_full, optimize=True)
        + np.einsum("pqrr->pq", eri_full, optimize=True)
    )

    lambda_T = np.sum(np.abs(T))

    # Two electron operator contributions
    lambda_W = 0.25 * np.einsum("ijP,klP->", np.abs(sf_factors), np.abs(sf_factors), optimize=True)
    lambda_tot = lambda_T + lambda_W

    return lambda_tot
