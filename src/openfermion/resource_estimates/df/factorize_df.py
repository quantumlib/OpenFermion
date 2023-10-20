# coverage:ignore
""" Double factorization rank reduction of ERIs """
import numpy as np
from openfermion.resource_estimates.utils import eigendecomp


def factorize(eri_full, thresh):
    """Do double factorization of the ERI tensor

    Args:
       eri_full (np.ndarray) - 4D (N x N x N x N) full ERI tensor
       thresh (float) - threshold for double factorization

    Returns:
       eri_rr (np.ndarray) - 4D approximate ERI tensor reconstructed
           from df_factors vectors
       df_factors (np.ndarray) - 3D (N x N x M) tensor containing DF vectors
       rank (int) - rank retained from initial eigendecomposition
       num_eigenvectors (int) - number of eigenvectors
    """

    n_orb = eri_full.shape[0]
    assert n_orb**4 == len(eri_full.flatten())

    # First, do an eigendecomposition of ERIs
    L = eigendecomp(eri_full.reshape(n_orb**2, n_orb**2), tol=0.0)
    L = L.reshape(n_orb, n_orb, -1)  # back to (N x N x rank)

    sf_rank = L.shape[2]

    num_eigenvectors = 0  # rolling number of eigenvectors
    df_factors = []  # collect the selected vectors
    for rank in range(sf_rank):
        Lij = L[:, :, rank]
        e, v = np.linalg.eigh(Lij)
        normSC = np.sum(np.abs(e))

        truncation = normSC * np.abs(e)

        idx = truncation > thresh
        plus = np.sum(idx)
        num_eigenvectors += plus

        if plus == 0:
            break

        e_selected = np.diag(e[idx])
        v_selected = v[:, idx]

        Lij_selected = v_selected.dot(e_selected).dot(v_selected.T)
        df_factors.append(Lij_selected)

    # raw factors from DF algorithm
    df_factors = np.asarray(df_factors).T

    # double-factorized and re-constructed ERIs
    eri_rr = np.einsum('ijP,klP', df_factors, df_factors, optimize=True)

    return eri_rr, df_factors, rank, num_eigenvectors
