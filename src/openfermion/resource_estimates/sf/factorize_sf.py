#coverage:ignore
""" Single factorization of the ERI tensor """
import numpy as np
from openfermion.resource_estimates.utils import eigendecomp


def factorize(eri_full, rank):
    """ Do single factorization of the ERI tensor

    Args:
       eri_full (np.ndarray) - 4D (N x N x N x N) full ERI tensor
       rank (int) - number of vectors to retain in ERI rank-reduction procedure

    Returns:
       eri_rr (np.ndarray) - 4D approximate ERI tensor reconstructed from LR vec
       LR (np.ndarray) - 3D (N x N x rank) tensor containing SF vectors
    """
    n_orb = eri_full.shape[0]
    assert n_orb**4 == len(eri_full.flatten())

    L = eigendecomp(eri_full.reshape(n_orb**2, n_orb**2), tol=1e-16)

    # Do rank-reduction of ERIs following ERI factorization
    if rank is None:
        LR = L[:, :]
    else:
        LR = L[:, :rank]
    eri_rr = np.einsum('ik,kj->ij', LR, LR.T, optimize=True)
    eri_rr = eri_rr.reshape(n_orb, n_orb, n_orb, n_orb)
    LR = LR.reshape(n_orb, n_orb, -1)
    if rank is not None:
        try:
            assert LR.shape[2] == rank
        except AssertionError:
            sys.exit(
                "LR.shape:     %s\nrank: %s\nLR.shape and rank are inconsistent"
                % (LR.shape, rank))

    return eri_rr, LR
