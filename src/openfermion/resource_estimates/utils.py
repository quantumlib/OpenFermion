#coverage:ignore
""" Utilities for FT costing calculations """
from typing import Tuple, Optional
import sys
import os
import h5py
import numpy as np


def QR(L: int, M1: int) -> Tuple[int, int]:
    """ This gives the optimal k and minimum cost for a QROM over L values of
        size M.

    Args:
        L (int) -
        M1 (int) -

    Returns:
       k_opt (int) - k that yields minimal (optimal) cost of QROM
       val_opt (int) - minimal (optimal) cost of QROM
    """
    k = 0.5 * np.log2(L / M1)
    try:
        assert k >= 0
    except AssertionError:
        sys.exit("In function QR: \
        \n  L is smaller than M: increase RANK or lower THRESH \
        (or alternatively decrease CHI)")
    value = lambda k: L / np.power(2, k) + M1 * (np.power(2, k) - 1)
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)


def QR2(L1: int, L2: int, M1: int) -> Tuple[int, int, int]:
    """ This gives the optimal k values and minimum cost for a QROM using
        two L values of size M,
        e.g. the optimal k values for the QROM on two registers.
    Args:
        L1 (int) -
        L2 (int) -
        M1 (int) -

    Returns:
       k1_opt (int) - k1 that yields minimal (optimal) cost of QROM with two reg
       k2_opt (int) - k2 that yields minimal (optimal) cost of QROM with two reg
       val_opt (int) - minimal (optimal) cost of QROM
    """

    k1_opt, k2_opt = 0, 0
    val_opt = 1e50
    # Doing this as a stupid loop for now, worth refactoring but cost is quick
    # Biggest concern is if k1 / k2 range is not large enough!
    for k1 in range(1, 17):
        for k2 in range(1, 17):
            value = np.ceil(L1 / np.power(2, k1)) * np.ceil(L2 / \
                np.power(2, k2)) +\
                M1 * (np.power(2, k1 + k2) - 1)
            if value < val_opt:
                val_opt = value
                k1_opt = k1
                k2_opt = k2

    assert val_opt.is_integer()
    return int(np.power(2, k1_opt)), int(np.power(2, k2_opt)), int(val_opt)


def QI(L: int) -> Tuple[int, int]:
    """ This gives the opt k and minimum cost for an inverse QROM over L vals

    Args:
        L (int) -

    Returns:
       k_opt (int) - k that yiles minimal (optimal) cost of inverse QROM
       val_opt (int) - minimal (optimal) cost of inverse QROM
    """
    k = 0.5 * np.log2(L)
    assert k >= 0
    value = lambda k: L / np.power(2, k) + np.power(2, k)
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)


# Is this ever used? It's defined in costingsf.nb, but I don't it's ever called.
def QI2(L1: int, L2: int) -> Tuple[int, int, int]:
    """ This gives the optimal k values and minimum cost for inverse QROM
        using two L values,
        e.g. the optimal k values for the inverse QROM on two registers.

    Args:
        L1 (int) -
        L2 (int) -

    Returns:
       k1_opt (int) - k1 with minimal (optimal) cost of inverse QROM with 2 regs
       k2_opt (int) - k2 with minimal (optimal) cost of inverse QROM with 2 regs
       val_opt (int) - minimal (optimal) cost of inverse QROM with two registers
    """

    k1_opt, k2_opt = 0, 0
    val_opt = 1e50
    # Doing this as a stupid loop for now, worth refactoring but cost is quick
    # Biggest concern is if k1 / k2 range is not large enough!
    for k1 in range(1, 17):
        for k2 in range(1, 17):
            value = np.ceil(L1 / np.power(2, k1)) * np.ceil(L2 / \
                np.power(2, k2)) +\
                np.power(2, k1 + k2)
            if value < val_opt:
                val_opt = value
                k1_opt = k1
                k2_opt = k2

    assert val_opt.is_integer()
    return int(np.power(2, k1_opt)), int(np.power(2, k2_opt)), int(val_opt)


def power_two(m: int) -> int:
    """ Return the power of two that is a factor of m """
    assert m >= 0
    if m % 2 == 0:
        count = 0
        while (m > 0) and (m % 2 == 0):
            m = m // 2
            count += 1
        return count
    return 0


class RunSilent(object):
    """ Context manager to prevent function writing to stdout/stderr
        e.g. for noisy_function(), wrap it like so

        with RunSilent():
            noisy_function()

        ... and your terminal will no longer be littered with prints
    """

    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def eigendecomp(M, tol=1.15E-16):
    """ Decompose matrix M into L.L^T where rank(L) < rank(M) to some threshold

    Args:
       M (np.ndarray) - (N x N) positive semi-definite matrix to be decomposed
       tol (float) - eigenpairs with eigenvalue above tol will be kept

    Returns:
       L (np.ndarray) - (K x N) array such that K <= N and L.L^T = M
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Put in descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Truncate
    idx = np.where(eigenvalues > tol)[0]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    # eliminate eigenvalues from eigendecomposition
    L = np.einsum("ij,j->ij", eigenvectors, np.sqrt(eigenvalues))

    return L
