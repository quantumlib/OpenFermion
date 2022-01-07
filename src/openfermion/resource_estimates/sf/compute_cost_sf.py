#coverage:ignore
""" Determine costs for SF decomposition in QC """
from typing import Tuple
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # has analytic continutn to cplx
from openfermion.resource_estimates.utils import QR, QI, QR2, power_two


def compute_cost(n: int,
                 lam: float,
                 dE: float,
                 L: int,
                 chi: int,
                 stps: int,
                 verbose: bool = False) -> Tuple[int, int, int]:
    """ Determine fault-tolerant costs using SF decomposition in quantum chem

    Args:
        n (int) - the number of spin-orbitals
        lam (float) - the lambda-value for the Hamiltonian
        dE (float) - allowable error in phase estimation
        L (int) - the rank of the first decomposition
        chi (int) - equivalent to aleph_1 and aleph_2 in the document, the
            number of bits for the representation of the coefficients
        stps (int) - an approximate number of steps to choose the precision of
            single qubit rotations in preparation of the equal superposn state
        verbose (bool) - do additional printing of intermediates?

    Returns:
        step_cost (int) - Toffolis per step
        total_cost (int) - Total number of Toffolis
        total_qubit_count (int) - Total qubit count
    """

    # Number of trailing zeros
    eta = power_two(L + 1)

    # Number of qubits for the first register
    nL = np.ceil(np.log2(L + 1))

    # Number of qubits for p and q registers
    nN = np.ceil(np.log2(n // 2))

    oh = [0] * 20
    for p in range(20):
        # JJG note: arccos arg may be > 1
        v = np.round(np.power(2,p+1) / (2 * np.pi) * arccos(np.power(2,nL) /\
            np.sqrt((L + 1)/2**eta)/2))
        oh[p] = np.real(stps * (1 / (np.sin(3 * arcsin(np.cos(v * 2 * np.pi / \
            np.power(2,p+1)) * \
            np.sqrt((L + 1)/2**eta) / np.power(2,nL)))**2) - 1) + 4 * (p + 1))

    # Bits of precision for rotation
    br = int(np.argmin(oh) + 1)

    # The following costs are given in the list of steps on pages 43 and 44.

    # Cost of preparing the equal superposition state on the 1st register in 1a
    cost1a = 2 * (3 * nL - 3 * eta + 2 * br - 9)

    # We have added two qubits
    bL = nL + chi + 2

    # QROM costs for first register preparation in step 1(b)
    cost1b = QR(L + 1, bL)[-1] + QI(L + 1)[-1]

    # The inequality test of cost chi in step 1(c) and the controlled swap with
    # cost nL + 1 in step 1(d) (and their inversions).
    cost1cd = 2 * (chi + nL + 1)

    oh = [0] * 20
    nprime = int(n**2 // 8 + n // 4)
    for p in range(20):
        v = np.round(
            np.power(2, p + 1) / (2 * np.pi) *
            arccos(np.power(2, 2 * nN) / np.sqrt(nprime) / 2))
        oh[p] = np.real(20000 * (1 / (np.sin(3 * arcsin(np.cos(v * 2 * np.pi / \
            np.power(2, p+1)) * \
            np.sqrt(nprime) / np.power(2,2*nN)))**2) - 1) + 4 * (p + 1))

    # Bits of precision for rotation for preparing the next equal
    # superposition states
    br = int(np.argmin(oh) + 1)

    # Cost of preparing equal superposition of p and q registers in step 2(a).
    cost2a = 4 * (6 * nN + 2 * br - 7)

    # Cost of computing contiguous register in step 2 (b).
    cost2b = 4 * (nN**2 + nN - 1)

    # Number of coefficients for first state preparation on p & q.
    n1 = (L + 1) * nprime

    # Number of coefficients for second state preparation on p & q.
    n2 = L * nprime

    # The output size for the QROM for the second state preparation.
    bp = int(2 * nN + chi + 2)

    # Cost of QROMs for state preparation on p and q in step 2 (c).
    cost2c = QR2(L + 1, nprime, bp)[-1] + QI(n1)[-1] + QR2(L, nprime,
                                                           bp)[-1] + QI(n2)[-1]

    # The cost of the inequality test and controlled swap for the quantum alias
    # sampling in steps 2 (d) and (e).
    cost2de = 4 * (chi + 2 * nN)

    # Swapping the p & q registers for symmetry in step 3.  This needs to be
    # done and inverted twice, hence the factor of 4.
    cost3 = 4 * nN

    # The SELECT operation in step 4, which needs to be done twice.
    cost4 = 4 * n - 8

    # The reflection used on the 2nd register in the middle, with cost in step 6
    cost6 = 2 * nN + chi + 3

    # Step 7 involves performing steps 2 to 5 again, which are accounted for
    # above, but there is one extra Toffoli for checking that l <> 0.
    cost7 = 1

    # The total reflection for the quantum walk in step 9.
    cost9 = nL + 2 * nN + 2 * chi + 2

    # The two Toffolis for the control for phase estimation and iteration,
    # given in step 10.
    cost10 = 2

    # The number of steps.
    iters = np.ceil(np.pi * lam / (dE * 2))

    # The total Toffoli costs for a step.
    cost = cost1a + cost1b + cost1cd + cost2a + cost2b + cost2c + cost2de + \
           cost3 + cost4 + cost6 + cost7 + cost9 + cost10

    # Control for phase estimation and its iteration
    ac1 = 2 * np.ceil(np.log2(iters)) - 1

    # System qubits
    ac2 = n

    # First ell register that rotated and the flag for success
    ac3 = nL + 2

    # State preparation on the first register
    ac4 = nL + 2 * chi + 3

    # For p and q registers, the rotated qubit and the success flag
    ac5 = 2 * nN + 2

    # The size of the contiguous register
    ac6 = np.ceil(np.log2(nprime))

    kp = QR2(L + 1, nprime, bp)[:2]

    # The equal superposition state for the second state preparation
    ac7 = chi

    # The phase gradient state
    ac8 = br

    # The QROM on the p & q registers
    ac9 = kp[0] * kp[1] * bp + np.ceil(np.log2(
        (L + 1) / kp[0])) + np.ceil(np.log2(nprime / kp[1]))

    if verbose:
        print("[*] Top of routine")
        print("  [+] eta = ", eta)
        print("  [+] nL = ", nL)
        print("  [+] nN = ", nN)

    total_qubit_count = ac1 + ac2 + ac3 + ac4 + ac5 + ac6 + ac7 + ac8 + ac9

    # Sanity checks before returning as int
    assert cost.is_integer()
    assert iters.is_integer()
    assert total_qubit_count.is_integer()

    step_cost = int(cost)
    total_cost = int(cost * iters)
    total_qubit_count = int(total_qubit_count)

    return step_cost, total_cost, total_qubit_count
