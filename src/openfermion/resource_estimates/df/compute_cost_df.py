#coverage:ignore
""" Determine costs for DF decomposition in QC """
from typing import Tuple
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # has analytc continuation to cplx
from openfermion.resource_estimates.utils import QR, QI, power_two


def compute_cost(n: int,
                 lam: float,
                 dE: float,
                 L: int,
                 Lxi: int,
                 chi: int,
                 beta: int,
                 stps: int,
                 verbose: bool = False) -> Tuple[int, int, int]:
    """ Determine fault-tolerant costs using DF decomposition in quantum chem

    Args:
        n (int) - the number of spin-orbitals
        lam (float) - the lambda-value for the Hamiltonian
        dE (float) - allowable error in phase estimation
        L (int) - the rank of the first decomposition
        Lxi (int) - the total number of eigenvectors
        chi (int) - equivalent to aleph_1 and aleph_2 in the document, the
            number of bits for the representation of the coefficients
        beta (int) - equivalent to beth in the document, the number of bits
            for the rotations
        stps (int) - an approximate number of steps to choose the precision of
            single qubit rotations in preparation of the equal superpositn state
        verbose (bool) - do additional printing of intermediates?

    Returns:
        step_cost (int) - Toffolis per step
        total_cost (int) - Total number of Toffolis
        ancilla_cost (int) - Total ancilla cost
    """

    # The number of bits used for the second register.
    nxi = np.ceil(np.log2(n // 2))

    # The number of bits for the contiguous register.
    nLxi = np.ceil(np.log2(Lxi + n // 2))

    # The number of bits used for the first register.
    nL = np.ceil(np.log2(L + 1))

    # The power of 2 that is a factor of L + 1
    eta = power_two(L + 1)

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

    # The following costs are from the list starting on page 50.

    # The cost for preparing an equal superposition for preparing the first
    # register in step 1 (a). We double this cost to account for the inverse.
    cost1a = 2 * (3 * nL + 2 * br - 3 * eta - 9)

    # The output size for the QROM for the first state preparation in Eq. (C27)
    bp1 = nL + chi

    # The cost of the QROM for the first state preparation in step 1 (b) and
    # its inverse.
    cost1b = QR(L + 1, bp1)[1] + QI(L + 1)[1]

    # The cost for the inequality test, controlled swap and their inverse in
    # steps 1 (c) and (d)
    cost1cd = 2 * (chi + nL)

    # The total cost for preparing the first register in step 1.
    cost1 = cost1a + cost1b + cost1cd

    # The output size for the QROM for the data to prepare the equal
    # superposition on the second register, as given in Eq. (C29).
    bo = nxi + nLxi + br + 1

    # This is step 2. This is the cost of outputting the data to prepare the
    # equal superposition on the second register. We will assume it is not
    # uncomputed, because we want to keep the offset for applying the QROM for
    # outputting the rotations.
    cost2 = QR(L + 1, bo)[1] + QI(L + 1)[1]

    # The number of bits for rotating the ancilla for the second preparation.
    # We are just entering this manually because it is a typical value.
    br = 7

    # The cost of preparing an equal superposition over the second register in
    # a controlled way. We pay this cost 4 times.
    cost3a = 4 * (7 * nxi + 2 * br - 6)

    # The cost of the offset to apply the QROM for state preparation on the
    # second register.
    cost3b = 4 * (nLxi - 1)

    bp2 = nxi + chi + 2

    # The cost of the QROMs and inverse QROMs for the state preparation, where
    # in the first one we need + n/2 to account for the one-electron terms.
    cost3c = QR(Lxi + n // 2, bp2)[1] + QI(Lxi + n // 2)[1] + QR(
        Lxi, bp2)[1] + QI(Lxi)[1]

    # The inequality test and state preparations.
    cost3d = 4 * (nxi + chi)

    # The total costs for state preparations on register 2.
    cost3 = cost3a + cost3b + cost3c + cost3d

    # The cost of adding offsets in steps 4 (a) and (h).
    cost4ah = 4 * (nLxi - 1)

    # The costs of the QROMs and their inverses in steps 4 (b) and (g).
    cost4bg = QR(Lxi + n // 2, n * beta // 2)[1] + QI(Lxi + n // 2)[1] + QR(
        Lxi, n * beta // 2)[1] + QI(Lxi)[1]

    # The cost of the controlled swaps based on the spin qubit in steps 4c and f
    cost4cf = 2 * n

    # The controlled rotations in steps 4 (d) and (f).
    cost4df = 4 * n * (beta - 2)

    # The controlled Z operations in the middle for step 4 (e).
    cost4e = 3

    # This is the cost of the controlled rotations for step 4.
    cost4 = cost4ah + cost4bg + cost4cf + cost4df + cost4e

    # This is the cost of the reflection on the second register from step 6.
    cost6 = nxi + chi + 2

    # The cost of the final reflection req'd to construct the step of the
    # quantum walk from step 9.
    cost9 = nL + nxi + chi + 1

    # The extra two qubits for unary iteration and making the rflxn controlled.
    cost10 = 2

    # The Toffoli cost for a single step
    cost = cost1 + cost2 + cost3 + cost4 + cost6 + cost9 + cost10

    # The number of steps needed
    iters = np.ceil(np.pi * lam / (2 * dE))

    # Now the number of qubits from the list on page 54.

    k1 = np.power(2, QR(Lxi + n // 2, n * beta // 2)[0])

    # The control register for phase estimation and iteration on it.
    ac1 = np.ceil(np.log2(iters + 1)) * 2 - 1

    # The system qubits
    ac2 = n

    # The first register prepared, a rotated qubit and a flag qubit.
    ac3 = nL + 2

    # The output of the QROM, the equal superposition state and a flag qubit.
    ac4 = nL + chi * 2 + 1

    # The data used for preparing the equal superposition state on the 2nd reg
    ac5 = bo

    # The second register, a rotated qubit and a flag qubit.
    ac6 = nxi + 2

    # The second preparation QROM output.
    ac8 = bp2

    # The equal superposition state and the result of the inequality test.
    ac9 = chi + 1

    # The angles for rotations.
    ac10 = k1 * n * beta // 2

    # The phase gradient state.
    ac11 = beta

    # A control qubit for the spin.
    ac12 = 1

    # A T state.
    ac13 = 1

    if verbose:
        print("[*] Top of routine")
        print("  [+] nxi = ", nxi)
        print("  [+] nLxi = ", nLxi)
        print("  [+] nL = ", nL)
        print("  [+] eta = ", eta)
        print("  [+] cost3 = ", cost3)
        print("  [+] cost4 = ", cost4)
        print("  [+] cost = ", cost)
        print("  [+] iters = ", iters)

    ancilla_cost = ac1 + ac2 + ac3 + ac4 + ac5 + ac6 + ac8 + ac9 + ac10 + ac11\
                 + ac12 + ac13

    # Sanity checks before returning as int
    assert cost.is_integer()
    assert iters.is_integer()
    assert ancilla_cost.is_integer()

    step_cost = int(cost)
    total_cost = int(cost * iters)
    ancilla_cost = int(ancilla_cost)

    return step_cost, total_cost, ancilla_cost
