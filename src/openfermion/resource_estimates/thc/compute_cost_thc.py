# coverage:ignore
""" Determine costs for THC decomposition in QC """
from typing import Tuple
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # has analytc continuatn to cplx
from openfermion.resource_estimates.utils import QR, QI


def compute_cost(
    n: int, lam: float, dE: float, chi: int, beta: int, M: int, stps: int, verbose: bool = False
) -> Tuple[int, int, int]:
    """Determine fault-tolerant costs using THC decomposition in quantum chem

    Args:
        n (int) - the number of spin-orbitals
        lam (float) - the lambda-value for the Hamiltonian
        dE (float) - allowable error in phase estimation
        chi (int) - equivalent to aleph in the document, the number of bits for
            the representation of the coefficients
        beta (int) - equivalent to beth in the document, the number of bits for
            the rotations
        M (int) - the dimension for the THC decomposition
        stps (int) - an approximate number of steps to choose the precision of
            single qubit rotations in preparation of the equal superpositn state

    Returns:
        step_cost (int) - Toffolis per step
        total_cost (int) - Total number of Toffolis
        ancilla_cost (int) - Total ancilla cost
    """
    # The number of bits used for each register.
    nM = np.ceil(np.log2(M + 1))

    # This is the number of distinct items of data we need to output.
    d = M * (M + 1) // 2 + n // 2

    # The number of bits used for the contiguous register.
    nc = np.ceil(np.log2(d))

    # The output size is 2*log[M] for the alt values, chi for the keep value,
    # and 2 for the two sign bits, as in Eq. (29).
    m = 2 * nM + 2 + chi
    oh = [0] * 20
    for p in range(20):
        # arccos arg may be > 1
        v = np.round(np.power(2, p + 1) / (2 * np.pi) * arccos(np.power(2, nM) / np.sqrt(d) / 2))
        oh[p] = stps * (
            1
            / (
                np.sin(
                    3
                    * arcsin(
                        np.cos(v * 2 * np.pi / np.power(2, p + 1)) * np.sqrt(d) / np.power(2, nM)
                    )
                )
                ** 2
            )
            - 1
        ) + 4 * (p + 1)

    # Set it to be the number of bits that minimises the cost, usually 7.
    # Python is 0-index, so need to add the one back in vs mathematica nb
    br = np.argmin(oh) + 1

    # This is the costing for preparing the equal superposition over the
    # input registers from below Eq. (27).
    cp1 = 2 * (10 * nM + 2 * br - 9)

    # This is the cost of computing the contiguous register and inverting it.
    # This is with a sophisticated scheme adding together triplets of bits.
    # This is the cost of step 2 in the list on pages 15 and 16, with a factor
    # of 2 to account for its inverse.
    cp2 = 2 * (nM**2 + nM - 1)

    # This is the cost of the QROM for the state preparation in step 3 and its
    # inverse. Note: arg_min is first value, min is second value
    cp3 = QR(d, m)[1] + QI(d)[1]

    # The cost for the inequality test in step 4 and its inverse.
    cp4 = 2 * chi

    # The cost 2*nM for the controlled swap in step 5 and its inverse.
    cp5 = 4 * nM

    # Then there is a cost of nM + 1 for swapping the mu and nu registers in
    # step 6, where the + 1 is because we need to control on two registers.
    # There is the same cost for the inverse.
    cp6 = 2 * nM + 2

    # This is the total cost in Eq. (32).
    CPCP = cp1 + cp2 + cp3 + cp4 + cp5 + cp6

    # This is the cost of swapping based on the spin register. It is steps 1
    # and 7 from the list of steps on pages 16 and 17.
    cs1 = 2 * n

    # The QROM for the rotation angles the first time in step 2. and the second
    # part is the cost for inverting them with advanced QROM (step 6).
    cs2a = M + n / 2 - 2

    # The QROM for the rotation angles the second time. Here the cost M - 2 is
    # for generating the angles the second time (again step 2), and QI[M] is
    # for inverting the QROM (step 6).
    cs2b = M - 2

    # The cost of the rotations in steps 3 and 5, which must be done 4 times.
    cs3 = 4 * n * (beta - 2)

    # Cost for making the Z doubly controlled in step 4, and a controlled swap
    #  of the spin qubits. Note: is this really just "2"? Does this need a var?
    cs4 = 2

    # want the argument, so that is first element
    k1 = np.power(2, QI(M + n // 2)[0])

    # The cost for inverting the rotation angles the first time in step 6.
    cs6a = np.ceil(M / k1) + np.ceil(n / 2 / k1) + k1

    # The cost for inverting the rotation angles the second time in step 6.
    # value is in second output argument of QI
    cs6b = QI(M)[1]

    # The total select cost in Eq. (42).
    CS = cs1 + cs2a + cs2b + cs3 + cs4 + cs6a + cs6b

    # The cost of the reflections used in Eq. (43).
    costref = 2 * nM + chi + 4

    # The total cost in Eq. (43).
    cost = CPCP + CS + costref

    # The cost of the control register and temporary registers given for part 1.
    ac1 = lambda iters: 2 * np.ceil(np.log2(iters + 1)) - 1

    # Qubits for the system register in part 2.
    ac2 = n

    # The cost for the \[Mu] and \[Nu] registers in part 3.
    ac3 = 2 * nM

    # This is for the equal superposition state to perform the inequal
    # test with the keep register.
    ac4 = chi

    # Some minor qubit counts in parts 5 to 9.
    ac59 = 7

    # The phase gradient state in part 10.
    ac10 = beta

    # The contiguous register in part 11
    ac11 = nc

    # we want arg_min so first argument
    kt = np.power(2, QR(d, m))[0]

    # The qubits used for the QROM in part 12, of which only m are persistent.
    ac12 = m * kt + np.ceil(np.log2(d / kt))

    # This is the data needed for the rotations in part 13.
    ac13 = beta * n / 2

    # These are the ancillas needed for adding into the phase gradient state
    # in part 14.
    ac14 = beta - 2

    # These are the temporary ancillas in between erasing the first QROM
    # ancillas and inverting that QROM. The + m is for output of the first QROM.
    acc = ac13 + ac14 + m

    # Total number of iterations
    iters = np.ceil(np.pi * lam / (dE * 2))
    aca = ac1(iters) + ac2 + ac3 + ac4 + ac59 + ac10 + ac11

    if verbose:
        print("[*] Top of routine")
        print("  [+] nM = ", nM)
        print("  [+] d = ", d)
        print("  [+] nc = ", nc)
        print("  [+] m = ", m)
        print("  [+] oh = ", oh)
        print("  [+] br = ", br)
        print("  [+] cp1 = ", cp1)
        print("  [+] cp2 = ", cp2)
        print("    [*] QR[d,m] = ", QR(d, m))
        print("    [*] QI[d] = ", QI(d))
        print("  [+] cp3 = ", cp3)
        print("  [+] cp4 = ", cp4)
        print("  [+] cp5 = ", cp5)
        print("  [+] cp6 = ", cp6)
        print("  [+] CPCP = ", CPCP)
        print("  [+] cs1 = ", cs1)
        print("  [+] cs2a = ", cs2a)
        print("  [+] cs2b = ", cs2b)
        print("  [+] cs3 = ", cs3)
        print("  [+] k1 = ", k1)
        print("  [+] cs6a = ", cs6a)
        print("  [+] cs6b = ", cs6b)
        print("  [+] CS = ", CS)
        print("  [+] costref = ", costref)
        print("  [+] cost = ", cost)
        print("  [+] ac1 = ", ac1)
        print("  [+] ac2 = ", ac2)
        print("  [+] ac3 = ", ac3)
        print("  [+] ac4 = ", ac4)
        print("  [+] ac10 = ", ac10)
        print("  [+] ac11 = ", ac11)
        print("  [+] kt = ", kt)
        print("  [+] ac12 = ", ac12)
        print("  [+] ac13 = ", ac13)
        print("  [+] ac14 = ", ac14)
        print("  [+] acc = ", acc)
        print("  [+] iters = ", iters)
        print("  [+] aca = ", aca)

    # Sanity checks before returning as int
    # assert cost.is_integer()
    # assert iters.is_integer()
    # assert aca.is_integer()
    # assert ac12.is_integer()
    # assert acc.is_integer()

    step_cost = int(cost)
    total_cost = int(cost * iters)
    ancilla_cost = int(np.max([aca + ac12, aca + acc]))

    # step-cost, Toffoli count, logical qubits
    return step_cost, total_cost, ancilla_cost
