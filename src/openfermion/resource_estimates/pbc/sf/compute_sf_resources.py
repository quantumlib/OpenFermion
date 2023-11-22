# coverage: ignore
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Estimate the physical, logical, and Toffoli gate requirements for
single-factorization qubitization.  Single-Factorization uses
an LCU formed from a symmeterized Cholesky decomposition of the integrals
"""
from typing import Tuple

import numpy as np
from numpy.lib.scimath import arccos, arcsin
from sympy import factorint

from openfermion.resource_estimates.utils import QI
from openfermion.resource_estimates.utils import QR2 as QR2_of
from openfermion.resource_estimates.pbc.resources import ResourceEstimates, QR3, QR2, QI2


def compute_cost(
    num_spin_orbs: int,
    lambda_tot: float,
    num_aux: int,
    kmesh: list[int],
    dE_for_qpe: float = 0.0016,
    chi: int = 10,
) -> ResourceEstimates:
    """Determine fault-tolerant costs using single factorizated Hamiltonian.

    Light wrapper around _compute_cost to automate choice of stps paramter.

    Arguments:
        num_spin_orbs: the number of spin-orbitals
        lambda_tot: the lambda-value for the Hamiltonian
        num_sym_unique: number of symmetry unique terms kept in the sparse
            Hamiltonian
        dE_for_qpe: allowable error in phase estimation
        chi: the number of bits for the representation of the coefficients
    Returns:
        resources: sparse resources
    """
    # run once to determine stps parameter
    init_cost = _compute_cost(
        num_spin_orbs, lambda_tot, num_aux, dE_for_qpe, chi, 20_000, kmesh[0], kmesh[1], kmesh[2]
    )
    steps = init_cost[0]
    final_cost = _compute_cost(
        num_spin_orbs, lambda_tot, num_aux, dE_for_qpe, chi, steps, kmesh[0], kmesh[1], kmesh[2]
    )
    estimates = ResourceEstimates(
        toffolis_per_step=final_cost[0], total_toffolis=final_cost[1], logical_qubits=final_cost[2]
    )
    return estimates


def _compute_cost(
    n: int,
    lam: float,
    M: int,
    dE: float,
    chi: int,
    stps: int,
    Nkx: int,
    Nky: int,
    Nkz: int,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Determine fault-tolerant costs using SF decomposition in quantum chem

    Args:
        n: - the number of spin-orbitals. When using this for
                  k-point sampling, this is equivalent to N times N_k.
        lam: the lambda-value for the Hamiltonian
        M: The combined number of values of n
        dE:  allowable error in phase estimation
        chi: equivalent to aleph_1 and aleph_2 in the document, the
            number of bits for the representation of the coefficients
        stps: an approximate number of steps to choose the precision of
            single qubit rotations in preparation of the equal superposn state
        Nkx: num k-points in x-direction
        Nky: num k-points in y-direction
        Nkz: num k-points in z-direction
        verbose: do additional printing of intermediates?

    Returns:
        step_cost: Toffolis per step
        total_cost: Total number of Toffolis
        total_qubit_count: Total qubit count
    """
    nNk = (
        max(np.ceil(np.log2(Nkx)), 1)
        + max(np.ceil(np.log2(Nky)), 1)
        + max(np.ceil(np.log2(Nkz)), 1)
    )
    Nk = Nkx * Nky * Nkz
    L = Nk * n**2 // 2

    # Number of trailing zeros by computing smallest prime factor exponent
    # Number of trailing zeros
    factors = factorint(L)
    eta = factors[min(list(sorted(factors.keys())))]
    if L % 2 == 1:
        eta = 0

    # Number of qubits for the first register
    nL = np.ceil(np.log2(L))

    # Number of qubits for p and q registers
    nN = np.ceil(np.log2(n // 2))

    nMN = np.ceil(np.log2(M * Nk + 1))

    oh = [0] * 20
    for p in range(20):
        # JJG note: arccos arg may be > 1
        # v = Round[2^p/(2*\[Pi])*ArcCos[2^nL/Sqrt[(L + 1)/2^\[Eta]]/2]];
        v = np.round(
            np.power(2, p + 1) / (2 * np.pi) * arccos(np.power(2, nMN) / np.sqrt(M * Nk + 1) / 2)
        )
        #   oh[[p]] = stps*(1/N[Sin[3*ArcSin[Cos[v*2*\[Pi]/2^p]*Sqrt[(L +
        #   1)/2^\[Eta]]/2^nL]]^2] - 1) + 4*p];
        #   stps*(1/N[Sin[3*ArcSin[Cos[v*2*\[Pi]/2^p]*Sqrt[M*Nk + 1]/2^nMN]]^2]
        #   - 1) + 4*p]
        oh[p] = np.real(
            stps
            * (
                1
                / (
                    np.sin(
                        3
                        * arcsin(
                            np.cos(v * 2 * np.pi / np.power(2, p + 1))
                            * np.sqrt(M * Nk + 1)
                            / np.power(2, nMN)
                        )
                    )
                    ** 2
                )
                - 1
            )
            + 4 * (p + 1)
        )

    # Bits of precision for rotation
    br1 = int(np.argmin(oh) + 1)

    oh2 = [0] * 20
    for p in range(20):
        # JJG note: arccos arg may be > 1
        # v = Round[2^p/(2*\[Pi])*ArcCos[2^nL/Sqrt[(L + 1)/2^\[Eta]]/2]];
        v = np.round(
            np.power(2, p + 1)
            / (2 * np.pi)
            * arccos(np.power(2, nL) / np.sqrt(L / np.power(2, eta)) / 2)
        )
        #   oh[[p]] = stps*(1/N[Sin[3*ArcSin[Cos[v*2*\[Pi]/2^p]*Sqrt[(L +
        #   1)/2^\[Eta]]/2^nL]]^2] - 1) + 4*p]; oh2[[p]] =
        #   stps*(1/N[Sin[3*ArcSin[Cos[v*2*\[Pi]/2^p]*Sqrt[L/2^\[Eta]]/2^nL]]^2]
        #   - 1) + 4*p];
        oh2[p] = np.real(
            stps
            * (
                1
                / (
                    np.sin(
                        3
                        * arcsin(
                            np.cos(v * 2 * np.pi / np.power(2, p + 1))
                            * np.sqrt(L / np.power(2, eta))
                            / np.power(2, nL)
                        )
                    )
                    ** 2
                )
                - 1
            )
            + 4 * (p + 1)
        )
    br2 = int(np.argmin(oh2) + 1)

    # Cost of preparing the equal superposition state on the 1st register in 1a
    cost1a = 2 * (3 * nMN + 2 * br1 - 9)

    # We have added two qubits for ell <> 0.
    # bMN = 2*(np.ceil(np.log2(Nk)) + np.ceil(np.log2(M))) + chi + 2
    bMN = nMN + 2 * np.ceil(np.log2(Nk)) + chi + 2

    # QROM costs for first register preparation in step 1(b)
    cost1b = QR3(M * Nk + 1, bMN)[-1] + QI(M * Nk + 1)[-1]

    # The inequality test of cost chi in step 1(c) and the controlled swap with
    # cost nL + 1 in step 1(d) (and their inversions).
    cost1cd = 2 * (chi + np.ceil(np.log2((Nk)) + np.ceil(np.log2(M)) + 1))

    # Cost of preparing equal superposition of p and q registers in step 2(a).
    cost2a = 4 * (3 * nL - 3 * eta + 2 * br2 - 9)

    # The output size for the QROM for the second state preparation.
    bp = 6 * np.ceil(np.log2(Nk) / 3) + 4 * nN + chi + 3

    # Cost of QROMs for state preparation on p and q in step 2 (c).
    cost2b = QR2(M * Nk + 1, L, bp) + QI2(M * Nk + 1, L) + QR2(M * Nk, L, bp) + QI2(M * Nk, L)

    # The cost of the inequality test and controlled swap for the
    # quantum alias sampling in steps 2(c) and (d).
    cost2cd = 4 * (chi + 3 * np.ceil(np.log2(Nk) / 3) + 2 * nN + 1)

    # Computing k - Q and k' - Q, then below we perform adjustments for non -
    # modular arithmetic.
    cost3 = 4 * nNk
    if Nkx == 2 ** np.ceil(np.log2(Nkx)):
        cost3 = cost3 - 2 * np.ceil(np.log2(Nkx))
    if Nky == 2 ** np.ceil(np.log2(Nky)):
        cost3 = cost3 - 2 * np.ceil(np.log2(Nky))
    if Nkz == 2 ** np.ceil(np.log2(Nkz)):
        cost3 = cost3 - 2 * np.ceil(np.log2(Nkz))

    # The SELECT operation in step 4, which needs to be done twice.
    cost4 = 2 * (3 * n * Nk - 4)

    # The reflection used on the 2nd register in the middle, with cost in step 6
    cost6 = nL + chi + 5

    # Step 7 involves performing steps 2 to 5 again, which are accounted for
    # above, but there is one extra Toffoli for checking that l <> 0.
    cost7 = 4

    # The total reflection for the quantum walk in step 9.
    # with an extra + 1 for k - point sampling.
    cost9 = nMN + nL + 2 * chi + 4

    # The two Toffolis for the control for phase estimation and iteration,
    # given in step 10.
    cost10 = 2

    # The number of steps.
    iters = np.ceil(np.pi * lam / (dE * 2))

    # The total Toffoli costs for a step.
    cost = (
        cost1a
        + cost1b
        + cost1cd
        + cost2a
        + cost2b
        + cost2cd
        + cost3
        + cost4
        + cost6
        + cost7
        + cost9
        + cost10
    )

    # Control for phase estimation and its iteration
    ac1 = 2 * np.ceil(np.log2(iters)) - 1

    # System qubits
    ac2 = n * Nk

    # First ell register that rotated and the flag for success
    ac3 = nMN + 2

    # State preparation on the first register
    ac4 = bMN + chi + 1

    # For p and q registers, the rotated qubit and the success flag
    ac5 = nL + 2

    kp = QR2_of(M * Nk + 1, L, bp)[:2]

    ac6 = chi

    # The equal superposition state for the second state preparation
    ac7 = np.max([br1, br2])

    # The phase gradient state
    ac8 = 4

    # The QROM on the p & q registers
    ac9 = kp[0] * kp[1] * bp + np.ceil(np.log2((M * Nk + 1) / kp[0])) + np.ceil(np.log2(L / kp[1]))

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
