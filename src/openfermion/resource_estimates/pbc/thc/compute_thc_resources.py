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
# coverage:ignore
""" Determine costs for THC decomposition in QC """
from typing import Tuple, Union
import numpy as np
from numpy.lib.scimath import arccos, arcsin
from sympy import factorint
from openfermion.resource_estimates.utils import QI

from openfermion.resource_estimates.pbc.resources.data_types import ResourceEstimates
from openfermion.resource_estimates.pbc.resources.qrom import QR3


def compute_cost(
    num_spin_orbs: int,
    lambda_tot: float,
    thc_dim: int,
    kmesh: list[int],
    dE_for_qpe: float = 0.0016,
    chi: int = 10,
    beta: Union[int, None] = None,
) -> ResourceEstimates:
    """Determine fault-tolerant costs using THC factorization representation of
        symmetry adapted integrals.

    Light wrapper around _compute_cost.

    Arguments:
        num_spin_orbs: the number of spin-orbitals
        lambda_tot: the lambda-value for the Hamiltonian
        kmesh: kpoint mesh.
        thc_dim: THC dimension (M).
        dE_for_qpe: allowable error in phase estimation
        chi: the number of bits for the representation of the coefficients
        beta: the number of bits for controlled rotations
    Returns:
        resources: THC factorized resource estimates
    """
    # run once to determine stps parameter
    thc_costs = _compute_cost(
        n=num_spin_orbs,
        lam=lambda_tot,
        dE=dE_for_qpe,
        chi=chi,
        beta=beta,
        M=thc_dim,
        Nkx=kmesh[0],
        Nky=kmesh[1],
        Nkz=kmesh[2],
        stps=20_000,  # not used
    )
    resources = ResourceEstimates(
        toffolis_per_step=thc_costs[0], total_toffolis=thc_costs[1], logical_qubits=thc_costs[2]
    )
    return resources


def _compute_cost(
    n: int,
    lam: float,
    dE: float,
    chi: int,
    beta: int,
    M: int,
    Nkx: int,
    Nky: int,
    Nkz: int,
    stps: int,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Determine fault-tolerant costs using THC decomposition in quantum chem

    Arguments:
        n: the number of spin-orbitals
        lam: the lambda-value for the Hamiltonian
        dE: allowable error in phase estimation
        chi: equivalent to aleph in the document, the number of bits for
            the representation of the coefficients
        beta: equivalent to beth in the document, the number of bits for
            the rotations
        M: the dimension for the THC decomposition
        Nkx: This is the number of values of k for the k - point sampling,
                with each component
        Nky: This is the number of values of k for the k - point sampling,
                with each component
        Nkz: This is the number of values of k for the k - point sampling,
                with each component
        stps: an approximate number of steps to choose the precision of
            single qubit rotations in preparation of the equal superpositn state

    Returns:
        step_cost: Toffolis per step
        total_cost: Total number of Toffolis
        ancilla_cost: Total ancilla cost
    """
    nk = (
        max(np.ceil(np.log2(Nkx)), 1)
        + max(np.ceil(np.log2(Nky)), 1)
        + max(np.ceil(np.log2(Nkz)), 1)
    )
    Nk = Nkx * Nky * Nkz

    # (*Temporarily set as number of even numbers.*)
    nc = 3 - Nkx % 2 - Nky % 2 - Nkz % 2

    # The number of steps needed
    iters = np.ceil(np.pi * lam / (2 * dE))

    # This is the number of distinct items of data we need to output
    # see Eq. (28).*)
    d = int(32 * (Nk + 2**nc) * M**2 + n * Nk / 2)

    # The number of bits used for the contiguous register
    nc = np.ceil(np.log2(d))

    nM = np.ceil(np.log2(M))

    # The output size is 2* Log[M] for the alt values, chi for the keep value,
    # and 2 for the two sign bits.
    m = 2 * (2 * nM + nk + 8) + chi

    oh = [0] * 20
    for p in range(20):
        # arccos arg may be > 1
        v = np.round(np.power(2, p + 1) / (2 * np.pi) * arccos(np.power(2, nc) / np.sqrt(d) / 2))
        oh[p] = stps * (
            1
            / (
                np.sin(
                    3
                    * arcsin(
                        np.cos(v * 2 * np.pi / np.power(2, p + 1)) * np.sqrt(d) / np.power(2, nc)
                    )
                )
                ** 2
            )
            - 1
        ) + 4 * (p + 1)

    # Set it to be the number of bits that minimises the cost, usually 7.
    # Python is 0-index, so need to add the one back in vs mathematica nb
    br = np.argmin(oh) + 1

    if d % 2 == 0:
        factors = factorint(int(d))
        eta = factors[min(list(sorted(factors.keys())))]
    else:
        eta = 0

    cp1 = 2 * (3 * nc - 3 * eta + 2 * br - 9)

    # This is the cost of the QROM for the state preparation in step 3 and its
    cp3 = QR3(d, m)[1] + QI(d)[1]

    # The cost for the inequality test in step 4 and its inverse.
    cp4 = 2 * chi

    # the cost of inequality test and controlled swap of mu and nu registers
    cp5 = 2 * (2 * nM + nk + 7)

    # The cost of an inequality test and controlled - swap of the mu and nu
    # registers
    cp6 = 4 * nM + 12

    CPCP = cp1 + cp3 + cp4 + cp5 + cp6

    # The cost of preparing the k superposition. The 7 here is the assumed
    # number # of bits for the ancilla rotation which makes the probability of
    # failure negligible.
    cks = 4 * (Nkx + Nky + Nkz + 8 * nk + 6 * 7 - 24)

    # The cost of the arithmetic computing k - Q, and controlling swaps for
    # the one-body term
    cka = 12 * nk

    # This is the cost of swapping based on the spin register
    cs1 = 3 * n * Nk / 2

    # The cost of controlled swaps into working registers based on the k or
    # k - Q value.
    cs2 = 4 * n * (Nk - 1)

    # The QROM for the rotation angles the first time.
    cs2a = QR3(Nk * (M + n / 2), n * beta)[1] + QI(Nk * (M + n / 2))[1]

    # The QROM for the rotation angles the second time.
    cs2b = QR3(Nk * M, n * beta)[1] + QI(Nk * M)[1]

    # The cost of the rotations.
    cs3 = 16 * n * (beta - 2)

    # Cost for constructing contiguous register for outputting rotations.
    cs4 = 12 * Nk + 4 * np.ceil(np.log2(Nk * (M + n / 2))) + 4 * np.ceil(np.log2(Nk * M))

    # The cost of the controlled selection of the X vs Y.
    cs5 = 2 + 4 * (Nk - 1)

    # Cost of computing K-Q-G
    cs6 = 2 * nk

    # Cost of computing contiguous register.
    cs6 = cs6 + np.ceil(np.log2(Nkx)) * np.ceil(np.log2(Nky))
    cs6 = cs6 + np.ceil(np.log2(Nkx * Nky))
    cs6 = cs6 + np.ceil(np.log2(Nkx * Nky)) * np.ceil(np.log2(Nky))
    cs6 = cs6 + np.ceil(np.log2(Nk))
    cs6 = cs6 + np.ceil(np.log2(Nk)) * np.ceil(np.log2(Nkx))
    cs6 = cs6 + np.ceil(np.log2(Nkx * Nk))
    cs6 = cs6 + np.ceil(np.log2(Nkx * Nk)) * np.ceil(np.log2(Nky))
    cs6 = cs6 + np.ceil(np.log2(Nkx * Nky * Nk))
    cs6 = cs6 + np.ceil(np.log2(Nkx * Nky * Nk)) * np.ceil(np.log2(Nkz))
    cs6 = cs6 + np.ceil(np.log2(Nk**2))
    cs6 = cs6 + np.ceil(np.log2(Nkx**2)) * np.ceil(np.log2(M))
    cs6 = cs6 + np.ceil(np.log2(Nk**2 * M))

    # the times for is for computing and uncomputing twice
    cs6 = cs6 * 4

    # The QROM cost once we computed the contiguous register
    cs6 = cs6 + 2 * (QR3(Nk**2 * M, nk + chi)[1] + QI(Nk**2 * M)[1])

    # The remaining state preparation cost with coherent alias sampling.
    cs6 = cs6 + 4 * (nk + chi)

    # The costs of generating the symmetry in Q, with a +1 for the one-body
    # control.
    cs6 = cs6 + 4 * nk + 1

    # The total select cost.
    CS = cks + cka + cs1 + cs2 + cs2a + cs2b + cs3 + cs4 + cs5 + cs6

    # The reflection cost.
    costref = nc + 2 * nk + 3 * chi + 9

    cost = CPCP + CS + costref

    # Qubits for control for phase estimation
    ac1 = 2 * np.ceil(np.log2(iters + 1)) - 1

    # system qubits
    ac2 = n * Nk

    # various control qubits
    ac3 = nc + chi + nk + 10

    # phase gradient state
    ac4 = beta

    # T state
    ac5 = 1

    # kp = 2^QRa[d, m]
    kp = np.power(2, QR3(d, m)[0])

    # First round of QROM.
    ac12 = m * kp + np.ceil(np.log2(d / kp)) - 1

    # Temporary qubits from QROM.
    ac6 = m

    # First round of QROM.
    ac7 = m * (kp - 1) + np.ceil(np.log2(d / kp)) - 1

    # Qubit from inequality test for state preparation.
    ac8 = 1

    # Qubit from inequality test for mu, nu
    ac9 = 1

    # QROM for constructing superposition on k.
    ac10 = 2 * nk + 3 * 7

    # The contiguous register and k-Q
    ac11 = np.ceil(np.log2(Nk**2 * M) + nk)

    # output for k-state
    ac12 = nk + chi

    kn = np.power(2, QR3(Nk**2 * M, nk + chi)[0])

    ac13 = (kn - 1) * (nk + chi) + np.ceil(np.log2(Nk**2 * M)) - 1

    ac14 = chi + 1

    # The contiguous register.
    ac15 = np.ceil(np.log2(Nk * (M + n / 2)))

    kr = np.power(2, QR3(Nk * (M + n / 2), n * beta)[0])

    #
    ac16 = n * beta * kr + np.ceil(np.log2(Nk * (M + n / 2))) - 1

    # common ancilla costs
    aca = ac1 + ac2 + ac3 + ac4 + ac5 + ac6

    acc = np.max([ac13, ac14 + ac15 + ac16])

    acc = np.max([ac7, ac8 + ac9 + ac10 + ac11 + ac12 + acc])

    step_cost = int(cost)
    total_cost = int(cost * iters)
    ancilla_cost = int(aca + acc)

    # step-cost, Toffoli count, logical qubits
    return step_cost, total_cost, ancilla_cost
