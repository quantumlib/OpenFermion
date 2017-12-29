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

"""Module to compute Trotter errors in the plane-wave dual basis."""
from __future__ import absolute_import
from future.utils import iteritems, itervalues

import numpy

from openfermion.config import *
from openfermion.hamiltonians import fermi_hubbard
from openfermion.ops import FermionOperator, normal_ordered
from openfermion.utils import count_qubits
from openfermion.utils._dual_basis_trotter_error import (
    dual_basis_error_bound, dual_basis_error_operator, stagger_with_info)


def fermi_hubbard_error_operator(terms, indices=None, is_hopping_operator=None,
                                 verbose=False):
    """Determine the difference between the exact generator of unitary
    evolution and the approximate generator given by the second-order
    Trotter-Suzuki expansion for the Hubbard model.

    Args:
        terms: a list of FermionOperators in the Hamiltonian in the
               order in which they will be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        verbose: Whether to print percentage progress.

    Returns:
        The difference between the true and effective generators of time
            evolution for a single Trotter step.

    Notes: follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum Chemistry".
    """
    return dual_basis_error_operator(
        terms, indices=indices, is_hopping_operator=is_hopping_operator,
        verbose=verbose)


def fermi_hubbard_error_bound(terms, indices=None, is_hopping_operator=None,
                              verbose=False):
    """Numerically upper bound the error in the ground state energy
    for the second-order Trotter-Suzuki expansion.

    Args:
        terms: a list of single-term FermionOperators in the Hamiltonian
            to be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        verbose: Whether to print percentage progress.

    Returns:
        A float upper bound on norm of error in the ground state energy.

    Notes:
        Follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum
        Chemistry" to calculate the error operator.
    """
    # Return the 1-norm of the error operator (upper bound on error).
    return dual_basis_error_bound(
        terms, indices=indices, is_hopping_operator=is_hopping_operator,
        verbose=verbose)


def simulation_ordered_grouped_hubbard_terms_with_info(
        hubbard_hamiltonian):
    """Give terms from the Fermi-Hubbard Hamiltonian in simulated order.

    Uses the simulation ordering, grouping terms into hopping
    (i^ j + j^ i) and number (i^j^ i j + c_i i^ i + c_j j^ j) operators.
    Pre-computes term information (indices each operator acts on, as
    well as whether each operator is a hopping operator.

    Args:
        hubbard_hamiltonian (FermionOperator): The Hamiltonian.
        original_ordering (list): The initial Jordan-Wigner canonical order.

    Returns:
        A 3-tuple of terms from the Hubbard Hamiltonian in order of
        simulation, the indices they act on, and whether they are hopping
        operators (both also in the same order).

    Notes:
        Assumes that the Hubbard model has spin and is on a 2D square
        aperiodic lattice.
    """
    hamiltonian = normal_ordered(hubbard_hamiltonian)
    n_qubits = count_qubits(hamiltonian)
    side_length = int(numpy.sqrt(n_qubits / 2.0))

    ordered_terms = []
    ordered_indices = []
    ordered_is_hopping_operator = []

    original_ordering = list(range(n_qubits))
    for i in range(0, n_qubits - side_length, 2 * side_length):
        for j in range(2 * bool(i % (4 * side_length)), 2 * side_length, 4):
            original_ordering[i+j], original_ordering[i+j+1] = (
                original_ordering[i+j+1], original_ordering[i+j])

    input_ordering = list(original_ordering)

    # Follow odd-even transposition sort. In alternating steps, swap each even
    # qubits with the odd qubit to its right, and in the next step swap each
    # the odd qubits with the even qubit to its right. Do this until the input
    # ordering has been reversed.
    parities = [False, True] * int(side_length / 2 + 1)  # int floors here
    for parity in parities:
        results = stagger_with_info(
            hubbard_hamiltonian, input_ordering, parity)
        terms_in_step, indices_in_step, is_hopping_operator_in_step = results

        ordered_terms.extend(terms_in_step)
        ordered_indices.extend(indices_in_step)
        ordered_is_hopping_operator.extend(is_hopping_operator_in_step)

    input_ordering = list(original_ordering)

    parities = [True] + [False, True] * int(side_length / 2)  # int floors here
    for parity in parities:
        results = stagger_with_info(
            hubbard_hamiltonian, input_ordering, parity)
        terms_in_step, indices_in_step, is_hopping_operator_in_step = results

        ordered_terms.extend(terms_in_step)
        ordered_indices.extend(indices_in_step)
        ordered_is_hopping_operator.extend(is_hopping_operator_in_step)

    return (ordered_terms, ordered_indices, ordered_is_hopping_operator)
