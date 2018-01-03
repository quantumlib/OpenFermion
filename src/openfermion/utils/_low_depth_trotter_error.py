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
from openfermion.hamiltonians import jellium_model, wigner_seitz_length_scale
from openfermion.ops import FermionOperator, normal_ordered
from openfermion.utils import count_qubits, Grid
from openfermion.utils._commutators import (
    double_commutator,
    trivially_double_commutes_dual_basis,
    trivially_double_commutes_dual_basis_using_term_info)


def low_depth_second_order_trotter_error_operator(
        terms, indices=None, is_hopping_operator=None, jellium_only=False,
        verbose=False):
    """Determine the difference between the exact generator of unitary
    evolution and the approximate generator given by the second-order
    Trotter-Suzuki expansion.

    Args:
        terms: a list of FermionOperators in the Hamiltonian in the
               order in which they will be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        jellium_only: Whether the terms are from the jellium Hamiltonian only,
                      rather than the full dual basis Hamiltonian (i.e. whether
                      c_i = c for all number operators i^ i, or whether they
                      depend on i as is possible in the general case).
        verbose: Whether to print percentage progress.

    Returns:
        The difference between the true and effective generators of time
            evolution for a single Trotter step.

    Notes: follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum Chemistry",
        applied to the "stagger"-based Trotter step for detailed in
        Kivlichan et al., "Quantum Simulation of Electronic Structure with
        Linear Depth and Connectivity", arxiv:1711.04789.
    """
    more_info = bool(indices)
    n_terms = len(terms)

    if verbose:
        import time
        start = time.time()

    error_operator = FermionOperator.zero()
    for beta in range(n_terms):
        if verbose and beta % (n_terms // 30) == 0:
            print('%4.3f percent done in' % (
                (float(beta) / n_terms) ** 3 * 100), time.time() - start)

        for alpha in range(beta + 1):
            for alpha_prime in range(beta):
                # If we have pre-computed info on indices, use it to determine
                # trivial double commutation.
                if more_info:
                    if (not
                        trivially_double_commutes_dual_basis_using_term_info(
                            indices[alpha], indices[beta],
                            indices[alpha_prime], is_hopping_operator[alpha],
                            is_hopping_operator[beta],
                            is_hopping_operator[alpha_prime], jellium_only)):
                        # Determine the result of the double commutator.
                        double_com = double_commutator(
                            terms[alpha], terms[beta], terms[alpha_prime],
                            indices[beta], indices[alpha_prime],
                            is_hopping_operator[beta],
                            is_hopping_operator[alpha_prime])
                        if alpha == beta:
                            double_com /= 2.0

                        error_operator += double_com

                # If we don't have more info, check for trivial double
                # commutation using the terms directly.
                elif not trivially_double_commutes_dual_basis(
                        terms[alpha], terms[beta], terms[alpha_prime]):
                    double_com = double_commutator(
                        terms[alpha], terms[beta], terms[alpha_prime])

                    if alpha == beta:
                        double_com /= 2.0

                    error_operator += double_com

    error_operator /= 12.0
    return error_operator


def low_depth_second_order_trotter_error_bound(
        terms, indices=None, is_hopping_operator=None,
        jellium_only=False, verbose=False):
    """Numerically upper bound the error in the ground state energy
    for the second-order Trotter-Suzuki expansion.

    Args:
        terms: a list of single-term FermionOperators in the Hamiltonian
            to be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        jellium_only: Whether the terms are from the jellium Hamiltonian only,
                      rather than the full dual basis Hamiltonian (i.e. whether
                      c_i = c for all number operators i^ i, or whether they
                      depend on i as is possible in the general case).
        verbose: Whether to print percentage progress.

    Returns:
        A float upper bound on norm of error in the ground state energy.

    Notes:
        Follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum
        Chemistry" to calculate the error operator, for the "stagger"-based
        Trotter step for detailed in Kivlichan et al., "Quantum Simulation
        of Electronic Structure with Linear Depth and Connectivity",
        arxiv:1711.04789.
    """
    # Return the 1-norm of the error operator (upper bound on error).
    return numpy.sum(numpy.absolute(list(
        low_depth_second_order_trotter_error_operator(
            terms, indices, is_hopping_operator,
            jellium_only, verbose).terms.values())))


def simulation_ordered_grouped_low_depth_terms_with_info(
        hamiltonian, input_ordering=None):
    """Give terms from the dual basis Hamiltonian in simulated order.

    Uses the simulation ordering, grouping terms into hopping
    (i^ j + j^ i) and number (i^j^ i j + c_i i^ i + c_j j^ j) operators.
    Pre-computes term information (indices each operator acts on, as
    well as whether each operator is a hopping operator.

    Args:
        hamiltonian (FermionOperator): The Hamiltonian.
        input_ordering (list): The initial Jordan-Wigner canonical order.

    Returns:
        A 3-tuple of terms from the Hamiltonian in order of simulation,
        the indices they act on, and whether they are hopping operators
        (both also in the same order).

    Notes:
        Follows the "stagger"-based simulation order discussed in Kivlichan
        et al., "Quantum Simulation of Electronic Structure with Linear
        Depth and Connectivity", arxiv:1711.04789; as such, the only
        permitted types of terms are hopping (i^ j + j^ i) and potential
        terms which are products of at most two number operators.
    """
    n_qubits = count_qubits(hamiltonian)
    hamiltonian = normal_ordered(hamiltonian)

    ordered_terms = []
    ordered_indices = []
    ordered_is_hopping_operator = []

    # If no input mode ordering is specified, default to range(n_qubits).
    try:
        input_ordering = list(input_ordering)
    except TypeError:
        input_ordering = list(range(n_qubits))

    # Half a second-order Trotter step reverses the input ordering: this tells
    # us how much we need to include in the ordered list of terms.
    final_ordering = list(reversed(input_ordering))

    # Follow odd-even transposition sort. In alternating steps, swap each even
    # qubits with the odd qubit to its right, and in the next step swap each
    # the odd qubits with the even qubit to its right. Do this until the input
    # ordering has been reversed.
    parity = 0
    while input_ordering != final_ordering:
        results = stagger_with_info(
            hamiltonian, input_ordering, parity)
        terms_in_step, indices_in_step, is_hopping_operator_in_step = results

        ordered_terms.extend(terms_in_step)
        ordered_indices.extend(indices_in_step)
        ordered_is_hopping_operator.extend(is_hopping_operator_in_step)

        # Alternate even and odd steps of the reversal procedure.
        parity = 1 - parity

    return (ordered_terms, ordered_indices, ordered_is_hopping_operator)


def stagger_with_info(hamiltonian, input_ordering, parity):
    """Give terms simulated in a single stagger of a Trotter step.

    Groups terms into hopping (i^ j + j^ i) and number
    (i^j^ i j + c_i i^ i + c_j j^ j) operators.
    Pre-computes term information (indices each operator acts on, as
    well as whether each operator is a hopping operator).

    Args:
        hamiltonian (FermionOperator): The Hamiltonian.
        input_ordering (list): The initial Jordan-Wigner canonical order.
        parity (boolean): Whether to determine the terms from the next even
            (False = 0) or odd (True = 1) stagger.

    Returns:
        A 3-tuple of terms from the Hamiltonian that are simulated in the
        stagger, the indices they act on, and whether they are hopping
        operators (all in the same order).

    Notes:
        The "staggers" used here are the left (parity=False) and right
        (parity=True) staggers detailed in Kivlichan et al., "Quantum
        Simulation of Electronic Structure with Linear Depth and
        Connectivity", arxiv:1711.04789. As such, the Hamiltonian must be
        in the form discussed in that paper. This constrains it to have
        only hopping terms (i^ j + j^ i) and potential terms which are
        products of at most two number operators (n_i or n_i n_j).
    """
    terms_in_step = []
    indices_in_step = []
    is_hopping_operator_in_step = []

    zero = FermionOperator.zero()
    n_qubits = count_qubits(hamiltonian)

    # A single round of odd-even transposition sort.
    for i in range(parity, n_qubits - 1, 2):
        # Always keep the max on the left to avoid having to normal order.
        left = max(input_ordering[i], input_ordering[i + 1])
        right = min(input_ordering[i], input_ordering[i + 1])

        # Calculate the hopping operators in the Hamiltonian.
        left_hopping_operator = FermionOperator(
            ((left, 1), (right, 0)), hamiltonian.terms.get(
                ((left, 1), (right, 0)), 0.0))
        right_hopping_operator = FermionOperator(
            ((right, 1), (left, 0)), hamiltonian.terms.get(
                ((right, 1), (left, 0)), 0.0))

        # Calculate the two-number operator l^ r^ l r in the Hamiltonian.
        two_number_operator = FermionOperator(
            ((left, 1), (right, 1), (left, 0), (right, 0)),
            hamiltonian.terms.get(
                ((left, 1), (right, 1), (left, 0), (right, 0)), 0.0))

        # Calculate the left number operator, left^ left.
        left_number_operator = FermionOperator(
            ((left, 1), (left, 0)), hamiltonian.terms.get(
                ((left, 1), (left, 0)), 0.0))

        # Calculate the right number operator, right^ right.
        right_number_operator = FermionOperator(
            ((right, 1), (right, 0)), hamiltonian.terms.get(
                ((right, 1), (right, 0)), 0.0))

        # Divide single-number terms by n_qubits-1 to avoid over-counting.
        # Each qubit is swapped n_qubits-1 times total.
        left_number_operator /= (n_qubits - 1)
        right_number_operator /= (n_qubits - 1)

        # If the overall hopping operator isn't close to zero, append it.
        # Include the indices it acts on and that it's a hopping operator.
        if not (left_hopping_operator +
                right_hopping_operator).isclose(zero):
            terms_in_step.append(left_hopping_operator +
                                 right_hopping_operator)
            indices_in_step.append(set((left, right)))
            is_hopping_operator_in_step.append(True)

        # If the overall number operator isn't close to zero, append it.
        # Include the indices it acts on and that it's a number operator.
        if not (two_number_operator + left_number_operator +
                right_number_operator).isclose(zero):
            terms_in_step.append(two_number_operator +
                                 left_number_operator +
                                 right_number_operator)
            indices_in_step.append(set((left, right)))
            is_hopping_operator_in_step.append(False)

        # Modify the current Jordan-Wigner canonical ordering in-place.
        input_ordering[i], input_ordering[i + 1] = (input_ordering[i + 1],
                                                    input_ordering[i])

    return terms_in_step, indices_in_step, is_hopping_operator_in_step


def ordered_low_depth_terms_no_info(hamiltonian):
    """Give terms from Hamiltonian in dictionary output order.

    Args:
        hamiltonian (FermionOperator): The Hamiltonian.

    Returns:
        A list of terms from the Hamiltonian in simulated order.

    Notes:
        Assumes the Hamiltonian is in the form discussed in Kivlichan
        et al., "Quantum Simulation of Electronic Structure with Linear
        Depth and Connectivity", arxiv:1711.04789. This constrains the
        Hamiltonian to have only hopping terms (i^ j + j^ i) and potential
        terms which are products of at most two number operators (n_i or
        n_i n_j).
    """
    n_qubits = count_qubits(hamiltonian)
    hamiltonian = normal_ordered(hamiltonian)
    terms = []

    for operators, coefficient in iteritems(hamiltonian.terms):
        terms += [FermionOperator(operators, coefficient)]

    return terms


def dual_basis_jellium_hamiltonian(grid_length, dimension=3,
                                   wigner_seitz_radius=10., n_particles=None,
                                   spinless=True):
    """Return the jellium Hamiltonian with the given parameters.

    Args:
        grid_length (int): The number of spatial orbitals per dimension.
        dimension (int): The dimension of the system.
        wigner_seitz_radius (float): The radius per particle in Bohr.
        n_particles (int): The number of particles in the system.
                           Defaults to half filling if not specified.
    """
    n_qubits = grid_length ** dimension
    if not spinless:
        n_qubits *= 2

    if n_particles is None:
        # Default to half filling fraction.
        n_particles = n_qubits // 2

    if not (0 <= n_particles <= n_qubits):
        raise ValueError('n_particles must be between 0 and the number of'
                         ' spin-orbitals.')

    # Compute appropriate length scale.
    length_scale = wigner_seitz_length_scale(
        wigner_seitz_radius, n_particles, dimension)

    grid = Grid(dimension, grid_length, length_scale)
    hamiltonian = jellium_model(grid, spinless=spinless, plane_wave=False)
    hamiltonian = normal_ordered(hamiltonian)
    hamiltonian.compress()
    return hamiltonian
