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

"""Code for evaluating Trotter errors for diagonal Coulomb Hamiltonians."""

import numpy

from openfermion import (count_qubits,
                         FermionOperator,
                         get_fermion_operator,
                         normal_ordered)
from openfermion.utils._low_depth_trotter_error import (
    simulation_ordered_grouped_low_depth_terms_with_info)
from openfermion.utils._commutator_diagonal_coulomb_operator import (
    commutator_ordered_diagonal_coulomb_with_two_body_operator)


def diagonal_coulomb_potential_and_kinetic_terms_as_arrays(hamiltonian):
    """Give the potential and kinetic terms of a diagonal Coulomb Hamiltonian
    as arrays.

    Args:
        hamiltonian (FermionOperator): The diagonal Coulomb Hamiltonian to
                                       separate the potential and kinetic terms
                                       for. Identity is arbitrarily chosen
                                       to be part of the potential.

    Returns:
        Tuple of (potential_terms, kinetic_terms). Both elements of the tuple
        are numpy arrays of FermionOperators.
    """
    if not isinstance(hamiltonian, FermionOperator):
        try:
            hamiltonian = normal_ordered(get_fermion_operator(hamiltonian))
        except TypeError:
            raise TypeError('hamiltonian must be either a FermionOperator '
                            'or DiagonalCoulombHamiltonian.')

    potential = FermionOperator.zero()
    kinetic = FermionOperator.zero()

    for term, coeff in hamiltonian.terms.items():
        acted = set(term[i][0] for i in range(len(term)))
        if len(acted) == len(term) / 2:
            potential += FermionOperator(term, coeff)
        else:
            kinetic += FermionOperator(term, coeff)

    potential_terms = numpy.array([
        FermionOperator(term, coeff) for term, coeff in potential.terms.items()
    ])

    kinetic_terms = numpy.array(
        [FermionOperator(term, coeff) for term, coeff in kinetic.terms.items()])

    return (potential_terms, kinetic_terms)


def bit_mask_of_modes_acted_on_by_fermionic_terms(
        fermion_term_list, n_qubits=None):
    """Create a mask of which modes of the system are acted on by which terms.

    Args:
        fermion_term_list (list of FermionOperators): A list of fermionic terms
            to calculate the bitmask for.
        n_qubits (int): The number of qubits (modes) in the system. If not
                        specified, defaults to the maximum of any term in
                        fermion_term_list.

    Returns:
        An n_qubits x len(fermion_term_list) boolean numpy array of whether
        each term acts on the given mode index.

    Raises:
        ValueError: if n_qubits is too small for the given terms.
    """
    if n_qubits is None:
        n_qubits = 0
        for term in fermion_term_list:
            n_qubits = max(n_qubits, count_qubits(term))

    mask = numpy.zeros((n_qubits, len(fermion_term_list)), dtype=bool)

    for term_number, term in enumerate(fermion_term_list):
        actions = term.terms
        for action in actions:
            for single_operator in action:
                mode = single_operator[0]
                try:
                    mask[mode][term_number] = True
                except IndexError:
                    raise ValueError('Bad n_qubits: must be greater than '
                                     'highest mode in any FermionOperator.')

    return mask


def split_operator_trotter_error_operator_diagonal_two_body(hamiltonian,
                                                            order):
    """Compute the split-operator Trotter error of a diagonal two-body
    Hamiltonian.

    Args:
        hamiltonian (FermionOperator): The diagonal Coulomb Hamiltonian to
                                       compute the Trotter error for.
        order (str): Whether to simulate the split-operator Trotter step
                     with the kinetic energy T first (order='T+V') or with
                     the potential energy V first (order='V+T').

    Returns:
        error_operator: The second-order Trotter error operator.

    Notes:
        The second-order split-operator Trotter error is calculated from the
        double commutator [T, [V, T]] + [V, [V, T]] / 2 when T is simulated
        before V (i.e. exp(-iTt/2) exp(-iVt) exp(-iTt/2)), and from the
        double commutator [V, [T, V]] + [T, [T, V]] / 2 when V is simulated
        before T, following Equation 9 of "The Trotter Step Size Required for
        Accurate Quantum Simulation of Quantum Chemistry" by Poulin et al.
        The Trotter error operator is then obtained by dividing by 12.
    """
    n_qubits = count_qubits(hamiltonian)

    potential_terms, kinetic_terms = (
        diagonal_coulomb_potential_and_kinetic_terms_as_arrays(hamiltonian))

    # Cache halved potential and kinetic terms for the second commutator.
    halved_potential_terms = potential_terms / 2.0
    halved_kinetic_terms = kinetic_terms / 2.0

    # Assign the outer term of the second commutator based on the ordering.
    outer_potential_terms = (halved_potential_terms if order == 'T+V' else
                             potential_terms)
    outer_kinetic_terms = (halved_kinetic_terms if order == 'V+T' else
                           kinetic_terms)

    potential_mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
        potential_terms, n_qubits)
    kinetic_mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
        kinetic_terms, n_qubits)

    error_operator = FermionOperator.zero()

    for potential_term in potential_terms:
        modes_acted_on_by_potential_term = set()

        for potential_term_action in potential_term.terms:
            modes_acted_on_by_potential_term.update(
                set(operator[0] for operator in potential_term_action))

        if not modes_acted_on_by_potential_term:
            continue

        potential_term_mode_mask = numpy.logical_or.reduce(
            [kinetic_mask[mode] for mode in modes_acted_on_by_potential_term])

        for kinetic_term in kinetic_terms[potential_term_mode_mask]:
            inner_commutator_term = (
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    potential_term, kinetic_term))

            modes_acted_on_by_inner_commutator = set()
            for inner_commutator_action in inner_commutator_term.terms:
                modes_acted_on_by_inner_commutator.update(
                    set(operator[0] for operator in inner_commutator_action))

            if not modes_acted_on_by_inner_commutator:
                continue

            inner_commutator_mode_mask = numpy.logical_or.reduce(
                [potential_mask[mode]
                 for mode in modes_acted_on_by_inner_commutator])

            # halved_potential_terms for T+V order, potential_terms for V+T
            for outer_potential_term in outer_potential_terms[
                    inner_commutator_mode_mask]:
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    outer_potential_term, inner_commutator_term,
                    prior_terms=error_operator)

            inner_commutator_mode_mask = numpy.logical_or.reduce(
                [kinetic_mask[qubit]
                 for qubit in modes_acted_on_by_inner_commutator])

            # kinetic_terms for T+V order, halved_kinetic_terms for V+T
            for outer_kinetic_term in outer_kinetic_terms[
                    inner_commutator_mode_mask]:
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    outer_kinetic_term, inner_commutator_term,
                    prior_terms=error_operator)

    # Divide by 12 to match the error operator definition.
    # If order='V+T', also flip the sign to account for inner_commutator_term
    # not flipping between the different orderings.
    if order == 'T+V':
        error_operator /= 12.0
    else:
        error_operator /= -12.0

    return error_operator


def fermionic_swap_trotter_error_operator_diagonal_two_body(
        hamiltonian, external_potential_at_end=False):
    """Compute the fermionic swap network Trotter error of a diagonal
    two-body Hamiltonian.

    Args:
        hamiltonian (FermionOperator): The diagonal Coulomb Hamiltonian to
                                       compute the Trotter error for.

    Returns:
        error_operator: The second-order Trotter error operator.

    Notes:
        Follows Eq 9 of Poulin et al., arXiv:1406.4920, applied to the
        Trotter step detailed in Kivlichan et al., arxiv:1711.04789.
    """
    single_terms = numpy.array(
        simulation_ordered_grouped_low_depth_terms_with_info(
            hamiltonian,
            external_potential_at_end=external_potential_at_end)[0])

    # Cache the halved terms for use in the second commutator.
    halved_single_terms = single_terms / 2.0

    term_mode_mask = bit_mask_of_modes_acted_on_by_fermionic_terms(
        single_terms, count_qubits(hamiltonian))

    error_operator = FermionOperator.zero()

    for beta, term_beta in enumerate(single_terms):
        modes_acted_on_by_term_beta = set()
        for beta_action in term_beta.terms:
            modes_acted_on_by_term_beta.update(
                set(operator[0] for operator in beta_action))

        beta_mode_mask = numpy.logical_or.reduce(
            [term_mode_mask[mode] for mode in modes_acted_on_by_term_beta])

        # alpha_prime indices that could have a nonzero commutator, i.e.
        # there's overlap between the modes the corresponding terms act on.
        valid_alpha_primes = numpy.where(beta_mode_mask)[0]

        # Only alpha_prime < beta enters the error operator; filter for this.
        valid_alpha_primes = valid_alpha_primes[valid_alpha_primes < beta]

        for alpha_prime in valid_alpha_primes:
            term_alpha_prime = single_terms[alpha_prime]

            inner_commutator_term = (
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    term_beta, term_alpha_prime))

            modes_acted_on_by_inner_commutator = set()
            for inner_commutator_action in inner_commutator_term.terms:
                modes_acted_on_by_inner_commutator.update(
                    set(operator[0] for operator in inner_commutator_action))

            # If the inner commutator has no action, the commutator is zero.
            if not modes_acted_on_by_inner_commutator:
                continue

            inner_commutator_mask = numpy.logical_or.reduce(
                [term_mode_mask[mode]
                 for mode in modes_acted_on_by_inner_commutator])

            # alpha indices that could have a nonzero commutator.
            valid_alphas = numpy.where(inner_commutator_mask)[0]
            # Filter so alpha <= beta in the double commutator.
            valid_alphas = valid_alphas[valid_alphas <= beta]

            for alpha in valid_alphas:
                # If alpha = beta, only use half the term.
                if alpha != beta:
                    outer_term_alpha = single_terms[alpha]
                else:
                    outer_term_alpha = halved_single_terms[alpha]

                # Add the partial double commutator to the error operator.
                commutator_ordered_diagonal_coulomb_with_two_body_operator(
                    outer_term_alpha, inner_commutator_term,
                    prior_terms=error_operator)

    # Divide by 12 to match the error operator definition.
    error_operator /= 12.0
    return error_operator
