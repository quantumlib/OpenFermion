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

"""FermionOperator stores a sum of products of fermionic ladder operators."""
import numpy

from future.utils import iteritems
from openfermion.ops import SymbolicOperator, prune_unused_indices


class FermionOperatorError(Exception):
    pass


def normal_ordered_term(term, coefficient):
    """Return a normal ordered FermionOperator corresponding to single term.

    Args:
        term: A tuple of tuples. The first element of each tuple is
            an integer indicating the mode on which a fermion ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient: The coefficient of the term.

    Returns:
        ordered_term (FermionOperator): The normal ordered form of the input.
            Note that this might have more terms.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original term, the
        runtime of this method is exponential in the number of qubits.
    """
    # Iterate from left to right across operators and reorder to normal
    # form. Swap terms operators into correct position by moving from
    # left to right across ladder operators.
    term = list(term)
    ordered_term = FermionOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if raising on right and lowering on left.
            if right_operator[1] and not left_operator[1]:
                term[j - 1] = right_operator
                term[j] = left_operator
                coefficient *= -1

                # Replace a a^\dagger with 1 - a^\dagger a
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1)::]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_term(
                        tuple(new_term), -coefficient)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:

                # If same two operators are repeated, evaluate to zero.
                if right_operator[0] == left_operator[0]:
                    return ordered_term

                # Swap if same ladder type but lower index on left.
                elif right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator
                    coefficient *= -1

    # Add processed term and return.
    ordered_term += FermionOperator(tuple(term), coefficient)
    return ordered_term


def normal_ordered(fermion_operator):
    """Compute and return the normal ordered form of a FermionOperator.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original term, the
        runtime of this method is exponential in the number of qubits.
    """
    ordered_operator = FermionOperator()
    for term, coefficient in fermion_operator.terms.items():
        ordered_operator += normal_ordered_term(term, coefficient)
    return ordered_operator


def freeze_orbitals(fermion_operator, occupied, unoccupied=None, prune=True):
    """Fix some orbitals to be occupied and others unoccupied.

    Removes all operators acting on the specified orbitals, and renumbers the
    remaining orbitals to eliminate unused indices. The sign of each term
    is modified according to the ladder uperator anti-commutation relations in
    order to preserve the expectation value of the operator.

    Args:
        occupied: A list containing the indices of the orbitals that are to be
            assumed to be occupied.
        unoccupied: A list containing the indices of the orbitals that are to
            be assumed to be unoccupied.
    """
    new_operator = fermion_operator
    frozen = [(index, 1) for index in occupied]
    if unoccupied is not None:
        frozen += [(index, 0) for index in unoccupied]

    # Loop over each orbital to be frozen. Within each term, move all
    # ops acting on that orbital to the right side of the term, keeping
    # track of sign flips that come from swapping operators.
    for item in frozen:
        tmp_operator = FermionOperator()
        for term in new_operator.terms:
            new_term = []
            new_coef = new_operator.terms[term]
            current_occupancy = item[1]
            n_ops = 0  # Number of operations on index that have been moved
            n_swaps = 0  # Number of swaps that have been done

            for op in enumerate(reversed(term)):
                if op[1][0] is item[0]:
                    n_ops += 1

                    # Determine number of swaps needed to bring the op in
                    # front of all ops acting on other indices
                    n_swaps += op[0] - n_ops

                    # Check if the op annihilates the current state
                    if current_occupancy == op[1][1]:
                        new_coef = 0

                    # Update current state
                    current_occupancy = (current_occupancy + 1) % 2
                else:
                    new_term.insert(0, op[1])
            if n_swaps % 2 is 1:
                new_coef *= -1
            if new_coef is not 0 and current_occupancy is item[1]:
                tmp_operator += FermionOperator(tuple(new_term), new_coef)
        new_operator = tmp_operator

    # For occupied frozen orbitals, we must also bring together the creation
    # operator from the ket and the annihilation operator from the bra when
    # evaluating expectation values. This can result in an additional minus
    # sign.
    for term in new_operator.terms:
        for index in occupied:
            for op in term:
                if op[0] > index:
                    new_operator.terms[term] *= -1

    # Renumber indices to remove frozen orbitals
    new_operator = prune_unused_indices(new_operator)

    return new_operator


class FermionOperator(SymbolicOperator):
    """FermionOperator stores a sum of products of fermionic ladder operators.

    In OpenFermion, we describe fermionic ladder operators using the shorthand:
    'q^' = a^\dagger_q
    'q' = a_q
    where {'p^', 'q'} = delta_pq

    One can multiply together these fermionic ladder operators to obtain a
    fermionic term. For instance, '2^ 1' is a fermion term which
    creates at orbital 2 and destroys at orbital 1. The FermionOperator class
    also stores a coefficient for the term, e.g. '3.17 * 2^ 1'.

    The FermionOperator class is designed (in general) to store sums of these
    terms. For instance, an instance of FermionOperator might represent
    3.17 2^ 1 - 66.2 * 8^ 7 6^ 2
    The Fermion Operator class overloads operations for manipulation of
    these objects by the user.

    FermionOperator is a subclass of SymbolicOperator. Importantly, it has
    attributes set as follows::

        actions = (1, 0)
        action_strings = ('^', '')
        action_before_index = False
        different_indices_commute = False

    See the documentation of SymbolicOperator for more details.

    Example:
        .. code-block:: python

            ham = (FermionOperator('0^ 3', .5)
                   + .5 * FermionOperator('3^ 0'))
            # Equivalently
            ham2 = FermionOperator('0^ 3', 0.5)
            ham2 += FermionOperator('3^ 0', 0.5)

    Note:
        Adding FermionOperators is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a FermionOperator
        with a scalar.
    """
    actions = (1, 0)
    action_strings = ('^', '')
    action_before_index = False
    different_indices_commute = False

    def is_normal_ordered(self):
        """Return whether or not term is in normal order.

        In our convention, normal ordering implies terms are ordered
        from highest tensor factor (on left) to lowest (on right). Also,
        ladder operators come first.
        """
        for term in self.terms:
            for i in range(1, len(term)):
                for j in range(i, 0, -1):
                    right_operator = term[j]
                    left_operator = term[j - 1]
                    if right_operator[1] and not left_operator[1]:
                        return False
                    elif (right_operator[1] == left_operator[1] and
                          right_operator[0] >= left_operator[0]):
                        return False
        return True

    def is_molecular_term(self):
        """Query whether term has correct form to be from a molecular.

        Require that term is particle-number conserving (same number of
        raising and lowering operators). Require that term has 0, 2 or 4
        ladder operators. Require that term conserves spin (parity of
        raising operators equals parity of lowering operators).
        """
        for term in self.terms:
            if len(term) not in (0, 2, 4):
                return False

            # Make sure term conserves particle number and spin.
            spin = 0
            particles = 0
            for operator in term:
                particles += (-1) ** operator[1]  # add 1 if create, else -1
                spin += (-1) ** (operator[0] + operator[1])
            if not (particles == spin == 0):
                return False
        return True
