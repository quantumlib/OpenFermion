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

"""BosonOperator stores a sum of products of bosonic ladder operators."""
import numpy

from future.utils import iteritems
from openfermion.ops import SymbolicOperator


class BosonOperatorError(Exception):
    pass


def normal_ordered_term(term, coefficient):
    """Return a normal ordered BosonOperator corresponding to single term.

    Args:
        term: A tuple of tuples. The first element of each tuple is
            an integer indicating the mode on which a boson ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient: The coefficient of the term.

    Returns:
        ordered_term (BosonOperator): The normal ordered form of the input.
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
    ordered_term = BosonOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if raising on right and lowering on left.
            # b b^\dagger -> b^\dagger b
            if right_operator[1] and not left_operator[1]:
                term[j - 1] = right_operator
                term[j] = left_operator

                # Replace b b^\dagger with 1 + b^\dagger b
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1)::]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_term(
                        tuple(new_term), coefficient)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:
                # Swap if same ladder type but lower index on left.
                if right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator

    # Add processed term and return.
    ordered_term += BosonOperator(tuple(term), coefficient)
    return ordered_term


def normal_ordered(boson_operator):
    """Compute and return the normal ordered form of a BosonOperator.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original term, the
        runtime of this method is exponential in the number of qubits.
    """
    ordered_operator = BosonOperator()
    for term, coefficient in boson_operator.terms.items():
        ordered_operator += normal_ordered_term(term, coefficient)
    return ordered_operator


class BosonOperator(SymbolicOperator):
    """BosonOperator stores a sum of products of bosonic ladder operators.

    In OpenFermion, we describe bosonic ladder operators using the shorthand:
    'i^' = b^\dagger_i
    'j' = b_j
    where ['i', 'j^'] = delta_ij is the commutator.

    One can multiply together these bosonic ladder operators to obtain a
    bosonic term. For instance, '2^ 1' is a bosonic term which
    creates at mode 2 and destroys at mode 1. The BosonicOperator class
    also stores a coefficient for the term, e.g. '3.17 * 2^ 1'.

    The BosonOperator class is designed (in general) to store sums of these
    terms. For instance, an instance of BosonOperator might represent
    3.17 2^ 1 - 66.2 * 8^ 7 6^ 2
    The Bosonic Operator class overloads operations for manipulation of
    these objects by the user.

    BosonOperator is a subclass of SymbolicOperator. Importantly, it has
    attributes set as follows::

        actions = (1, 0)
        action_strings = ('^', '')
        action_before_index = False
        different_indices_commute = True

    See the documentation of SymbolicOperator for more details.

    Example:
        .. code-block:: python

            H = (BosonOperator('0^ 3', .5)
                   + .5 * BosonOperator('3^ 0'))
            # Equivalently
            H2 = BosonOperator('0^ 3', 0.5)
            H2 += BosonOperator('3^ 0', 0.5)

    Note:
        Adding BosonOperator is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a BosonOperator
        with a scalar.
    """
    actions = (1, 0)
    action_strings = ('^', '')
    action_before_index = False
    different_indices_commute = True

    def is_normal_ordered(self):
        """Return whether or not term is in normal order.

        In our convention, ladder operators come first.
        Note that unlike the Fermion operator, due to the commutation
        of ladder operators with different indices, the BosonOperator
        sorts ladder operators by index.
        """
        for term in self.terms:
            for i in range(1, len(term)):
                for j in range(i, 0, -1):
                    right_operator = term[j]
                    left_operator = term[j - 1]
                    if (right_operator[0] == left_operator[0] and
                          right_operator[1] > left_operator[1]):
                        return False
        return True

    def is_boson_preserving(self):
        """Query whether the term preserves particle number.

        This is equivalent to requiring the same number of
        raising and lowering operators in each term.
        """
        for term in self.terms:
            # Make sure term conserves particle number
            particles = 0
            for operator in term:
                particles += (-1) ** operator[1]  # add 1 if create, else -1
            if not (particles == 0):
                return False
        return True