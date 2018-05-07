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

"""Quadrature operator stores a sum of products of canonical quadrature operators."""
import numpy

from future.utils import iteritems
from openfermion.ops import SymbolicOperator


class QuadOperatorError(Exception):
    pass


def normal_ordered_term(term, coefficient, hbar=1.):
    """Return a normal ordered BosonOperator corresponding to single term.

    Args:
        term: A tuple of tuples. The first element of each tuple is
            an integer indicating the mode on which a boson ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient: The coefficient of the term.
        hbar (float): the value of hbar used in the definition of the commutator
            [q_i, p_j] = i hbar delta_ij. By default hbar=1.

    Returns:
        ordered_term (QuadOperator): The normal ordered form of the input.
            Note that this might have more terms.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, q operators come first.
    """
    # Iterate from left to right across operators and reorder to normal
    # form. Swap terms operators into correct position by moving from
    # left to right across ladder operators.
    term = list(term)
    ordered_term = QuadOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if q on right and p on left.
            # p q -> q p
            if right_operator[1] == 'q' and not left_operator[1] == 'q':
                term[j - 1] = right_operator
                term[j] = left_operator

                # Replace p q with i hbar + q p
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1)::]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_term(
                        tuple(new_term), -coefficient*1j*hbar)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:

                # Swap if same type but lower index on left.
                if right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator

    # Add processed term and return.
    ordered_term += QuadOperator(tuple(term), coefficient)
    return ordered_term


def normal_ordered(quad_operator, hbar=1.):
    """Compute and return the normal ordered form of a QuadOperator.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, q operators come first.

    Args:
        quad_operator (QuadOperator): the quadrature operator.
        hbar (float): the value of hbar used in the definition of the commutator
            [q_i, p_j] = i hbar delta_ij. By default hbar=1.
    """
    ordered_operator = QuadOperator()
    for term, coefficient in quad_operator.terms.items():
        ordered_operator += normal_ordered_term(term, coefficient, hbar=hbar)
    return ordered_operator


class QuadOperator(SymbolicOperator):
    """QuadOperator stores a sum of products of canonical quadrature operators.

    They are defined in terms of the bosonic ladder operators:
    q = sqrt{hbar/2}(b+b^)
    p = -isqrt{hbar/2}(b-b^)
    where hbar is a constant appearing in the commutator of q and p:
    [q, p] = i hbar

    In OpenFermion, we describe the canonical quadrature operators acting
    on quantum modes 'i' and 'j' using the shorthand:
    'qi' = q_i
    'pj' = p_j
    where ['qi', 'pj'] = i hbar delta_ij is the commutator.

    The QuadOperator class is designed (in general) to store sums of these
    terms. For instance, an instance of QuadOperator might represent

    .. code-block:: python

        H = 0.5 * QuadOperator('q0 p5') + 0.3 * QuadOperator('q0')

    Note for a QuadOperator to be a Hamiltonian which is a hermitian
    operator, the coefficients of all terms must be real.

    QuadOperator is a subclass of SymbolicOperator. Importantly, it has
    attributes set as follows::

        actions = ('q', 'p')
        action_strings = ('q', 'p')
        action_before_index = True
        different_indices_commute = True

    See the documentation of SymbolicOperator for more details.

    Example:
        .. code-block:: python

            H = (QuadOperator('p0 q3', 0.5)
                   + 0.6 * QuadOperator('p3 q0'))
            # Equivalently
            H2 = QuadOperator('p0 q3', 0.5)
            H2 += QuadOperator('p3 q0', 0.6)

    Note:
        Adding QuadOperator is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a QuadOperator
        with a scalar.
    """
    actions = ('q', 'p')
    action_strings = ('q', 'p')
    action_before_index = True
    different_indices_commute = True

    def is_normal_ordered(self):
        """Return whether or not term is in normal order.

        In our convention, q operators come first.
        Note that unlike the Fermion operator, due to the commutation
        of quadrature operators with different indices, the QuadOperator
        sorts quadrature operators by index.
        """
        for term in self.terms:
            for i in range(1, len(term)):
                for j in range(i, 0, -1):
                    right_operator = term[j]
                    left_operator = term[j - 1]
                    if (right_operator[0] == left_operator[0] and
                          right_operator[1] == 'q' and left_operator[1] == 'p'):
                        return False
        return True

    def is_linear(self):
        """Query whether the term is quadratic or lower in the
        quadrature operators.
        """
        for term in self.terms:
            q_count = dict()
            p_count = dict()
            for operator in term:
                if operator[1] == 'q':
                    q_count[operator[0]] = q_count.get(operator[0], 0) + 1
                elif operator[1] == 'p':
                    p_count[operator[0]] = p_count.get(operator[0], 0) + 1

            if (any(v > 2 for v in q_count.values()) or
              any(v > 2 for v in p_count.values())):
                return False
        return True
