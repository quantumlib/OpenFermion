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

    Optional arugments:
        hbar (float): the value of hbar in the definition of the canonical
            quadrature operators. By default, hbar=1.

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

    def __init__(self, term=None, coefficient=1., hbar=1.):
        """
        Override in-place initialization of SymbolicOperator
        to take into account optional hbar argument.
        """
        super().__init__(term, coefficient)
        self.hbar = hbar

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
