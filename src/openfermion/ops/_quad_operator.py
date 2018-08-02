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

"""QuadOperator stores a sum of products of canonical quadrature operators."""

from openfermion.ops._symbolic_operator import SymbolicOperator


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

    @property
    def actions(self):
        """The allowed actions."""
        return ('q', 'p')

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('q', 'p')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return True

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return True

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
                            right_operator[1] == 'q' and
                            left_operator[1] == 'p'):
                        return False
        return True

    def is_gaussian(self):
        """Query whether the term is quadratic or lower in the
        quadrature operators.
        """
        for term in self.terms:
            if len(term) > 2:
                return False

        return True
