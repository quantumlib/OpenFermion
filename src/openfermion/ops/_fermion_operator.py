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

from openfermion.ops._symbolic_operator import SymbolicOperator


class FermionOperator(SymbolicOperator):
    r"""FermionOperator stores a sum of products of fermionic ladder operators.

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

    @property
    def actions(self):
        """The allowed actions."""
        return (1, 0)

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('^', '')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return False

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return False

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

    def is_two_body_number_conserving(self, check_spin_symmetry=False):
        """Query whether operator has correct form to be from a molecule.

        Require that term is particle-number conserving (same number of
        raising and lowering operators). Require that term has 0, 2 or 4
        ladder operators. Require that term conserves spin (parity of
        raising operators equals parity of lowering operators).

        Args:
            check_spin_symmetry (bool): Whether to check if
                operator conserves spin.
        """
        for term in self.terms:
            if len(term) not in (0, 2, 4):
                return False

            # Make sure term conserves particle number and (optionally) spin.
            spin = 0
            particles = 0
            for operator in term:
                particles += (-1) ** operator[1]  # add 1 if create, else -1
                spin += (-1) ** (operator[0] + operator[1])
            if particles:
                return False
            elif spin and check_spin_symmetry:
                return False
        return True
