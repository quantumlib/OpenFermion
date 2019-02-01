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

"""QubitOperator stores a sum of Pauli operators acting on qubits."""

import numpy

from openfermion.ops._symbolic_operator import SymbolicOperator


# Define products of all Pauli operators for symbolic multiplication.
_PAULI_OPERATOR_PRODUCTS = {('I', 'I'): (1., 'I'),
                            ('I', 'X'): (1., 'X'),
                            ('X', 'I'): (1., 'X'),
                            ('I', 'Y'): (1., 'Y'),
                            ('Y', 'I'): (1., 'Y'),
                            ('I', 'Z'): (1., 'Z'),
                            ('Z', 'I'): (1., 'Z'),
                            ('X', 'X'): (1., 'I'),
                            ('Y', 'Y'): (1., 'I'),
                            ('Z', 'Z'): (1., 'I'),
                            ('X', 'Y'): (1.j, 'Z'),
                            ('X', 'Z'): (-1.j, 'Y'),
                            ('Y', 'X'): (-1.j, 'Z'),
                            ('Y', 'Z'): (1.j, 'X'),
                            ('Z', 'X'): (1.j, 'Y'),
                            ('Z', 'Y'): (-1.j, 'X')}


class QubitOperator(SymbolicOperator):
    """
    A sum of terms acting on qubits, e.g., 0.5 * 'X0 X5' + 0.3 * 'Z1 Z2'.

    A term is an operator acting on n qubits and can be represented as:

    coefficient * local_operator[0] x ... x local_operator[n-1]

    where x is the tensor product. A local operator is a Pauli operator
    ('I', 'X', 'Y', or 'Z') which acts on one qubit. In math notation a term
    is, for example, 0.5 * 'X0 X5', which means that a Pauli X operator acts
    on qubit 0 and 5, while the identity operator acts on all other qubits.

    A QubitOperator represents a sum of terms acting on qubits and overloads
    operations for easy manipulation of these objects by the user.

    Note for a QubitOperator to be a Hamiltonian which is a hermitian
    operator, the coefficients of all terms must be real.

    .. code-block:: python

        hamiltonian = 0.5 * QubitOperator('X0 X5') + 0.3 * QubitOperator('Z0')

    QubitOperator is a subclass of SymbolicOperator. Importantly, it has
    attributes set as follows::

        actions = ('X', 'Y', 'Z')
        action_strings = ('X', 'Y', 'Z')
        action_before_index = True
        different_indices_commute = True

    See the documentation of SymbolicOperator for more details.

    Example:
        .. code-block:: python

            ham = ((QubitOperator('X0 Y3', 0.5)
                    + 0.6 * QubitOperator('X0 Y3')))
            # Equivalently
            ham2 = QubitOperator('X0 Y3', 0.5)
            ham2 += 0.6 * QubitOperator('X0 Y3')

    Note:
        Adding QubitOperators is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a QubitOperator
        with a scalar.
    """

    @property
    def actions(self):
        """The allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return True

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return True

    def renormalize(self):
        """Fix the trace norm of an operator to 1"""
        norm = self.induced_norm(2)
        if numpy.isclose(norm, 0.0):
            raise ZeroDivisionError(
                'Cannot renormalize empty or zero operator')
        else:
            self /= norm

    def _simplify(self, term, coefficient=1.0):
        """Simplify a term using commutator and anti-commutator relations."""
        if not term:
            return coefficient, term

        term = sorted(term, key=lambda factor: factor[0])

        new_term = []
        left_factor = term[0]
        for right_factor in term[1:]:
            left_index, left_action = left_factor
            right_index, right_action = right_factor

            # Still on the same qubit, keep simplifying.
            if left_index == right_index:
                new_coefficient, new_action = _PAULI_OPERATOR_PRODUCTS[
                        left_action, right_action]
                left_factor = (left_index, new_action)
                coefficient *= new_coefficient

            # Reached different qubit, save result and re-initialize.
            else:
                if left_action != 'I':
                    new_term.append(left_factor)
                left_factor = right_factor

        # Save result of final iteration.
        if left_factor[1] != 'I':
            new_term.append(left_factor)

        return coefficient, tuple(new_term)
