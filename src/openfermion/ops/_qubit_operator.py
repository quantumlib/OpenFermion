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
from openfermion.config import EQ_TOLERANCE
from openfermion.ops import SymbolicOperator


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


class QubitOperatorError(Exception):
    pass


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
    actions = ('X', 'Y', 'Z')
    action_strings = ('X', 'Y', 'Z')
    action_before_index = True
    different_indices_commute = True

    def __imul__(self, multiplier):
        """
        Override in-place multiply of SymbolicOperator

        Args:
          multiplier(complex float, or QubitOperator): multiplier
        """
        # Handle scalars.
        if isinstance(multiplier, (int, float, complex)):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle QubitOperator.
        elif isinstance(multiplier, QubitOperator):
            result_terms = dict()
            for left_term in self.terms:
                for right_term in multiplier.terms:
                    new_coefficient = (self.terms[left_term] *
                                       multiplier.terms[right_term])

                    # Loop through local operators and create new sorted list
                    # of representing the product local operator:
                    product_operators = []
                    left_operator_index = 0
                    right_operator_index = 0
                    n_operators_left = len(left_term)
                    n_operators_right = len(right_term)
                    while (left_operator_index < n_operators_left and
                           right_operator_index < n_operators_right):
                        (left_qubit, left_loc_op) = (
                            left_term[left_operator_index])
                        (right_qubit, right_loc_op) = (
                            right_term[right_operator_index])

                        # Multiply local operators acting on the same qubit
                        if left_qubit == right_qubit:
                            left_operator_index += 1
                            right_operator_index += 1
                            (scalar, loc_op) = _PAULI_OPERATOR_PRODUCTS[
                                (left_loc_op, right_loc_op)]

                            # Add new term.
                            if loc_op != 'I':
                                product_operators += [(left_qubit, loc_op)]
                                new_coefficient *= scalar
                            # Note if loc_op == 'I', then scalar == 1.0

                        # If left_qubit > right_qubit, add right_loc_op; else,
                        # add left_loc_op.
                        elif left_qubit > right_qubit:
                            product_operators += [(right_qubit, right_loc_op)]
                            right_operator_index += 1
                        else:
                            product_operators += [(left_qubit, left_loc_op)]
                            left_operator_index += 1

                    # Finish the remaining operators:
                    if left_operator_index == n_operators_left:
                        product_operators += right_term[
                            right_operator_index::]
                    elif right_operator_index == n_operators_right:
                        product_operators += left_term[left_operator_index::]

                    # Add to result dict
                    tmp_key = tuple(product_operators)
                    if tmp_key in result_terms:
                        result_terms[tmp_key] += new_coefficient
                    else:
                        result_terms[tmp_key] = new_coefficient
            self.terms = result_terms
            return self
        else:
            raise TypeError('Cannot in-place multiply term of invalid type ' +
                            'to QubitTerm.')
    
    def renormalize(self):
        """Fix the trace norm of an operator to 1"""
        norm = self.induced_norm(2)
        if norm < EQ_TOLERANCE:
            raise ZeroDivisionError('Cannot renormalize empty or zero operator')
        else:
            self /= norm