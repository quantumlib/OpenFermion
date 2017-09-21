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
import copy
import itertools

import numpy


EQ_TOLERANCE = 1e-12


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


class QubitOperator(object):
    """
    A sum of terms acting on qubits, e.g., 0.5 * 'X0 X5' + 0.3 * 'Z1 Z2'.

    A term is an operator acting on n qubits and can be represented as:

    coefficent * local_operator[0] x ... x local_operator[n-1]

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

    Attributes:
        terms (dict): **key**: A term represented by a tuple containing all
                      non-trivial local Pauli operators ('X', 'Y', or 'Z').
                      A non-trivial local Pauli operator is specified by a
                      tuple with the first element being an integer
                      indicating the qubit on which a non-trivial local
                      operator acts and the second element being a string,
                      either 'X', 'Y', or 'Z', indicating which non-trivial
                      Pauli operator acts on that qubit. Examples:
                      ((1, 'X'),) or ((1, 'X'), (4,'Z')) or the identity ().
                      The tuples representing the non-trivial local terms
                      are sorted according to the qubit number they act on,
                      starting from 0.
                      **value**: Coefficient of this term as a (complex) float
    """

    def __init__(self, term=None, coefficient=1.):
        """
        Inits a QubitOperator.

        The init function only allows to initialize one term. Additional terms
        have to be added using += (which is fast) or using + of two
        QubitOperator objects:

        Example:
            .. code-block:: python

                ham = ((QubitOperator('X0 Y3', 0.5)
                        + 0.6 * QubitOperator('X0 Y3')))
                # Equivalently
                ham2 = QubitOperator('X0 Y3', 0.5)
                ham2 += 0.6 * QubitOperator('X0 Y3')

        Note:
            Adding terms to QubitOperator is faster using += (as this is done
            by in-place addition). Specifying the coefficient in the __init__
            is faster than by multiplying a QubitOperator with a scalar as
            calls an out-of-place multiplication.

        Args:
            coefficient (complex float, optional): The coefficient of the
                first term of this QubitOperator. Default is 1.0.
            term (optional, empy tuple, a tuple of tuples, or a string):
                1) Default is None which means there are no terms in the
                   QubitOperator hence it is the "zero" Operator
                2) An empty tuple means there are no non-trivial Pauli
                   operators acting on the qubits hence only identities
                   with a coefficient (which by default is 1.0).
                3) A sorted tuple of tuples. The first element of each tuple
                   is an integer indicating the qubit on which a non-trivial
                   local operator acts, starting from zero. The second element
                   of each tuple is a string, either 'X', 'Y' or 'Z',
                   indicating which local operator acts on that qubit.
                4) A string of the form 'X0 Z2 Y5', indicating an X on
                   qubit 0, Z on qubit 2, and Y on qubit 5. The string should
                   be sorted by the qubit number. '' is the identity.

        Raises:
          QubitOperatorError: Invalid operators provided to QubitOperator.
        """
        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be a numeric type.')
        self.terms = {}
        if term is None:
            return
        elif isinstance(term, tuple):
            if term is ():
                self.terms[()] = coefficient
            else:
                # Test that input is a tuple of tuples and correct action
                for local_operator in term:
                    if (not isinstance(local_operator, tuple) or
                            len(local_operator) != 2):
                        raise ValueError("term specified incorrectly.")
                    qubit_num, action = local_operator
                    if not isinstance(action, str) or action not in 'XYZ':
                        raise ValueError("Invalid action provided: must be "
                                         "string 'X', 'Y', or 'Z'.")
                    if not (isinstance(qubit_num, int) and qubit_num >= 0):
                        raise QubitOperatorError("Invalid qubit number "
                                                 "provided to QubitTerm: "
                                                 "must be a non-negative "
                                                 "int.")
                # Sort and add to self.terms:
                term = list(term)
                term.sort(key=lambda loc_operator: loc_operator[0])
                self.terms[tuple(term)] = coefficient
        elif isinstance(term, str):
            list_ops = []
            for el in term.split():
                if len(el) < 2:
                    raise ValueError('term specified incorrectly.')
                list_ops.append((int(el[1:]), el[0]))
            # Test that list_ops has correct format of tuples
            for local_operator in list_ops:
                qubit_num, action = local_operator
                if not isinstance(action, str) or action not in 'XYZ':
                    raise ValueError("Invalid action provided: must be "
                                     "string 'X', 'Y', or 'Z'.")
                if not (isinstance(qubit_num, int) and qubit_num >= 0):
                    raise QubitOperatorError("Invalid qubit number "
                                             "provided to QubitTerm: "
                                             "must be a non-negative "
                                             "int.")
            # Sort and add to self.terms:
            list_ops.sort(key=lambda loc_operator: loc_operator[0])
            self.terms[tuple(list_ops)] = coefficient
        else:
            raise ValueError('term specified incorrectly.')

    def compress(self, abs_tol=1e-12):
        """
        Eliminates all terms with coefficients close to zero and removes
        imaginary parts of coefficients that are close to zero.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        new_terms = {}
        for term in self.terms:
            coeff = self.terms[term]
            if abs(coeff.imag) <= abs_tol:
                coeff = coeff.real
            if abs(coeff) > abs_tol:
                new_terms[term] = coeff
        self.terms = new_terms

    def isclose(self, other, rel_tol=1e-12, abs_tol=1e-12):
        """
        Returns True if other (QubitOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each term in self and other is
        less than the relative tolerance w.r.t. either other or self
        (symmetric test) or if the difference is less than the absolute
        tolerance.

        Args:
            other(QubitOperator): QubitOperator to compare against.
            rel_tol(float): Relative tolerance, must be greater than 0.0
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        # terms which are in both:
        for term in set(self.terms).intersection(set(other.terms)):
            a = self.terms[term]
            b = other.terms[term]
            # math.isclose does this in Python >=3.5
            if not abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol):
                return False
        # terms only in one (compare to 0.0 so only abs_tol)
        for term in set(self.terms).symmetric_difference(set(other.terms)):
            if term in self.terms:
                if not abs(self.terms[term]) <= abs_tol:
                    return False
            elif not abs(other.terms[term]) <= abs_tol:
                return False
        return True

    def __imul__(self, multiplier):
        """
        In-place multiply (*=) terms with scalar or QubitOperator.

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

                    # Finish the remainding operators:
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

    def __mul__(self, multiplier):
        """
        Return self * multiplier for a scalar, or a QubitOperator.

        Args:
          multiplier: A scalar, or a QubitOperator.

        Returns:
          product: A QubitOperator.

        Raises:
          TypeError: Invalid type cannot be multiply with QubitOperator.
        """
        if (isinstance(multiplier, (int, float, complex)) or
                isinstance(multiplier, QubitOperator)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError(
                'Object of invalid type cannot multiply with QubitOperator.')

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  QubitOperator and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of QubitOperator.

        Raises:
          TypeError: Object of invalid type cannot multiply QubitOperator.
        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with QubitOperator.')
        return self * multiplier

    def __truediv__(self, divisor):
        """
        Return self / divisor for a scalar.

        Note:
            This is always floating point division.

        Args:
          divisor: A scalar to divide by.

        Returns:
          A new instance of QubitOperator.

        Raises:
          TypeError: Cannot divide local operator by non-scalar type.

        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide QubitOperator by non-scalar type.')
        return self * (1.0 / divisor)

    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)

    def __itruediv__(self, divisor):
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide QubitOperator by non-scalar type.')
        self *= (1.0 / divisor)
        return self

    def __idiv__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__itruediv__(divisor)

    def __iadd__(self, addend):
        """
        In-place method for += addition of QubitOperator.

        Args:
          addend: A QubitOperator.

        Raises:
          TypeError: Cannot add invalid type.
        """
        if isinstance(addend, QubitOperator):
            for term in addend.terms:
                if term in self.terms:
                    if abs(addend.terms[term] + self.terms[term]) > 0.:
                        self.terms[term] += addend.terms[term]
                    else:
                        del self.terms[term]
                else:
                    self.terms[term] = addend.terms[term]
        else:
            raise TypeError('Cannot add invalid type to QubitOperator.')
        return self

    def __add__(self, addend):
        """ Return self + addend for a QubitOperator. """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __sub__(self, subtrahend):
        """
        Return self - subtrahend for a QubitOperator.

        Args:
          addend: A QubitOperator.

        Raises:
          TypeError: Cannot add invalid type.
        """
        if not isinstance(subtrahend, QubitOperator):
            raise TypeError('Cannot subtract invalid type to QubitOperator.')
        return self + (-1. * subtrahend)

    def __isub__(self, subtrahend):
        """
        In-place method for -= addition of QubitOperator.

        Args:
          subtrahend: A QubitOperator.

        Raises:
          TypeError: Cannot add invalid type.
        """
        if not isinstance(subtrahend, QubitOperator):
            raise TypeError('Cannot subtract invalid type to QubitOperator.')
        return self.__iadd__(-1. * subtrahend)

    def __neg__(self):
        return -1. * self

    def __str__(self):
        """Return an easy-to-read string representation."""
        if not self.terms:
            return '0'
        string_rep = ''
        for term in self.terms:
            tmp_string = '{}'.format(self.terms[term])
            if term == ():
                tmp_string += ' I'
            for operator in term:
                if operator[1] == 'X':
                    tmp_string += ' X{}'.format(operator[0])
                elif operator[1] == 'Y':
                    tmp_string += ' Y{}'.format(operator[0])
                else:
                    assert operator[1] == 'Z'
                    tmp_string += ' Z{}'.format(operator[0])
            string_rep += '{} +\n'.format(tmp_string)
        return string_rep[:-3]

    def __repr__(self):
        return str(self)
