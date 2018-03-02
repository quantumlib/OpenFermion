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

"""SymbolicOperator is the base class for FermionOperator and QubitOperator"""
import copy
import re

from openfermion.config import EQ_TOLERANCE


class SymbolicOperatorError(Exception):
    pass


def prune_unused_indices(symbolic_operator):
    """
    Remove indices that do not appear in any terms.

    Indices will be renumbered such that if an index i does not appear in
    any terms, then the next largest index that appears in at least one
    term will be renumbered to i.
    """

    # Determine which indices appear in at least one term
    indices = []
    for term in symbolic_operator.terms:
        for op in term:
            if op[0] not in indices:
                indices.append(op[0])
    indices.sort()

    # Construct a dict that maps the old indices to new ones
    index_map = {}
    for index in enumerate(indices):
        index_map[index[1]] = index[0]

    new_operator = copy.deepcopy(symbolic_operator)
    new_operator.terms.clear()

    # Replace the indices in the terms with the new indices
    for term in symbolic_operator.terms:
        new_term = [(index_map[op[0]], op[1]) for op in term]
        new_operator.terms[tuple(new_term)] = symbolic_operator.terms[term]

    return new_operator


class SymbolicOperator(object):
    """Base class for FermionOperator and QubitOperator.

    A SymbolicOperator stores an object which represents a weighted
    sum of terms; each term is a product of individual factors
    of the form (`index`, `action`), where `index` is a nonnegative integer
    and the possible values for `action` are determined by the subclass.
    For instance, for the subclass FermionOperator, `action` can be 1 or 0,
    indicating raising or lowering, and for QubitOperator, `action` is from
    the set {'X', 'Y', 'Z'}.
    The coefficients of the terms are stored in a dictionary whose
    keys are the terms.
    SymbolicOperators of the same type can be added or multiplied together.

    Note:
        Adding SymbolicOperators is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a SymbolicOperator
        with a scalar.

    Attributes:
        actions (tuple): A tuple of objects representing the possible actions.
            This should be defined in the subclass.
            e.g. for FermionOperator, this is (1, 0).
        action_strings (tuple): A tuple of string representations of actions.
            These should be in one-to-one correspondence with actions and
            listed in the same order.
            e.g. for FermionOperator, this is ('^', '').
        action_before_index (bool): A boolean indicating whether in string
            representations, the action should come before the index.
        different_indices_commute (bool): A boolean indicating whether
            factors acting on different indices commute.
        terms (dict):
            **key** (tuple of tuples): A dictionary storing the coefficients
            of the terms in the operator. The keys are the terms.
            A term is a product of individual factors; each factor is
            represented by a tuple of the form (`index`, `action`), and
            these tuples are collected into a larger tuple which represents
            the term as the product of its factors.
    """
    actions = ()
    action_strings = ()
    action_before_index = False
    different_indices_commute = False

    def __init__(self, term=None, coefficient=1.):
        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be a numeric type.')

        # Initialize the terms dictionary
        self.terms = {}

        # Detect if the input is the string representation of a sum of terms;
        # if so, initialization needs to be handled differently
        if isinstance(term, str) and '[' in term:
            self._long_string_init(term, coefficient)
            return

        # Zero operator: leave the terms dictionary empty
        if term is None:
            return

        # Parse the term
        # Sequence input
        elif isinstance(term, tuple) or isinstance(term, list):
            term = self._parse_sequence(term)
        # String input
        elif isinstance(term, str):
            term = self._parse_string(term)
        # Invalid input type
        else:
            raise ValueError('term specified incorrectly.')

        # Add the term to the dictionary
        self.terms[term] = coefficient

    def _long_string_init(self, long_string, coefficient):
        """
        Initialization from a long string representation.

        e.g. For FermionOperator:
            '1.5 [2^ 3] + 1.4 [3^ 0]'
        """

        pattern = '(.*?)\[(.*?)\]'  # regex for a term
        for match in re.findall(pattern, long_string, flags=re.DOTALL):

            # Determine the coefficient for this term
            coef_string = re.sub(r"\s+", "", match[0])
            if coef_string and coef_string[0] is '+':
                coef_string = coef_string[1:].strip()
            if coef_string == '':
                coef = 1.0
            elif coef_string == '-':
                coef = -1.0
            else:
                try:
                    if 'j' in coef_string:
                        if coef_string[0] == '-':
                            coef = -complex(coef_string[1:])
                        else:
                            coef = complex(coef_string)
                    else:
                        coef = float(coef_string)
                except ValueError:
                    raise ValueError(
                            'Invalid coefficient {}.'.format(coef_string))
            coef *= coefficient

            # Parse the term and add it to the dict
            term = self._parse_string(match[1])
            if term not in self.terms:
                self.terms[term] = coef
            else:
                self.terms[term] += coef

    def _parse_sequence(self, term):
        """Parse a term given as a sequence type (i.e., list, tuple, etc.).

        e.g. For QubitOperator:
            [('X', 2), ('Y', 0), ('Z', 3)] -> (('Y', 0), ('X', 2), ('Z', 3))
        """
        if not term:
            # Empty sequence
            return ()
        else:
            # Check that all factors in the term are valid
            for factor in term:
                if len(factor) != 2:
                    raise ValueError('Invalid factor {}.'.format(factor))

                index, action = factor

                if action not in self.actions:
                    raise ValueError('Invalid action in factor {}. '
                                     'Valid actions are: {}'.format(
                                         factor, self.actions))

                if not isinstance(index, int) or index < 0:
                    raise ValueError('Invalid index in factor {}. '
                                     'The index should be a non-negative '
                                     'integer.'.format(factor))

            # If factors with different indices commute, sort the factors
            # by index
            if self.different_indices_commute:
                term = sorted(term, key=lambda factor: factor[0])

            # Return a tuple
            return tuple(term)

    def _parse_string(self, term):
        """Parse a term given as a string.

        e.g. For FermionOperator:
            "2^ 3" -> ((2, 1), (3, 0))
        """
        factors = term.split()

        # Convert the string representations of the factors to tuples
        processed_term = []
        for factor in factors:
            # Get the index and action string
            if self.action_before_index:
                # The index is at the end of the string; find where it starts.
                if not factor[-1].isdigit():
                    raise ValueError('Invalid factor {}.'.format(factor))
                index_start = len(factor) - 1
                while index_start > 0 and factor[index_start - 1].isdigit():
                    index_start -= 1

                index = int(factor[index_start:])
                action_string = factor[:index_start]
            else:
                # The index is at the beginning of the string; find where
                # it ends
                if not factor[0].isdigit():
                    raise ValueError('Invalid factor {}.'.format(factor))
                index_end = 1
                while (index_end <= len(factor) - 1 and
                        factor[index_end].isdigit()):
                    index_end += 1

                index = int(factor[:index_end])
                action_string = factor[index_end:]
            # Check that the index is valid
            if index < 0:
                raise ValueError('Invalid index in factor {}. '
                                 'The index should be a non-negative '
                                 'integer.'.format(factor))
            # Convert the action string to an action
            if action_string in self.action_strings:
                action = self.actions[self.action_strings.index(action_string)]
            else:
                raise ValueError('Invalid action in factor {}. '
                                 'Valid actions are: {}'.format(
                                     factor, self.action_strings))

            # Add the factor to the list as a tuple
            processed_term.append((index, action))

        # If factors with different indices commute, sort the factors
        # by index
        if self.different_indices_commute:
            processed_term = sorted(processed_term,
                                    key=lambda factor: factor[0])

        # Return a tuple
        return tuple(processed_term)

    @classmethod
    def zero(cls):
        """
        Returns:
            additive_identity (SymbolicOperator):
                A symbolic operator o with the property that o+x = x+o = x for
                all operators x of the same class.
        """
        return cls(term=None)

    @classmethod
    def identity(cls):
        """
        Returns:
            multiplicative_identity (SymbolicOperator):
                A symbolic operator u with the property that u*x = x*u = x for
                all operators x of the same class.
        """
        return cls(term=())

    def __str__(self):
        """Return an easy-to-read string representation."""
        if not self.terms:
            return '0'
        string_rep = ''
        for term in self.terms:
            tmp_string = '{} ['.format(self.terms[term])
            for factor in term:
                index, action = factor
                action_string = self.action_strings[self.actions.index(action)]
                if self.action_before_index:
                    tmp_string += '{}{} '.format(action_string, index)
                else:
                    tmp_string += '{}{} '.format(index, action_string)
            string_rep += '{}] +\n'.format(tmp_string.strip())
        return string_rep[:-3]

    def __repr__(self):
        return str(self)

    def __imul__(self, multiplier):
        """In-place multiply (*=) with scalar or operator of the same type.

        Default implementation is to multiply coefficients and
        concatenate terms.

        Args:
            multiplier(complex float, or SymbolicOperator): multiplier
        Returns:
            product (SymbolicOperator): Mutated self.
        """
        # Handle scalars.
        if isinstance(multiplier, (int, float, complex)):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle operator of the same type
        elif isinstance(multiplier, self.__class__):
            result_terms = dict()
            for left_term in self.terms:
                for right_term in multiplier.terms:
                    new_coefficient = (self.terms[left_term] *
                                       multiplier.terms[right_term])
                    product_operators = left_term + right_term

                    # Update result dict.
                    product_operators = tuple(product_operators)
                    if product_operators in result_terms:
                        result_terms[product_operators] += new_coefficient
                    else:
                        result_terms[product_operators] = new_coefficient
            self.terms = result_terms
            return self

        # Invalid multiplier type
        else:
            raise TypeError('Cannot multiply {} with {}'.format(
                self.__class__.__name__, multiplier.__class__.__name__))

    def __mul__(self, multiplier):
        """Return self * multiplier for a scalar, or a SymbolicOperator.

        Args:
            multiplier: A scalar, or a SymbolicOperator.

        Returns:
            product (SymbolicOperator)

        Raises:
            TypeError: Invalid type cannot be multiply with SymbolicOperator.
        """
        if isinstance(multiplier, (int, float, complex, type(self))):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError(
                'Object of invalid type cannot multiply with ' +
                type(self) + '.')

    def __iadd__(self, addend):
        """In-place method for += addition of SymbolicOperator.

        Args:
            addend (SymbolicOperator): The operator to add.

        Returns:
            sum (SymbolicOperator): Mutated self.

        Raises:
            TypeError: Cannot add invalid type.
        """
        if isinstance(addend, type(self)):
            for term in addend.terms:
                self.terms[term] = (self.terms.get(term, 0.0) +
                                    addend.terms[term])
                if abs(self.terms[term]) < EQ_TOLERANCE:
                    del self.terms[term]
        else:
            raise TypeError('Cannot add invalid type to {}.'.format(
                            type(self)))

        return self

    def __add__(self, addend):
        """
        Args:
            addend (SymbolicOperator): The operator to add.

        Returns:
            sum (SymbolicOperator)
        """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __isub__(self, subtrahend):
        """In-place method for -= subtraction of SymbolicOperator.

        Args:
            subtrahend (A SymbolicOperator): The operator to subtract.

        Returns:
            difference (SymbolicOperator): Mutated self.

        Raises:
            TypeError: Cannot subtract invalid type.
        """
        if isinstance(subtrahend, type(self)):
            for term in subtrahend.terms:
                self.terms[term] = (self.terms.get(term, 0.0) -
                                    subtrahend.terms[term])
                if abs(self.terms[term]) < EQ_TOLERANCE:
                    del self.terms[term]
        else:
            raise TypeError('Cannot subtract invalid type from {}.'.format(
                            type(self)))
        return self

    def __sub__(self, subtrahend):
        """
        Args:
            subtrahend (SymbolicOperator): The operator to subtract.

        Returns:
            difference (SymbolicOperator)
        """
        minuend = copy.deepcopy(self)
        minuend -= subtrahend
        return minuend

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  SymbolicOperator and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of SymbolicOperator.

        Raises:
          TypeError: Object of invalid type cannot multiply SymbolicOperator.
        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with ' +
                type(self) + '.')
        return self * multiplier

    def __truediv__(self, divisor):
        """
        Return self / divisor for a scalar.

        Note:
            This is always floating point division.

        Args:
          divisor: A scalar to divide by.

        Returns:
          A new instance of SymbolicOperator.

        Raises:
          TypeError: Cannot divide local operator by non-scalar type.

        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide ' + type(self) +
                            ' by non-scalar type.')
        return self * (1.0 / divisor)

    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)

    def __itruediv__(self, divisor):
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide ' + type(self) +
                            ' by non-scalar type.')
        self *= (1.0 / divisor)
        return self

    def __idiv__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__itruediv__(divisor)

    def __neg__(self):
        """
        Returns:
            negation (SymbolicOperator)
        """
        return -1 * self

    def __pow__(self, exponent):
        """Exponentiate the SymbolicOperator.

        Args:
            exponent (int): The exponent with which to raise the operator.

        Returns:
            exponentiated (SymbolicOperator)

        Raises:
            ValueError: Can only raise SymbolicOperator to non-negative
                integer powers.
        """
        # Handle invalid exponents.
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError(
                'exponent must be a non-negative int, but was {} {}'.format(
                    type(exponent), repr(exponent)))

        # Initialized identity.
        exponentiated = self.__class__(())

        # Handle non-zero exponents.
        for i in range(exponent):
            exponentiated *= self
        return exponentiated

    def compress(self, abs_tol=EQ_TOLERANCE):
        """
        Eliminates all terms with coefficients close to zero and removes
        small imaginary and real parts.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        new_terms = {}
        for term in self.terms:
            coeff = self.terms[term]

            # Remove small imaginary and real parts
            if abs(coeff.imag) <= abs_tol:
                coeff = coeff.real
            if abs(coeff.real) <= abs_tol:
                coeff = 1.j * coeff.imag

            # Add the term if the coefficient is large enough
            if abs(coeff) > abs_tol:
                new_terms[term] = coeff

        self.terms = new_terms

    def isclose(self, other, rel_tol=EQ_TOLERANCE, abs_tol=EQ_TOLERANCE):
        """
        Returns True if other (SymbolicOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each term in self and other is
        less than the relative tolerance w.r.t. either other or self
        (symmetric test) or if the difference is less than the absolute
        tolerance.

        Args:
            other(SymbolicOperator): SymbolicOperator to compare against.
            rel_tol(float): Relative tolerance, must be greater than 0.0
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        if not isinstance(other, type(self)):
            raise TypeError('Cannot compare a {} with a {}'.format(
                type(self).__name__, type(other).__name__))

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

    def induced_norm(self, order=1):
        """
        Compute the induced p-norm of the operator.

        If we represent an operator as
        :math: `\sum_{j} w_j H_j`
        where :math: `w_j` are scalar coefficients then this norm is
        :math: `\left(\sum_{j} \| w_j \|^p \\right)^{\\frac{1}{p}}
        where :math: `p` is the order of the induced norm

        Args:
            order(int): the order of the induced norm.
        """
        norm = 0.
        for coefficient in self.terms.values():
            norm += abs(coefficient) ** order
        return norm ** (1. / order)
