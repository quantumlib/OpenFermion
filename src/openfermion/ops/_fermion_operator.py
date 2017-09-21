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
import copy
from openfermion.config import *
from future.utils import iteritems
import numpy


class FermionOperatorError(Exception):
    pass


def hermitian_conjugated(fermion_operator):
    """Return Hermitian conjugate of fermionic operator."""
    conjugate_operator = FermionOperator()
    for term, coefficient in iteritems(fermion_operator.terms):
        conjugate_term = tuple([(tensor_factor, 1 - action) for
                                (tensor_factor, action) in reversed(term)])
        conjugate_operator.terms[conjugate_term] = numpy.conjugate(coefficient)
    return conjugate_operator


def number_operator(n_orbitals, orbital=None, coefficient=1.):
    """Return a number operator.

    Args:
        n_orbitals (int): The number of spin-orbitals in the system.
        orbital (int, optional): The orbital on which to return the number
            operator. If None, return total number operator on all sites.
        coefficient (float): The coefficient of the term.
    Returns:
        operator (FermionOperator)
    """
    if orbital is None:
        operator = FermionOperator()
        for spin_orbital in range(n_orbitals):
            operator += number_operator(n_orbitals, spin_orbital, coefficient)
    else:
        operator = FermionOperator(((orbital, 1), (orbital, 0)), coefficient)
    return operator


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


def _parse_ladder_operator(ladder_operator_text):
    """
    Args:
        ladder_operator_text (str):
            A ladder operator term like '4' or '5^', or an invalid string.
    Returns:
        tuple[int, int]: The mode then raise-vs-lower.
    Raises:
        FermionOperatorError: Given invalid text that doesn't match /\d+^?/ .
    """
    inverted = 1 if ladder_operator_text.endswith('^') else 0
    mode_text = ladder_operator_text[:-1] if inverted else ladder_operator_text

    try:
        mode = int(mode_text)
        if mode < 0:
            raise ValueError()  # Merge with not-an-int failure case.
    except ValueError:
        raise FermionOperatorError(
            "Invalid ladder operator term '{}'.".format(ladder_operator_text))

    return mode, inverted


class FermionOperator(object):
    """FermionOperator stores a sum of products of fermionic ladder operators.

    In FermiLib, we describe fermionic ladder operators using the shorthand:
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

    Attributes:
        terms (dict):
            **key** (tuple of tuples): Each tuple represents a fermion term,
            i.e. a tensor product of fermion ladder operators with a
            coefficient. The first element is an integer indicating the
            mode on which a ladder operator acts and the second element is
            a bool, either '0' indicating annihilation, or '1' indicating
            creation in that mode; for example, '2^ 5' is ((2, 1), (5, 0)).
            **value** (complex float): The coefficient of term represented by
            key.
    """
    def __init__(self, term=None, coefficient=1.):
        """Initializes a FermionOperator.

        The init function only allows to initialize a FermionOperator
        consisting of a single term. If one desires to initialize a
        FermionOperator consisting of many terms, one must add those terms
        together by using either += (which is fast) or using +.

        Example:
            .. code-block:: python

                ham = (FermionOperator('0^ 3', .5)
                       + .5 * FermionOperator('3^ 0'))
                # Equivalently
                ham2 = FermionOperator('0^ 3', 0.5)
                ham2 += FermionOperator('3^ 0', 0.5)

        Note:
            Adding terms to FermionOperator is faster using += (as this
            is done by in-place addition). Specifying the coefficient in
            the __init__ is faster than by multiplying a QubitOperator
            with a scalar as calls an out-of-place multiplication.

        Args:
            term (tuple of tuples, a string, or optional):
                1) A tuple of tuples. The first element of each tuple is
                   an integer indicating the mode on which a fermion
                   ladder operator acts, starting from zero. The second
                   element of each tuple is an integer, either 1 or 0,
                   indicating whether creation or annihilation acts on
                   that mode.
                2) A string of the form '0^ 2', indicating creation in
                   mode 0 and annihilation in mode 2.
                3) default will result in the zero operator.
            coefficient (complex float, optional): The coefficient of the term.
                Default value is 1.0.

        Raises:
            FermionOperatorError: Invalid term provided to FermionOperator.
        """
        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be scalar.')
        self.terms = {}
        if term is None:
            return

        # String input.
        if isinstance(term, str):
            ladder_operators = tuple(_parse_ladder_operator(e)
                                     for e in term.split())
            self.terms[tuple(ladder_operators)] = coefficient

        # Tuple input.
        elif isinstance(term, tuple):
            self.terms[term] = coefficient

        # Invalid input.
        else:
            raise ValueError('Operators specified incorrectly.')

        # Check type.
        for stored_term in self.terms:
            for ladder_operator in stored_term:
                orbital, action = ladder_operator
                if not (isinstance(orbital, int) and orbital >= 0):
                    raise FermionOperatorError(
                        'Invalid tensor factor in FermionOperator:'
                        'must be a non-negative int.')
                if action not in (0, 1):
                    raise ValueError(
                        'Invalid action in FermionOperator: '
                        'Must be 0 (lowering) or 1 (raising).')

    @staticmethod
    def zero():
        """
        Returns:
            additive_identity (FermionOperator):
                A fermion operator o with the property that o+x = x+o = x for
                all fermion operators x.
        """
        return FermionOperator(term=None)

    @staticmethod
    def identity():
        """
        Returns:
            multiplicative_identity (FermionOperator):
                A fermion operator u with the property that u*x = x*u = x for
                all fermion operators x.
        """
        return FermionOperator(term=())

    def compress(self, abs_tol=EQ_TOLERANCE):
        """
        Eliminates all terms with coefficients close to zero and removes
        imaginary parts of coefficients that are close to zero.

        Args:
            abs_tol (float): Absolute tolerance, must be at least 0.0
        """
        new_terms = {}
        for term in self.terms:
            coeff = self.terms[term]
            if abs(coeff.imag) <= abs_tol:
                coeff = coeff.real
            if abs(coeff) > abs_tol:
                new_terms[term] = coeff
        self.terms = new_terms

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

    def __str__(self):
        """Return an easy-to-read string representation."""
        if not self.terms:
            return '0'
        string_rep = ''
        for term in self.terms:
            tmp_string = '{} ['.format(self.terms[term])
            for operator in term:
                if operator[1] == 1:
                    tmp_string += '{}^ '.format(operator[0])
                elif operator[1] == 0:
                    tmp_string += '{} '.format(operator[0])
            string_rep += '{}] +\n'.format(tmp_string.strip())
        return string_rep[:-3]

    def __repr__(self):
        return str(self)

    def isclose(self, other, rel_tol=EQ_TOLERANCE, abs_tol=EQ_TOLERANCE):
        """Returns True if other (FermionOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each terms in self and other is
        less than the relative tolerance w.r.t. either other or self
        (symmetric test) or if the difference is less than the absolute
        tolerance.

        Args:
            other (FermionOperator): FermionOperator to compare against.
            rel_tol (float): Relative tolerance, must be greater than 0.0
            abs_tol (float): Absolute tolerance, must be at least 0.0
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
        """In-place multiply (*=) terms with scalar or FermionOperator.

        Args:
          multiplier(complex float, or FermionOperator): multiplier
        Returns:
            product (FermionOperator): Mutated self.
        """
        # Handle scalars.
        if isinstance(multiplier, (int, float, complex)):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle FermionOperator.
        elif isinstance(multiplier, FermionOperator):
            result_terms = dict()
            for left_term in self.terms:
                for right_term in multiplier.terms:
                    new_coefficient = (self.terms[left_term] *
                                       multiplier.terms[right_term])
                    product_operators = left_term + right_term

                    # Add to result dict.
                    result_terms[tuple(product_operators)] = new_coefficient
            self.terms = result_terms
            return self
        else:
            raise TypeError('Cannot in-place multiply term of invalid type '
                            'to FermionOperator.')

    def __mul__(self, multiplier):
        """Return self * multiplier for a scalar, or a FermionOperator.

        Args:
            multiplier: A scalar, or a FermionOperator.

        Returns:
            product (FermionOperator)

        Raises:
            TypeError: Invalid type cannot be multiply with FermionOperator.
        """
        if isinstance(multiplier, (int, float, complex, FermionOperator)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError(
                'Object of invalid type cannot multiply with FermionOperator.')

    def __rmul__(self, multiplier):
        """Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for FermionOperator and left multiply
        is also queried as the default behavior.

        Args:
            multiplier: A scalar to multiply by.

        Returns:
            product (FermionOperator)

        Raises:
            TypeError: Object of invalid type cannot multiply FermionOperator.
        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with FermionOperator.')
        return self * multiplier

    def __truediv__(self, divisor):
        """Return self / divisor for a scalar.

        Note that this is always floating point division.

        Args:
            divisor (int|float|complex): A scalar to divide by.

        Returns:
            quotient (FermionOperator)

        Raises:
            TypeError: Cannot divide local operator by non-scalar type.
        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide QubitOperator by non-scalar type.')
        return self * (1.0 / divisor)

    def __div__(self, divisor):
        """For compatibility with Python 2.
        Args:
            divisor (int|float|complex): A scalar to divide by.
        Returns:
            quotient (FermionOperator)
        """
        return self.__truediv__(divisor)

    def __itruediv__(self, divisor):
        """
        Args:
            divisor (int|float|complex): A scalar to divide by.
        Returns:
            quotient (FermionOperator): Mutated self.
        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide QubitOperator by non-scalar type.')
        self *= (1.0 / divisor)
        return self

    def __idiv__(self, divisor):
        """For compatibility with Python 2.
        Args:
            divisor (int|float|complex): A scalar to divide by.
        Returns:
            quotient (FermionOperator): Mutated self.
        """
        return self.__itruediv__(divisor)

    def __iadd__(self, addend):
        """In-place method for += addition of FermionOperator.

        Args:
            addend (FermionOperator): The operator to add.

        Returns:
            sum (FermionOperator): Mutated self.

        Raises:
            TypeError: Cannot add invalid type.
        """
        if isinstance(addend, FermionOperator):
            for term in addend.terms:
                if term in self.terms:
                    if abs(addend.terms[term] +
                           self.terms[term]) < EQ_TOLERANCE:
                        del self.terms[term]
                    else:
                        self.terms[term] += addend.terms[term]
                else:
                    self.terms[term] = addend.terms[term]
        else:
            raise TypeError('Cannot add invalid type to FermionOperator.')
        return self

    def __add__(self, addend):
        """
        Args:
            addend (FermionOperator): The operator to add.

        Returns:
            sum (FermionOperator)
        """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __isub__(self, subtrahend):
        """In-place method for -= subtraction of FermionOperator.

        Args:
            subtrahend (A FermionOperator): The operator to subtract.

        Returns:
            difference (FermionOperator): Mutated self.

        Raises:
            TypeError: Cannot subtract invalid type.
        """
        if isinstance(subtrahend, FermionOperator):
            for term in subtrahend.terms:
                if term in self.terms:
                    if abs(self.terms[term] -
                           subtrahend.terms[term]) < EQ_TOLERANCE:
                        del self.terms[term]
                    else:
                        self.terms[term] -= subtrahend.terms[term]
                else:
                    self.terms[term] = -subtrahend.terms[term]
        else:
            raise TypeError('Cannot subtract invalid type.')
        return self

    def __sub__(self, subtrahend):
        """
        Args:
            subtrahend (FermionOperator): The operator to subtract.

        Returns:
            difference (FermionOperator)
        """
        minuend = copy.deepcopy(self)
        minuend -= subtrahend
        return minuend

    def __neg__(self):
        """
        Returns:
            negation (FermionOperator)
        """
        return -1 * self

    def __pow__(self, exponent):
        """Exponentiate the FermionOperator.

        Args:
            exponent (int): The exponent with which to raise the operator.

        Returns:
            exponentiated (FermionOperator)

        Raises:
            ValueError: Can only raise FermionOperator to non-negative
                integer powers.
        """
        # Handle invalid exponents.
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError(
                'exponent must be a non-negative int, but was {} {}'.format(
                    type(exponent), repr(exponent)))

        # Initialized identity.
        exponentiated = FermionOperator(())

        # Handle non-zero exponents.
        for i in range(exponent):
            exponentiated *= self
        return exponentiated
