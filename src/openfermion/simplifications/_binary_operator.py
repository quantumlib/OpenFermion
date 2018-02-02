# symbolic binary class -
# TODO: This corresponds to a single element of a decoder. Maybe we can implement a decoder class or just keep using it as a list of these?

import copy

class SymbolicBinaryError(Exception):
    pass

def parse_string(summand):
    term_list = []
    add_one = False
    for factor in summand.split():
        if add_one:
            raise SymbolicBinaryError('Invalid multiplication',summand)
        factor = factor.capitalize()
        if factor == '1':
            term_list.append((1, '1'))
            add_one = True
        elif 'W' in factor:
            q_idx = factor.strip('W')
            if not q_idx.isdigit():
                raise ValueError('Invalid index {}.'.format(q_idx))
            q_idx = int(q_idx)
            if q_idx < 0:
                raise SymbolicBinaryError('Invalid qubit index {}. '
                                          'it should be a non-negative '
                                          'integer.'.format(q_idx))
            term_list.append((q_idx, 'W'))
        else:
            raise ValueError('term specified incorrectly. {}'.format(factor))

    return tuple(term_list)

def binary_sum_rule(terms,additive):
    if additive not in terms:
        terms.append(additive)
    else:
        terms.remove(additive)
    return terms


class SymbolicBinary(object):

    actions = ('1','W')
    action_strings = ('1','W')
    action_before_index = True
    different_indices_commute = True

    def __init__(self, term=None):

        # Detect if the input is the string representation of a sum of terms;
        # if so, initialization needs to be handled differently
        self.terms = []

        if isinstance(term, str) and '+' in term:
            self._long_string_init(term)
            return

        # Initialize the terms dictionary

        # Zero operator: leave the terms list empty
        if term is None:
            return

        # Parse the term
        # Sequence input
        elif isinstance(term, tuple) or isinstance(term, list):
            self.terms = self._parse_sequence(term)

        # String input
        elif isinstance(term, str):
            self.terms.append(self._parse_string(term))
        # Invalid input type
        else:
            raise ValueError('term specified incorrectly.')


    def _long_string_init(self, term):
        """
        Initialization from a long string representation i.e.: 1 + w1 w2 + w3 w4'.
        """
        for summand in term.split(' + '):  # given as ('1 + w1 w2 + w3 w4')
            # for each
            parsed_summand = parse_string(summand)

            if self.different_indices_commute:
                processed_term = sorted(parsed_summand,
                                        key=lambda factor: factor[0])

            self.terms.append(parsed_summand)

    def _check_factor(self,term, factor):
        if len(factor) != 2:
            raise ValueError('Invalid factor {}.'.format(factor))

        index, action = factor
        if action == '1' and len(term) > 1:
            term.remove(factor)  # no need to keep 1 multiplier in the term

        if action not in self.actions:
            raise ValueError('Invalid action in factor {}. '
                             'Valid actions are: {}'.format(
                factor, self.actions))

        if not isinstance(index, int) or index < 0:
            raise ValueError('Invalid index in factor {}. '
                             'The index should be a non-negative '
                             'integer.'.format(factor))

    def _parse_sequence(self, term):
        """Parse a term given as a sequence type (i.e., list, tuple, etc.).
        e.g. [(1,'W'),(2,'W'),...]

        """
        is_summand = False
        term = list(term)
        if not term:
            # Empty sequence
            return ()
        else:
            # Check that all factors in the term are valid
            for factor in term:
                if isinstance(factor[0],tuple):  #factor is actually a summand
                    is_summand = True
                    for real_factor in factor:
                        self._check_factor(factor,real_factor)

                        if self.different_indices_commute:
                            sorted(term, key=lambda real_factor: real_factor[0])

                else:
                    self._check_factor(term,factor)

                # If factors with different indices commute, sort the factors
                # by index
            if not is_summand:
                if self.different_indices_commute:
                    term = sorted(term, key=lambda factor: factor[0])
                term = tuple(term)
            self.terms.append(term)
            # Return a tuple
            return self.terms

    def _parse_string(self, term):
        """Parse a term given as a string.

        e.g. 'W1 W2 W3'
        """

        # Convert the string representations of the factors to tuples
        processed_term = parse_string(term)

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
            additive_identity (SymbolicBinary):
                A symbolic operator o with the property that o+x = x+o = x for
                all operators x of the same class.
        """
        return cls(term=None)

    @classmethod
    def identity(cls):
        """
        Returns:
            multiplicative_identity (SymbolicBinary):
                A symbolic operator u with the property that u*x = x*u = x for
                all operators x of the same class.
        """
        return cls(term=(1,'1'))

    def __str__(self):
        """Return an easy-to-read string representation."""
        if not self.terms:
            return '0'
        string_rep = ''
        for term in self.terms:
            tmp_string = '['
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
            multiplier(complex float, or SymbolicBinary): multiplier
        Returns:
            product (SymbolicBinary): Mutated self.
        """
        # Handle scalars.
        if isinstance(multiplier, (int, float, complex)):
            for term in self.terms:
                mod_mul = multiplier%2
                if not mod_mul:
                    self.terms.remove(term)
            return self

        # Handle operator of the same type
        elif isinstance(multiplier, self.__class__):
            result_terms = []
            for left_term in self.terms:
                left_indices = set([term[0] for term in left_term if term[1]!='1'])

                for right_term in multiplier.terms:
                    right_indices = set([term[0] for term in right_term if term[1]!='1'])

                    if len(left_indices)==0 and len(right_indices)==0:
                        product_term = ((1,'1'),)
                        result_terms = binary_sum_rule(result_terms,product_term)
                        continue

                    indices = left_indices | right_indices # binary rule - 2 w^2 = w
                    product_term = [(qidx, 'W') for qidx in list(indices)]
                    result_terms = binary_sum_rule(result_terms, tuple(product_term))

            self.terms = result_terms
            return self

        # Invalid multiplier type
        else:
            raise TypeError('Cannot multiply {} with {}'.format(
                self.__class__.__name__, multiplier.__class__.__name__))
        

    def __mul__(self, multiplier):
        """Return self * multiplier for a scalar, or a SymbolicBinary.

        Args:
            multiplier: A scalar, or a SymbolicBinary.

        Returns:
            product (SymbolicBinary)

        Raises:
            TypeError: Invalid type cannot be multiply with SymbolicBinary.
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
        """In-place method for += addition of SymbolicBinary.

        Args:
            addend (SymbolicBinary): The operator to add.

        Returns:
            sum (SymbolicBinary): Mutated self.

        Raises:
            TypeError: Cannot add invalid type.
        """
        if isinstance(addend, type(self)):
            for term in addend.terms:
                self.terms = binary_sum_rule(self.terms,term)

        return self

    def __add__(self, addend):
        """
        Args:
            addend (SymbolicBinary): The operator to add.

        Returns:
            sum (SymbolicBinary)
        """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __isub__(self, subtrahend):
        """In-place method for -= subtraction of SymbolicBinary.

        Args:
            subtrahend (A SymbolicBinary): The operator to subtract.

        Returns:
            difference (SymbolicBinary): Mutated self.

        Raises:
            TypeError: Cannot subtract invalid type.
        """
        if isinstance(subtrahend, type(self)):
            for term in subtrahend.terms:
                if term in self.terms:
                    self.terms.remove(term)
                else:
                    raise ValueError('{} is not in {}'.format(term,self.terms))
        else:
            raise TypeError('Cannot subtract invalid type from {}.'.format(
                            type(self)))
        return self

    def __sub__(self, subtrahend):
        """
        Args:
            subtrahend (SymbolicBinary): The operator to subtract.

        Returns:
            difference (SymbolicBinary)
        """
        minuend = copy.deepcopy(self)
        minuend -= subtrahend
        return minuend

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  SymbolicBinary and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of SymbolicBinary.

        Raises:
          TypeError: Object of invalid type cannot multiply SymbolicBinary.
        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with ' +
                type(self) + '.')
        return self * multiplier


    def __pow__(self, exponent):
        """Exponentiate the SymbolicBinary.

        Args:
            exponent (int): The exponent with which to raise the operator.

        Returns:
            exponentiated (SymbolicBinary)

        Raises:
            ValueError: Can only raise SymbolicBinary to non-negative
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

        return self



if __name__ == '__main__':
    b1 = SymbolicBinary('1 + w1 w2')
    print 'b1:',b1.terms
    b2 = SymbolicBinary(((1,'1'),))
    print 'b2:',b2.terms
    b3 = SymbolicBinary(((3,'W'),(4,'W'),(1,'1')))
    print 'b3:',b3.terms
    b4 = b3*b2
    print 'b3*b2:',b4.terms
    b4 = b3*b1
    print 'b3*b1:',b4.terms
    b4 = b1*b2
    print 'b1*b2:',b4.terms
    b4 = b1 * b2
    print 'b1*b2:', b4.terms
    b5 = SymbolicBinary(((1,'W'),(2,'W')))
    print 'b5:',b5.terms
    b4 = b5+b1
    print 'b5+b1:',b4.terms
    b4 = b5*b1
    print 'b5*b1:',b4.terms
    print 'b4*b1:',(b4*b1).terms
    print '(b1+b3)^2:',((b1+b3)**2).terms
    print SymbolicBinary('w1').terms