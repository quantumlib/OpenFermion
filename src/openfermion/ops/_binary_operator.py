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

"""symbolic binary class for decoder definitions (arxiv 1712.07067) """

import copy

import numpy


class SymbolicBinaryError(Exception):
    pass


class SymbolicBinary(object):
    """The SymbolicBinary class provides an analytic representation
    of non-linear binary functions. An instance of this class describes
    a term of binary variables (variables of the values {0,1}, indexed
    by integers like w0, w1, w2 and so on) that is considered to be evaluated
    modulo 2. This implies the following set of rules:
    
    the binary addition  w1 + w1 = 0, 
    binary multiplication w2 * w2 = w2 
    and power rule w3 ^ 0  = 1, where raising to every other
    integer power than zero reproduces w3.
    
    Of course, we can also add a non-trivial constant, which is 1.
    Due to these binary rules, every function available will be a 
    multinomial like e.g.
    
    1 + w1 w2 + w0 w1 .
    
    These binary functions are used for non-linear binary codes in order
    to decompress qubit bases back into fermion bases.  
    In that instance, one SymbolicBinary object characterizes the occupation 
    of single orbital given a multi-qubit state in configuration 
    |w0> |w1> |w2> ... . 
    
    For initialization, the preferred data types is either a string of the 
    multinomial, where each variable and constant is to be well separated by
    a whitespace, or in its native form of nested tuples, 
    where there are two data types:
    1) a variable on qubit n:  (n, 'W')
    2) a constant term (1,'1') .
    
    A multinomial is a tuple (containing the sumamnds) of tuples 
    (containing the factors) of the above. After initialization,
    SymbolicBinary terms can be manipulated with the overloaded signs 
    +, * and ^, according to the binary rules mentioned.  

    Example:
        .. code-block:: python

            bin_fun = SymbolicBinary('1 + w1 w2 + w0 w1')
            # Equivalently
            bin_fun = SymbolicBinary([((1,'1'),)] +
            SymbolicBinary([((1,'W'),(2,'W')),((0,'W'),(1,'W'))])

    Attributes:
        terms (list): a list of tuples. Each tuple represents a summand of the
            SymbolicBinary term and each summand can contain multiple tuples
            representing the factors.
    """
    actions = ('1', 'W')
    action_strings = ('', 'W')

    # these are just informative for this class - not used
    action_before_index = True
    different_indices_commute = True

    def __init__(self, term=None):
        """ Initialize the SymbolicBinary based on term

        Args:
            term (str, list, tuple): used for initializing a SymbolicBinary

        Raises:
            ValueError: when term is not a string,list or tuple
        """
        self.terms = []

        # long string input
        if isinstance(term, str) and '+' in term:
            self._long_string_init(term)
            return

        # Zero operator: leave the terms list empty
        if term is None:
            return

        # Sequence input: list of tuples of tuples
        elif isinstance(term, tuple) or isinstance(term, list):
            self._parse_sequence(term)

        # String input
        elif isinstance(term, str):
            self.terms.append(tuple(self._parse_string(term)))

        # Invalid input type
        else:
            raise ValueError('term specified incorrectly.')

        self._check_terms()

    def _check_terms(self):
        """ Ensures all terms obey binary rules, updates terms in place."""
        sorted_input = []
        for item in self.terms:
            if len(item):
                binary_sum_rule(sorted_input, tuple(sorted(set(item))))

        self.terms = sorted_input

    def _long_string_init(self, term):
        """Initialization from a long string representation i.e.:
        1 + w1 w2 + w3 w4'. Updates terms in place.

        Args:
            term (str): a string representation of a SymbolicBinary
             that involves summation
        """
        for summand in term.split(' + '):
            # for each sum term
            parsed_summand = self._parse_string(summand)
            self.terms.append(parsed_summand)

        self._check_terms()

    def _check_factor(self, term, factor):
        """Checks types and values of all factors in a term,
        removes multiplications with 1.

        Args:
            term (list): a single term for SymbolicBinary
            factor: a factor in term

        Returns (list): updated term

        Raises:
            ValueError: invalid action/negative or non-integer qubit index
        """
        if len(factor) != 2:
            raise ValueError('Invalid factor {}.'.format(factor))

        term = list(term)
        index, action = factor

        if action not in self.actions:
            raise ValueError('Invalid action in factor {}. '
                             'Valid actions are: {}'.format(factor,
                                                            self.actions))

        if not isinstance(index, int) or index < 0:
            raise ValueError('Invalid index in factor {}.'
                             'The index should be a non-negative '
                             'integer.'.format(factor))

        if action == '1' and len(term) > 1:
            term.remove(factor)  # no need to keep 1 multiplier in the term

        return term

    def _parse_sequence(self, term):
        """ Parse a term given as a sequence type (i.e., list, tuple, etc.).
        e.g. [((1,'W'),(2,'W')),...]. Updates terms in place

        Args:
            term (list): of tuples of tuples.
        """
        if term:
            for summand in term:
                for factor in summand:
                    summand = self._check_factor(summand, factor)

                if self.different_indices_commute:
                    summand = sorted(summand,
                                     key=lambda real_factor: real_factor[0])
                self.terms.append(summand)
        else:
            self.terms = []

    @staticmethod
    def _parse_string(term):
        """ Parse a string term like 'w1 w2 w0'

        Args:
            term (str): string representation of SymbolicBinary term.

        Returns (tuple): parsed string term

        Raises:
          ValueError: Incorrect terms
        """
        term_list = []
        add_one = False
        for factor in term.split():

            # if 1 is already present; remove it
            if add_one:
                term_list.remove((1, '1'))  # if 1
                add_one = False

            factor = factor.capitalize()
            if 'W' in factor:
                q_idx = factor.strip('W')
                if q_idx.isdigit():
                    q_idx = int(q_idx)
                    term_list.append((q_idx, 'W'))
                else:
                    raise ValueError('Invalid index {}.'.format(q_idx))
            elif factor.isdigit():
                factor = int(factor) % 2
                if factor == 1:
                    # if there are other terms, no need to add another 1
                    if len(term_list) > 0:
                        continue
                    term_list.append((1, '1'))
                    add_one = True
                elif factor == 0:
                    return []
            else:
                raise ValueError(
                    'term specified incorrectly. {}'.format(factor))

        parsed_term = tuple(term_list)
        parsed_term = sorted(parsed_term, key=lambda multiplier: multiplier[0])
        return parsed_term

    def enumerate_qubits(self):
        """ Enumerates all qubits indexed in a given SymbolicBinary.

        Returns (list): a list of qubits
        """
        term_array = list(map(numpy.array, self.terms))
        qubits = []
        for summand in term_array:
            for factor in summand:
                if factor[1] != '1':
                    qubits.append(int(factor[0]))

        qubits = list(set(qubits))

        return qubits

    def shift(self, const):
        """ Shift all qubit indices by a given constant.

        Args:
            const (int): the constant to shift the indices by

        Raises:
            TypeError: const must be integer
        """
        if not isinstance(const, (numpy.int64, numpy.int32, int)):
            raise TypeError('can only shift qubit indices by an integer'
                            'received {}'.format(const))
        shifted_terms = []
        for summand in self.terms:
            shifted_summand = []
            for factor in summand:
                qubit, action = factor
                if action == 'W':
                    shifted_summand.append((qubit + const, 'W'))
                else:
                    shifted_summand.append(factor)
            shifted_terms.append(tuple(sorted(shifted_summand)))

        self.terms = shifted_terms

    def evaluate(self, binary_list):
        """Evaluates a SymbolicBinary

        Args:
            binary_list (list): a list of binary values corresponding
                each (n,'W').

        Returns (int, 0 or 1): result of the evaluation

        Raises:
          SymbolicBinaryError: Length of list provided must match the number
                of qubits indexed in symbolicBinary
        """
        all_qubits = self.enumerate_qubits()
        if max(all_qubits) + 1 != len(binary_list):
            raise SymbolicBinaryError(
                'the length of the binary list provided does not match'
                ' the number of qubits in the decoder')

        evaluation = 0
        for summand in self.terms:
            ev_tmp = 1.0
            for factor in summand:
                qubit, action = factor
                if action == 'W':
                    ev_tmp *= binary_list[qubit]
                else:
                    ev_tmp *= 1
            evaluation += ev_tmp

        return evaluation % 2

    def _add_one(self):
        """ Adds constant 1 to a SymbolicBinary. """

        # ((1,'1'),) can only exist as a loner in SymbolicBinary
        if ((1, '1'),) in self.terms:
            self.terms.remove(((1, '1'),))
        else:
            self.terms.append(((1, '1'),))

    @classmethod
    def zero(cls):
        """
        Returns:
            additive_identity (SymbolicBinary):
                A symbolic operator o with the property that o+x = x+o = x for
                all operators x of the same class.
        """
        return cls(term=[])

    @classmethod
    def identity(cls):
        """
        Returns:
            multiplicative_identity (SymbolicBinary):
                A symbolic operator u with the property that u*x = x*u = x for
                all operators x of the same class.
        """
        return cls(term=[((1, '1'),)])

    def __str__(self):
        """ Return an easy-to-read string representation."""
        if not self.terms:
            return '0'
        string_rep = ''
        for term in self.terms:
            tmp_string = '['
            for factor in term:
                index, action = factor
                action_string = self.action_strings[self.actions.index(action)]
                tmp_string += '{}{} '.format(action_string, index)
            string_rep += '{}] + '.format(tmp_string.strip())
        return string_rep[:-3]

    def __repr__(self):
        return str(self)

    def __imul__(self, multiplier):
        """ In-place multiply (*=) with a scalar or operator of the same type.

        Args:
            multiplier(int or SymbolicBinary): multiplier

        Returns:
            product (SymbolicBinary): Mutated self.

        Raises:
          TypeError: Object of invalid type cannot multiply SymbolicBinary.
        """

        # Handle integers.
        if isinstance(multiplier, (numpy.int64, numpy.int32, int)):
            mod_mul = int(multiplier % 2)
            if mod_mul:
                return self
            else:
                return self.zero()

        # Handle operator of the same type
        elif isinstance(multiplier, self.__class__):
            result_terms = []
            for left_term in self.terms:
                left_indices = set(
                    [term[0] for term in left_term if term[1] != '1'])

                for right_term in multiplier.terms:
                    right_indices = set(
                        [term[0] for term in right_term if term[1] != '1'])

                    if len(left_indices) == 0 and len(right_indices) == 0:
                        product_term = ((1, '1'),)
                        binary_sum_rule(result_terms, product_term)
                        continue

                    # binary rule - 2 w^2 = w
                    indices = left_indices | right_indices
                    product_term = [(qubit_index, 'W') for qubit_index in
                                    sorted(list(indices))]
                    binary_sum_rule(result_terms, tuple(product_term))

            self.terms = result_terms
            return self

        # Invalid multiplier type
        else:
            raise TypeError('Cannot multiply {} with {}'.format(
                self.__class__.__name__, multiplier.__class__.__name__))

    def __rmul__(self, multiplier):
        """ Return multiplier * self for a scalar or SymbolicBinary.

        Args:
          multiplier (int or SymbolicBinary): the multiplier of the
           SymbolicBinary object

        Returns:
          product (SymbolicBinary): A new instance of SymbolicBinary.

        Raises:
          TypeError: Object of invalid type cannot multiply SymbolicBinary.
        """
        if not isinstance(multiplier, (numpy.int64, numpy.int32, int,
                                       type(self))):
            raise TypeError(
                'Object of invalid type cannot multiply with ' +
                str(type(self)) + '.')
        return self * multiplier

    def __mul__(self, multiplier):
        """Return self * multiplier for int, or a SymbolicBinary.

        Args:
            multiplier (int or SymbolicBinary): the multiplier of the
           SymbolicBinary object

        Returns:
            product (SymbolicBinary): result of the multiplication

        Raises:
            TypeError: Invalid type cannot be multiply with SymbolicBinary.
        """
        if isinstance(multiplier, (numpy.int64, numpy.int32, int,
                                   type(self))):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError(
                'Object of invalid type cannot multiply with ' +
                str(type(self)) + '.')

    def __iadd__(self, addend):
        """In-place method for += addition of a int or a SymbolicBinary.

        Args:
            addend (int or SymbolicBinary): The operator to add.

        Returns:
            sum (SymbolicBinary): Mutated self.

        Raises:
            TypeError: Cannot add invalid type.
        """
        if isinstance(addend, type(self)):
            for term in addend.terms:
                binary_sum_rule(self.terms, term)
        if isinstance(addend, int):
            mod_add = addend % 2
            if mod_add:
                self._add_one()
        if not isinstance(addend, (numpy.int64, numpy.int32, int,
                                   type(self))):
            raise TypeError(
                'Object of invalid type cannot add with ' +
                str(type(self)) + '.')
        return self

    def __radd__(self, addend):
        """Method for right addition to SymbolicBinary.

        Args:
            addend (int or SymbolicBinary): The operator to add.

        Returns:
            sum (SymbolicBinary): the sum of terms

        Raises:
            TypeError: Cannot add invalid type.
        """
        if not isinstance(addend, (numpy.int64, numpy.int32, int,
                                   type(self))):
            raise TypeError(
                'Object of invalid type cannot add with ' +
                str(type(self)) + '.')
        return self + addend

    def __add__(self, addend):
        """ 
        Left addition of SymbolicBinary.
        Args:
            addend (int or SymbolicBinary): The operator or int to add.

        Returns:
            sum (SymbolicBinary): the sum of terms
        """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __pow__(self, exponent):
        """Exponentiate the SymbolicBinary.

        Args:
            exponent (int): The exponent with which to raise the operator.

        Returns:
            exponentiated (SymbolicBinary): Exponentiated symbolicBinary

        Raises:
            TypeError: Can only raise SymbolicBinary to non-negative
                integer powers.
        """
        # Handle invalid exponents.
        if not isinstance(exponent, (numpy.int64, numpy.int32, int)):
            raise TypeError(
                'exponent must be int, but was {} {}'.format(
                    type(exponent), repr(exponent)))
        else:
            if exponent < 0:
                raise TypeError(
                    'exponent must be non-negative, but was {}'.format(
                        exponent))

        # Check if exponent is zero - if yes return self, if not return zero.
        if exponent == 0:
            return self.identity()
        else:
            return self


def binary_sum_rule(terms, summand):
    """ Updates terms in place based on binary rules.
    Args:
        terms: symbolicBinary terms
        summand: new potential addition to term
    """
    if summand not in terms:
        terms.append(summand)
    else:
        terms.remove(summand)
