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

"""The MajoranaOperator data structure."""

import itertools
import numpy


class MajoranaOperator:
    r"""A linear combination of products of Majorana operators.

    A system of N fermionic modes can be described using 2N Majorana operators
    :math:`\gamma_1, \ldots, \gamma_{2N}`
    as an alternative to using N fermionic annihilation operators. The algebra
    of Majorana operators amounts to the relation

    .. math::
        \{\gamma_i, \gamma_j\}
         = \gamma_i \gamma_j + \gamma_j \gamma_i
         = 2 \delta_{ij}

    Note that this implies :math:`\gamma_i^2 = 1`.

    The MajoranaOperator class stores a linear combination of products
    of Majorana operators. Each product is represented as a tuple of
    integers representing the indices of the operators. As an example,
    `MajoranaOperator((2, 3, 5), -1.5)` initializes an operator with
    a single term which represents the operator
    :math:`-1.5 \gamma_2 \gamma_3 \gamma_5`. MajoranaOperators can be
    added, subtracted, multiplied, and divided by scalars. They can be
    compared for approximate numerical equality using `==`.

    Attributes:
        terms: A dictionary from term, represented by a tuple of integers,
        to the coefficient of the term in the linear combination.
    """

    def __init__(self, term=None, coefficient=1.0):
        """Initialize a MajoranaOperator with a single term.

        Args:
            term (Tuple[int]): The indices of a Majorana operator term
                to start off with
            coefficient (complex): The coefficient of the term

        Returns:
            MajoranaOperator
        """
        self.terms = {}
        if term is not None:
            term, parity = _sort_majorana_term(term)
            self.terms[term] = coefficient * (-1)**parity

    @staticmethod
    def from_dict(terms):
        """Initialize a MajoranaOperator from a terms dictionary.

        WARNING: The given dictionary is not validated whatsoever. It's up
        to you to ensure that it is properly formed.

        Args:
            terms: A dictionary from Majorana term to coefficient
        """
        op = MajoranaOperator()
        op.terms = terms
        return op

    def commutes_with(self, other):
        """Test commutation with another MajoranaOperator"""
        if not isinstance(other, type(self)):
            raise TypeError(
                'Can only test commutation with another MajoranaOperator.')

        if len(self.terms) == 1 and len(other.terms) == 1:
            return _majorana_terms_commute(list(self.terms.keys())[0],
                                           list(other.terms.keys())[0])
        return self*other == other*self

    def with_basis_rotated_by(self, transformation_matrix):
        r"""Change to a basis of new Majorana operators.

        The input to this method is a real orthogonal matrix :math:`O`.
        It returns a new MajoranaOperator which is equivalent to the old one
        but rewritten in terms of a new basis of Majorana operators.
        Let the original Majorana operators be denoted by
        :math:`\gamma_i` and the new operators be denoted by
        :math:`\tilde{\gamma_i}`. Then they are related by the equation

        .. math::

            \tilde{\gamma_i} = \sum_j O_{ij} \gamma_j.

        Args:
            transformation_matrix: A real orthogonal matrix representing
                the basis transformation.
        Returns:
            The rotated operator.
        """
        if not _is_real_orthogonal(transformation_matrix):
            raise ValueError("Transformation matrix is not real orthogonal.")

        rotated_op = MajoranaOperator()
        for term, coeff in self.terms.items():
            rotated_term = _rotate_basis(term, transformation_matrix)
            rotated_term *= coeff
            rotated_op += rotated_term
        return rotated_op

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        for term, coefficient in other.terms.items():
            if term in self.terms:
                self.terms[term] += coefficient
            else:
                self.terms[term] = coefficient

        return self

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        terms = {}
        terms.update(self.terms)

        for term, coefficient in other.terms.items():
            if term in terms:
                terms[term] += coefficient
            else:
                terms[term] = coefficient

        return MajoranaOperator.from_dict(terms)

    def __isub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        for term, coefficient in other.terms.items():
            if term in self.terms:
                self.terms[term] -= coefficient
            else:
                self.terms[term] = coefficient

        return self

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        terms = {}
        terms.update(self.terms)
        for term, coefficient in other.terms.items():
            if term in terms:
                terms[term] -= coefficient
            else:
                terms[term] = -coefficient
        return MajoranaOperator.from_dict(terms)

    def __mul__(self, other):
        if not isinstance(other, (type(self), int, float, complex)):
            return NotImplemented

        if isinstance(other, (int, float, complex)):
            terms = {term: coefficient*other
                     for term, coefficient in self.terms.items()}
            return MajoranaOperator.from_dict(terms)

        terms = {}
        for left_term, left_coefficient in self.terms.items():
            for right_term, right_coefficient in other.terms.items():
                new_term, parity = _merge_majorana_terms(left_term, right_term)
                coefficient = left_coefficient*right_coefficient*(-1)**parity
                if new_term in terms:
                    terms[new_term] += coefficient
                else:
                    terms[new_term] = coefficient
        return MajoranaOperator.from_dict(terms)

    def __imul__(self, other):
        if not isinstance(other, (type(self), int, float, complex)):
            return NotImplemented

        if isinstance(other, (int, float, complex)):
            for term in self.terms:
                self.terms[term] *= other
            return self

        return self * other

    def __rmul__(self, other):
        if not isinstance(other, (int, float, complex)):
            return NotImplemented
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, (int, float, complex)):
            return NotImplemented

        terms = {term: coefficient/other
                 for term, coefficient in self.terms.items()}
        return MajoranaOperator.from_dict(terms)

    def __itruediv__(self, other):
        if not isinstance(other, (int, float, complex)):
            return NotImplemented

        for term in self.terms:
            self.terms[term] /= other
        return self

    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)

    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        if other < 0:
            raise TypeError('Cannot raise MajoranaOperator to negative power.')

        result = MajoranaOperator(())
        for _ in range(other):
            result *= self
        return result

    def __neg__(self):
        return -1 * self

    def __eq__(self, other):
        """Approximate numerical equality."""
        if not isinstance(other, type(self)):
            return NotImplemented

        for term in self.terms.keys() | other.terms.keys():
            if term in self.terms and term in other.terms:
                if not numpy.isclose(self.terms[term], other.terms[term]):
                    return False
            elif term in self.terms:
                if not numpy.isclose(self.terms[term], 0.0):
                    return False
            else:
                if not numpy.isclose(other.terms[term], 0.0):
                    return False
        return True

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        if not self.terms:
            return '0'
        lines = []
        for term, coeff in sorted(self.terms.items()):
            if numpy.isclose(coeff, 0.0):
                continue
            lines.append('{} {} +'.format(coeff, term))
        if not lines:
            return '0'
        return '\n'.join(lines)[:-2]

    def __repr__(self):
        return 'MajoranaOperator.from_dict(terms={!r})'.format(self.terms)


def _sort_majorana_term(term):
    """Sort a Majorana term.

    Args:
        term (Tuple[int]): The indices of a Majorana operator term

    Returns:
        Tuple[Tuple[int], int]. The first object returned is a sorted list
        representing the indices acted upon. The second object is the parity
        of the term. A parity of 1 indicates that the term should include
        a minus sign.
    """
    if len(term) < 2:
        return term, 0
    center = len(term) // 2
    left_term, left_parity = _sort_majorana_term(term[:center])
    right_term, right_parity = _sort_majorana_term(term[center:])
    merged_term, merge_parity = _merge_majorana_terms(left_term, right_term)
    return merged_term, (left_parity + right_parity + merge_parity) % 2


def _merge_majorana_terms(left_term, right_term):
    """Merge two Majorana terms.

    Args:
        left_term (Tuple[int]): The left-hand term
        right_term (Tuple[int]): The right-hand term

    Returns:
        Tuple[Tuple[int], int]. The first object returned is a sorted list
        representing the indices acted upon. The second object is the parity
        of the term. A parity of 1 indicates that the term should include
        a minus sign.
    """
    merged_term = []
    parity = 0
    i, j = 0, 0
    while i < len(left_term) and j < len(right_term):
        if left_term[i] < right_term[j]:
            merged_term.append(left_term[i])
            i += 1
        elif left_term[i] > right_term[j]:
            merged_term.append(right_term[j])
            j += 1
            parity += len(left_term) - i
        else:
            parity += len(left_term) - i - 1
            i += 1
            j += 1
    if i == len(left_term):
        merged_term.extend(right_term[j:])
    else:
        merged_term.extend(left_term[i:])
    return tuple(merged_term), parity % 2


def _majorana_terms_commute(term_a, term_b):
    """Whether two Majorana terms commute.

    Args:
        term_a (Tuple[int]): The indices of a Majorana operator term
        term_b (Tuple[int]): The indices of a Majorana operator term

    Returns:
        bool. Whether The terms commute.
    """
    intersection = 0
    i, j = 0, 0
    while i < len(term_a) and j < len(term_b):
        if term_a[i] < term_b[j]:
            i += 1
        elif term_a[i] > term_b[j]:
            j += 1
        else:
            intersection += 1
            i += 1
            j += 1
    parity = (len(term_a)*len(term_b) - intersection) % 2
    return not parity


def _rotate_basis(term, transformation_matrix):
    n = transformation_matrix.shape[0]
    rotated_op = MajoranaOperator()
    for tup in itertools.product(range(n), repeat=len(term)):
        coeff = 1.0
        for i, j in zip(term, tup):
            coeff *= transformation_matrix[j, i]
        rotated_op += MajoranaOperator(tup, coeff)
    return rotated_op


def _is_real_orthogonal(matrix):
    n, m = matrix.shape
    return (n == m
            and numpy.allclose(numpy.imag(matrix), 0.0)
            and numpy.allclose(numpy.dot(matrix.T, matrix), numpy.eye(n)))
