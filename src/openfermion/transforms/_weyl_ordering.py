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

"""Weyl ordering on bosonic operators."""
import itertools

import numpy
from scipy.special import binom


from openfermion.ops import (BosonOperator, QuadOperator)


def mccoy(mode, op_a, op_b, m, n):
    """ Implement the McCoy formula on two operators of the
    form op_a^m op_b^n.

    Args:
        mode (int): the mode number the two operators act on.
        op_a: the label of operator a. This can be any hashable type.
        op_b: the label of operator b. This can be any hashable type.
        m (int): the power of operator a.
        n (int): the power of operator b.
    """
    new_op = dict()
    for r in range(0, n+1):
        coeff = binom(n, r)/(2**n)
        new_term = tuple([(mode, op_b)]*r + [(mode, op_a)]*m \
                    + [(mode, op_b)]*(n-r))
        if new_term not in new_op:
            new_op[tuple(new_term)] = coeff
        else:
            new_op[tuple(new_term)] += coeff
    return new_op


def weyl_ordering(operator):
    """ Apply the Weyl ordering to a BosonOperator or QuadOperator.

    The Weyl ordering is performed via McCoy's formula:

    q^m p^n -> (1/ 2^n) sum_{r=0}^{n} Binomial(n, r) q^r p^m q^{n-r}

    Returns:
        transformed_operator: an operator of the same class as in the input.

    Warning:
        The runtime of this method is exponential in the maximum locality
        of the original operator.
    """
    if isinstance(operator, BosonOperator):
        transformed_operator = BosonOperator()
        for term in operator.terms:
            # Initialize identity matrix.
            transformed_term = BosonOperator((), operator.terms[term])

            # convert term into the form \prod_i {bd_i^m b_i^n}
            modes = dict()
            for op in term:
                if op[0] not in modes:
                    modes[op[0]] = [0, 0]

                modes[op[0]][1-op[1]] += 1

            # Replace {bd_i^m b_i^n} -> S({bd_i^m b_i^n})
            for mode, (m, n) in modes.items():
                qtmp = BosonOperator()
                qtmp.terms = mccoy(mode, 1, 0, m, n)
                transformed_term *= qtmp

        if operator.terms:
            transformed_operator += transformed_term

    elif isinstance(operator, QuadOperator):
        transformed_operator = QuadOperator()
        for term in operator.terms:
            # Initialize identity matrix.
            transformed_term = QuadOperator((), operator.terms[term])

            # convert term into the form \prod_i {q_i^m p_i^n}
            modes = dict()
            for op in term:
                if op[0] not in modes:
                    modes[op[0]] = [0, 0]

                if op[1] == 'q':
                    modes[op[0]][0] += 1
                elif op[1] == 'p':
                    modes[op[0]][1] += 1

            # replace {q_i^m p_i^n} -> S({q_i^m p_i^n})
            for mode, (m, n) in modes.items():
                qtmp = QuadOperator()
                qtmp.terms = mccoy(mode, 'q', 'p', m, n)
                transformed_term *= qtmp

        if operator.terms:
            transformed_operator += transformed_term

    else:
        raise TypeError("operator must be a BosonOperator or "
                        "QuadOperator.")

    return transformed_operator
