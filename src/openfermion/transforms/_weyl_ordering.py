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
        new_term = tuple([(mode, op_b)]*r + [(mode, op_a)]*m
                         + [(mode, op_b)]*(n-r))
        if new_term not in new_op:
            new_op[tuple(new_term)] = coeff
        else:
            new_op[tuple(new_term)] += coeff
    return new_op


def weyl_polynomial_quantization(polynomial):
    r""" Apply the Weyl quantization to a phase space polynomial.

    The Weyl quantization is performed by applying McCoy's formula
    directly to a polynomial term of the form q^m p^n:

    q^m p^n ->
        (1/ 2^n) sum_{r=0}^{n} Binomial(n, r) \hat{q}^r \hat{p}^m q^{n-r}

    where q and p are phase space variables, and \hat{q} and \hat{p}
    are quadrature operators.

    The input is provided in the form of a string, for example

    .. code-block:: python

        weyl_polynomial_quantization('q0^2 p0^3 q1^3')

    where 'q' or 'p' is the phase space quadrature variable, the integer
    directly following is the mode it is with respect to, and '^2' is the
    polynomial power.

    Args:
        polynomial (str): polynomial function of q and p of the form
            'qi^m pj^n ...' where i,j are the modes, and m, n the powers.

    Returns:
        QuadOperator: the Weyl quantization of the phase space function.

    Warning:
        The runtime of this method is exponential in the maximum locality
        of the original operator.
    """
    # construct the equivalent QuadOperator
    poly = dict()

    if polynomial:
        for term in polynomial.split():
            if '^' in term:
                op, pwr = term.split('^')
                pwr = int(pwr)
            else:
                op = term
                pwr = 1

            mode = int(op[1:])

            if mode not in poly:
                poly[mode] = [0, 0]

            if op[0] == 'q':
                poly[mode][0] += pwr
            elif op[0] == 'p':
                poly[mode][1] += pwr

        # replace {q_i^m p_i^n} -> S({q_i^m p_i^n})
        operator = QuadOperator('')
        for mode, (m, n) in poly.items():
            qtmp = QuadOperator()
            qtmp.terms = mccoy(mode, 'q', 'p', m, n)
            operator *= qtmp
    else:
        operator = QuadOperator.zero()

    return operator


def symmetric_ordering(operator, ignore_coeff=True, ignore_identity=True):
    """ Apply the symmetric ordering to a BosonOperator or QuadOperator.

    The symmetric ordering is performed by applying McCoy's formula
    directly to polynomial terms of quadrature operators:

    q^m p^n -> (1/ 2^n) sum_{r=0}^{n} Binomial(n, r) q^r p^m q^{n-r}

    Note: in general, symmetric ordering is performed on a single term
    containing the tensor product of various operators. However, this
    function can also be applied to a sum of these terms, and the symmetric
    product is distributed over the summed terms.

    In this case, Hermiticity cannot be guaranteed - as such, by default
    term coefficients and identity operators are ignored. However, this
    behavior can be modified via keyword arguments describe below if necessary.

    Args:
        operator: either a BosonOperator or QuadOperator.
        ignore_coeff (bool): By default, the coefficients for
            each term are ignored; S(a q^m p^n) = S(q^m p^n), and
            the returned operator is always Hermitian.
            If set to False, then instead the coefficients are taken into
            account; S(q^m p^n) = a S(q^m p^n). In this case, if
            a is a complex coefficient, it is not guaranteed that the
            the returned operator will be Hermitian.
        ignore_identity (bool): By default, identity terms are ignore;
            S(I) = 0. If set to False, then instead S(I) = I.

    Returns:
        transformed_operator: an operator of the same class as in the input.

    Warning:
        The runtime of this method is exponential in the maximum locality
        of the original operator.
    """
    if isinstance(operator, BosonOperator):
        transformed_operator = BosonOperator()
        for term in operator.terms:
            if ignore_coeff:
                coeff = 1
            else:
                coeff = operator.terms[term]

            # Initialize identity matrix.
            transformed_term = BosonOperator('', coeff)

            if term:
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

            if term or (not ignore_identity):
                transformed_operator += transformed_term

    elif isinstance(operator, QuadOperator):
        transformed_operator = QuadOperator()
        for term in operator.terms:
            if ignore_coeff:
                coeff = 1
            else:
                coeff = operator.terms[term]

            # Initialize identity matrix.
            transformed_term = QuadOperator('', coeff)

            if term:
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

            if term or (not ignore_identity):
                transformed_operator += transformed_term

    else:
        raise TypeError("operator must be a BosonOperator or "
                        "QuadOperator.")

    return transformed_operator
