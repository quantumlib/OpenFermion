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

"""Faster commutators for two-body operators with diagonal Coulomb terms."""

import warnings

from openfermion import FermionOperator, normal_ordered


def commutator_diagonal_coulomb_hamiltonians(
        operator_a, operator_b, result=None):
    """Compute the commutator of two-body operators with only diagonal
    Coulomb interactions.

    Args:
        operator_a, operator_b: FermionOperators to compute the commutator of.
            All terms in both operators must be normal-ordered, and further all
            terms must be either hopping operators (i^ j) or diagonal
            Coulomb operators (i^ i or i^ j^ i j).
        result (optional): The initial FermionOperator to add to.

    Returns:
        result: A FermionOperator with the commutator added to it.

    Notes:
        This function is actually slightly more general than promised, in that
        it works for *any* two-body operator_b, provided that operator_a has
        the required form. Similarly, if operator_a is an arbitrary two-body
        operator and operator_b is a 1-body operator, it will also work.
        All operators must be normal-ordered regardless.

        The function could be readily extended to the case of arbitrary
        two-body operator_a given that operator_b has the desired form;
        however, the extra check slows it down without desirable added utility.
    """
    if result is None:
        result = FermionOperator.zero()

    for term_a in operator_a.terms:
        coeff_a = operator_a.terms[term_a]
        for term_b in operator_b.terms:
            coeff_b = operator_b.terms[term_b]

            coefficient = coeff_a * coeff_b

            # If term_a == term_b, the terms commute; nothing to add.
            if term_a == term_b:
                continue

            # Case 1: both operators are two-body, operator_a is i^ j^ i j.
            if (len(term_a) == len(term_b) == 4 and
                    term_a[0][0] == term_a[2][0] and
                    term_a[1][0] == term_a[3][0]):
                b_create = (term_b[0][0], term_b[1][0])
                b_annihilate = (term_b[2][0], term_b[3][0])

                if (term_a[2][0] == term_b[0][0] and
                        term_a[3][0] == term_b[1][0]):
                    result.terms[term_b] = (
                        result.terms.get(term_b, 0.0) - coefficient)
                elif (term_a[0][0] == term_b[2][0] and
                      term_a[1][0] == term_b[3][0]):
                    result.terms[term_b] = (
                        result.terms.get(term_b, 0.0) + coefficient)

                elif term_a[0][0] in b_annihilate:
                    if term_a[1][0] in b_create or term_a[0][0] in b_create:
                        continue
                    new_coeff = coefficient
                    new_term = list(term_b)
                    new_term.insert(2, (term_a[1][0], 1))
                    new_term.append((term_a[1][0], 0))

                    # Normal order if j is less than term_a's creation parts.
                    if new_term[1][0] < new_term[2][0]:
                        new_term[1], new_term[2] = new_term[2], new_term[1]
                        new_coeff *= -1
                        if new_term[0][0] < new_term[1][0]:
                            new_term[0], new_term[1] = new_term[1], new_term[0]
                            new_coeff *= -1

                    # Normal order if j < term_a's second annihilation part.
                    if new_term[4][0] < new_term[5][0]:
                        new_term[4], new_term[5] = new_term[5], new_term[4]
                        new_coeff *= -1

                    # Add the new normal-ordered term to the result.
                    result.terms[tuple(new_term)] = (
                        result.terms.get(tuple(new_term), 0.0) + new_coeff)

                elif term_a[1][0] in b_annihilate:
                    if term_a[0][0] in b_create or term_a[1][0] in b_create:
                        continue
                    new_coeff = coefficient
                    new_term = list(term_b)
                    new_term.insert(0, (term_a[0][0], 1))
                    new_term.insert(3, (term_a[0][0], 0))

                    # Normal order if i is less than term_b's creation parts.
                    if new_term[0][0] < new_term[1][0]:
                        new_term[0], new_term[1] = new_term[1], new_term[0]
                        new_coeff *= -1
                        if new_term[1][0] < new_term[2][0]:
                            new_term[1], new_term[2] = new_term[2], new_term[1]
                            new_coeff *= -1

                    # Normal order if i < term_b's second annihilation part.
                    if new_term[3][0] < new_term[4][0]:
                        new_term[3], new_term[4] = new_term[4], new_term[3]
                        new_coeff *= -1

                    # Add the new normal-ordered term to the result.
                    result.terms[tuple(new_term)] = (
                        result.terms.get(tuple(new_term), 0.0) + new_coeff)

                elif term_a[0][0] in b_create:
                    new_coeff = -coefficient
                    new_term = list(term_b)
                    new_term.insert(0, (term_a[1][0], 1))
                    new_term.insert(3, (term_a[1][0], 0))

                    # Normal order if j is less than term_b's creation parts.
                    if new_term[0][0] < new_term[1][0]:
                        new_term[0], new_term[1] = new_term[1], new_term[0]
                        new_coeff *= -1
                        if new_term[1][0] < new_term[2][0]:
                            new_term[1], new_term[2] = new_term[2], new_term[1]
                            new_coeff *= -1

                    # Normal order if j < term_b's annihilation parts.
                    if new_term[3][0] < new_term[4][0]:
                        new_term[3], new_term[4] = new_term[4], new_term[3]
                        new_coeff *= -1
                        if new_term[4][0] < new_term[5][0]:
                            new_term[4], new_term[5] = new_term[5], new_term[4]
                            new_coeff *= -1

                    # Add the new normal-ordered term to the result.
                    result.terms[tuple(new_term)] = (
                        result.terms.get(tuple(new_term), 0.0) + new_coeff)

                elif term_a[1][0] in b_create:
                    new_coeff = -coefficient
                    new_term = list(term_b)
                    new_term.insert(0, (term_a[0][0], 1))
                    new_term.insert(3, (term_a[0][0], 0))

                    # Normal order if i is less than term_b's creation part.
                    if new_term[0][0] < new_term[1][0]:
                        new_term[0], new_term[1] = new_term[1], new_term[0]
                        new_coeff *= -1

                    # Normal order if i < term_b's annihilation parts.
                    if new_term[3][0] < new_term[4][0]:
                        new_term[3], new_term[4] = new_term[4], new_term[3]
                        new_coeff *= -1
                        if new_term[4][0] < new_term[5][0]:
                            new_term[4], new_term[5] = new_term[5], new_term[4]
                            new_coeff *= -1

                    # Add the new normal-ordered term to the result.
                    result.terms[tuple(new_term)] = (
                        result.terms.get(tuple(new_term), 0.0) + new_coeff)

            # Case 2: commutator of a 1-body and a 2-body operator
            elif (len(term_b) == 4 and len(term_a) == 2) or (
                    len(term_a) == 4 and len(term_b) == 2):

                # Standardize to a being the 1-body operator (and b 2-body)
                if len(term_a) == 4 and len(term_b) == 2:
                    a_create = term_b[0][0]
                    a_annihilate = term_b[1][0]
                    b_create = (term_a[0][0], term_a[1][0])
                    b_annihilate = (term_a[2][0], term_a[3][0])
                    new_term = list(term_a)
                    coefficient *= -1
                else:
                    a_create = term_a[0][0]
                    a_annihilate = term_a[1][0]
                    b_create = (term_b[0][0], term_b[1][0])
                    b_annihilate = (term_b[2][0], term_b[3][0])
                    new_term = list(term_b)

                # If both terms are composed of number operators, they commute.
                if a_create == a_annihilate and b_create == b_annihilate:
                    continue

                # If the annihilation part of a is in the creation parts of b
                if a_annihilate in b_create:
                    new_coeff = coefficient
                    new_inner_term = list(new_term)

                    # Determine which creation part(s) of a to substitute in
                    if a_annihilate == b_create[0]:
                        new_inner_term[0] = (a_create, 1)
                    elif a_annihilate == b_create[1]:
                        new_inner_term[1] = (a_create, 1)

                    # Normal order if necessary
                    if new_inner_term[0][0] < new_inner_term[1][0]:
                        new_inner_term[0], new_inner_term[1] = (
                            new_inner_term[1], new_inner_term[0])
                        new_coeff *= -1

                    if new_inner_term[0][0] > new_inner_term[1][0]:
                        result.terms[tuple(new_inner_term)] = (
                            result.terms.get(tuple(new_inner_term), 0.0) +
                            new_coeff)

                # If the creation part of a is in the annihilation parts of b
                if a_create in b_annihilate:
                    new_coeff = -coefficient

                    # Determine which annihilation part(s) of a to sub in
                    if a_create == b_annihilate[0]:
                        new_term[2] = (a_annihilate, 0)
                    elif a_create == b_annihilate[1]:
                        new_term[3] = (a_annihilate, 0)

                    # Normal order if necessary
                    if new_term[2][0] < new_term[3][0]:
                        new_term[2], new_term[3] = new_term[3], new_term[2]
                        new_coeff *= -1

                    if new_term[2][0] > new_term[3][0]:
                        result.terms[tuple(new_term)] = (
                            result.terms.get(tuple(new_term), 0.0) + new_coeff)

            # Case 3: both terms are one-body operators (both length 2)
            elif len(term_a) == 2 and len(term_b) == 2:
                if (term_a[0][0] == term_b[1][0] and
                        term_b[0][0] == term_a[1][0]):
                    new_term_a = ((term_a[0][0], 1), (term_a[0][0], 0))
                    new_term_b = ((term_b[0][0], 1), (term_b[0][0], 0))

                    result.terms[new_term_a] = (
                        result.terms.get(new_term_a, 0.0) + coefficient)
                    result.terms[new_term_b] = (
                        result.terms.get(new_term_b, 0.0) - coefficient)

                elif term_a[1][0] == term_b[0][0]:
                    term_ab = ((term_a[0][0], 1), (term_b[1][0], 0))

                    result.terms[term_ab] = (
                        result.terms.get(term_ab, 0.0) + coefficient)

                elif term_a[0][0] == term_b[1][0]:
                    term_ba = ((term_b[0][0], 1), (term_a[1][0], 0))

                    result.terms[term_ba] = (
                        result.terms.get(term_ba, 0.0) - coefficient)

            # Final case (case 4): violation of the input promise. Still
            # compute the commutator, but warn the user.
            else:
                warnings.warn('Defaulted to standard commutator evaluation '
                              'due to an out-of-spec operator.')
                additional = FermionOperator.zero()
                additional.terms[term_a + term_b] = coefficient
                additional.terms[term_b + term_a] = -coefficient
                additional = normal_ordered(additional)

                result += additional

    return result
