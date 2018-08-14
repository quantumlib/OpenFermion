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


def commutator_ordered_diagonal_coulomb_with_two_body_operator(
        operator_a, operator_b, prior_terms=None):
    """Compute the commutator of two-body operators provided that both are
    normal-ordered and that the first only has diagonal Coulomb interactions.

    Args:
        operator_a: The first FermionOperator argument of the commutator.
            All terms must be normal-ordered, and furthermore either hopping
            operators (i^ j) or diagonal Coulomb operators (i^ i or i^ j^ i j).
        operator_b: The second FermionOperator argument of the commutator.
            operator_b can be any arbitrary two-body operator.
        prior_terms (optional): The initial FermionOperator to add to.

    Returns:
        The commutator, or the commutator added to prior_terms if provided.

    Notes:
        The function could be readily extended to the case of arbitrary
        two-body operator_a given that operator_b has the desired form;
        however, the extra check slows it down without desirable added utility.
    """
    if prior_terms is None:
        prior_terms = FermionOperator.zero()

    for term_a in operator_a.terms:
        coeff_a = operator_a.terms[term_a]
        for term_b in operator_b.terms:
            coeff_b = operator_b.terms[term_b]

            coefficient = coeff_a * coeff_b

            # If term_a == term_b the terms commute, nothing to add.
            if term_a == term_b or not term_a or not term_b:
                continue

            # Case 1: both operators are two-body, operator_a is i^ j^ i j.
            if (len(term_a) == len(term_b) == 4 and
                    term_a[0][0] == term_a[2][0] and
                    term_a[1][0] == term_a[3][0]):
                _commutator_two_body_diagonal_with_two_body(
                    term_a, term_b, coefficient, prior_terms)

            # Case 2: commutator of a 1-body and a 2-body operator
            elif (len(term_b) == 4 and len(term_a) == 2) or (
                    len(term_a) == 4 and len(term_b) == 2):
                _commutator_one_body_with_two_body(
                    term_a, term_b, coefficient, prior_terms)

            # Case 3: both terms are one-body operators (both length 2)
            elif len(term_a) == 2 and len(term_b) == 2:
                _commutator_one_body_with_one_body(
                    term_a, term_b, coefficient, prior_terms)

            # Final case (case 4): violation of the input promise. Still
            # compute the commutator, but warn the user.
            else:
                warnings.warn('Defaulted to standard commutator evaluation '
                              'due to an out-of-spec operator.')
                additional = FermionOperator.zero()
                additional.terms[term_a + term_b] = coefficient
                additional.terms[term_b + term_a] = -coefficient
                additional = normal_ordered(additional)

                prior_terms += additional

    return prior_terms


def _commutator_one_body_with_one_body(one_body_action_a, one_body_action_b,
                                       coefficient, prior_terms):
    """Compute the commutator of two one-body operators specified by actions.

    Args:
        one_body_action_a, one_body_action_b (tuple): single terms of one-body
            FermionOperators (i^ j or i^ i).
        coefficient (complex float): coefficient of the commutator.
        prior_terms (FermionOperator): prior terms to add the commutator to.
    """
    # In the case that both the creation and annihilation operators of the
    # two actions pair, two new terms must be added.
    if (one_body_action_a[0][0] == one_body_action_b[1][0] and
            one_body_action_b[0][0] == one_body_action_a[1][0]):
        new_one_body_action_a = ((one_body_action_a[0][0], 1),
                                 (one_body_action_a[0][0], 0))
        new_one_body_action_b = ((one_body_action_b[0][0], 1),
                                 (one_body_action_b[0][0], 0))

        prior_terms.terms[new_one_body_action_a] = (
            prior_terms.terms.get(new_one_body_action_a, 0.0) + coefficient)
        prior_terms.terms[new_one_body_action_b] = (
            prior_terms.terms.get(new_one_body_action_b, 0.0) - coefficient)

    # A single pairing causes the mixed action a[0]^ b[1] to be added
    elif one_body_action_a[1][0] == one_body_action_b[0][0]:
        action_ab = ((one_body_action_a[0][0], 1),
                     (one_body_action_b[1][0], 0))

        prior_terms.terms[action_ab] = (
            prior_terms.terms.get(action_ab, 0.0) + coefficient)

    # The other single pairing adds the mixed action b[0]^ a[1]
    elif one_body_action_a[0][0] == one_body_action_b[1][0]:
        action_ba = ((one_body_action_b[0][0], 1),
                     (one_body_action_a[1][0], 0))

        prior_terms.terms[action_ba] = (
            prior_terms.terms.get(action_ba, 0.0) - coefficient)


def _commutator_one_body_with_two_body(action_a, action_b,
                                       coefficient, prior_terms):
    """Compute commutator of action-specified one- and two-body operators.

    Args:
        action_a, action_b (tuple): single terms, one one-body and the other
            two-body, from normal-ordered FermionOperators. It does not matter
            which is one- or two-body so long as only one of each appears.
        coefficient (complex float): coefficient of the commutator.
        prior_terms (FermionOperator): prior terms to add the commutator to.
    """
    # Determine which action is 1-body and which is 2-body.
    # Label the creation and annihilation parts of the two terms.
    if len(action_a) == 4 and len(action_b) == 2:
        one_body_create = action_b[0][0]
        one_body_annihilate = action_b[1][0]
        two_body_create = (action_a[0][0], action_a[1][0])
        two_body_annihilate = (action_a[2][0], action_a[3][0])
        new_action = list(action_a)
        # Flip coefficient because we reversed the commutator's arguments.
        coefficient *= -1
    else:
        one_body_create = action_a[0][0]
        one_body_annihilate = action_a[1][0]
        two_body_create = (action_b[0][0], action_b[1][0])
        two_body_annihilate = (action_b[2][0], action_b[3][0])
        new_action = list(action_b)

    # If both terms are composed of number operators, they commute.
    if one_body_create == one_body_annihilate and (
            two_body_create == two_body_annihilate):
        return

    # If the one-body annihilation is in the two-body creation parts
    if one_body_annihilate in two_body_create:
        new_coeff = coefficient
        new_inner_action = list(new_action)

        # Determine which creation part(s) of the one-body action to use
        if one_body_annihilate == two_body_create[0]:
            new_inner_action[0] = (one_body_create, 1)
        elif one_body_annihilate == two_body_create[1]:
            new_inner_action[1] = (one_body_create, 1)

        # Normal order if necessary
        if new_inner_action[0][0] < new_inner_action[1][0]:
            new_inner_action[0], new_inner_action[1] = (
                new_inner_action[1], new_inner_action[0])
            new_coeff *= -1

        # Add the resulting term.
        if new_inner_action[0][0] > new_inner_action[1][0]:
            prior_terms.terms[tuple(new_inner_action)] = (
                prior_terms.terms.get(tuple(new_inner_action), 0.0) +
                new_coeff)

    # If the one-body creation is in the two-body annihilation parts
    if one_body_create in two_body_annihilate:
        new_coeff = -coefficient

        # Determine which annihilation part(s) of the one-body action to sub in
        if one_body_create == two_body_annihilate[0]:
            new_action[2] = (one_body_annihilate, 0)
        elif one_body_create == two_body_annihilate[1]:
            new_action[3] = (one_body_annihilate, 0)

        # Normal order if necessary
        if new_action[2][0] < new_action[3][0]:
            new_action[2], new_action[3] = new_action[3], new_action[2]
            new_coeff *= -1

        # Add the resulting term.
        if new_action[2][0] > new_action[3][0]:
            prior_terms.terms[tuple(new_action)] = (
                prior_terms.terms.get(tuple(new_action), 0.0) + new_coeff)


def _commutator_two_body_diagonal_with_two_body(
        diagonal_coulomb_action, arbitrary_two_body_action,
        coefficient, prior_terms):
    """Compute the commutator of two two-body operators specified by actions.

    Args:
        diagonal_coulomb_action (tuple): single term of a diagonal Coulomb
            FermionOperator (i^ j^ i j). Must be in normal-ordered form,
            i.e. i > j.
        arbitrary_two_body_action (tuple): arbitrary single term of a two-body
            FermionOperator, in normal-ordered form, i.e. i^ j^ k l with
            i > j, k > l.
        coefficient (complex float): coefficient of the commutator.
        prior_terms (FermionOperator): prior terms to add the commutator to.

    Notes:
        The function could be readily extended to the case of reversed input
        arguments (where diagonal_coulomb_action is the arbitrary one, and
        arbitrary_two_body_action is from a diagonal Coulomb FermionOperator);
        however, the extra check slows it down without significantly increased
        utility.
    """
    # Identify creation and annihilation parts of arbitrary_two_body_action.
    arb_2bdy_create = (arbitrary_two_body_action[0][0],
                       arbitrary_two_body_action[1][0])
    arb_2bdy_annihilate = (arbitrary_two_body_action[2][0],
                           arbitrary_two_body_action[3][0])

    # The first two sub-cases cover when the creations and annihilations of
    # diagonal_coulomb_action and arbitrary_two_body_action totally pair up.
    if (diagonal_coulomb_action[2][0] == arbitrary_two_body_action[0][0] and
            diagonal_coulomb_action[3][0] == arbitrary_two_body_action[1][0]):
        prior_terms.terms[arbitrary_two_body_action] = (
            prior_terms.terms.get(arbitrary_two_body_action, 0.0) -
            coefficient)

    elif (diagonal_coulomb_action[0][0] == arbitrary_two_body_action[2][0] and
          diagonal_coulomb_action[1][0] == arbitrary_two_body_action[3][0]):
        prior_terms.terms[arbitrary_two_body_action] = (
            prior_terms.terms.get(arbitrary_two_body_action, 0.0) +
            coefficient)

    # Exactly one of diagonal_coulomb_action's creations matches one of
    # arbitrary_two_body_action's annihilations.
    elif diagonal_coulomb_action[0][0] in arb_2bdy_annihilate:
        # Nothing gets added if there's an unbalanced double creation.
        if diagonal_coulomb_action[1][0] in arb_2bdy_create or (
                diagonal_coulomb_action[0][0] in arb_2bdy_create):
            return

        _add_three_body_term(
            arbitrary_two_body_action, coefficient,
            diagonal_coulomb_action[1][0], prior_terms)

    elif diagonal_coulomb_action[1][0] in arb_2bdy_annihilate:
        # Nothing gets added if there's an unbalanced double creation.
        if diagonal_coulomb_action[0][0] in arb_2bdy_create or (
                diagonal_coulomb_action[1][0] in arb_2bdy_create):
            return

        _add_three_body_term(arbitrary_two_body_action, coefficient,
                             diagonal_coulomb_action[0][0], prior_terms)

    elif diagonal_coulomb_action[0][0] in arb_2bdy_create:
        _add_three_body_term(arbitrary_two_body_action, -coefficient,
                             diagonal_coulomb_action[1][0], prior_terms)

    elif diagonal_coulomb_action[1][0] in arb_2bdy_create:
        _add_three_body_term(arbitrary_two_body_action, -coefficient,
                             diagonal_coulomb_action[0][0], prior_terms)


def _add_three_body_term(two_body_action, coefficient, mode, prior_terms):
    new_action = list(two_body_action)

    # Insert creation and annihilation operators into the two-body action.
    new_action.insert(0, (mode, 1))
    new_action.insert(3, (mode, 0))

    # Normal order the creation operators of the new action.
    # Each exchange in the action flips the sign of the coefficient.
    if new_action[0][0] < new_action[1][0]:
        new_action[0], new_action[1] = new_action[1], new_action[0]
        coefficient *= -1
        if new_action[1][0] < new_action[2][0]:
            new_action[1], new_action[2] = new_action[2], new_action[1]
            coefficient *= -1

    # Normal order the annihilation operators of the new action.
    # Each exchange in the action flips the sign of the coefficient.
    if new_action[3][0] < new_action[4][0]:
        new_action[3], new_action[4] = new_action[4], new_action[3]
        coefficient *= -1
        if new_action[4][0] < new_action[5][0]:
            new_action[4], new_action[5] = new_action[5], new_action[4]
            coefficient *= -1

    # Add the new normal-ordered term to the prior terms.
    prior_terms.terms[tuple(new_action)] = (
        prior_terms.terms.get(tuple(new_action), 0.0) + coefficient)
