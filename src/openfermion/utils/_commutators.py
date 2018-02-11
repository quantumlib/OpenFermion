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

"""Module to compute commutators, with optimizations for specific systems."""
from __future__ import absolute_import
from future.utils import itervalues

import numpy

from openfermion.ops import FermionOperator, normal_ordered


def commutator(operator_a, operator_b):
    """Compute the commutator of two operators.

    Args:
        operator_a, operator_b: Operators in commutator. Any operators
            are accepted so long as implicit subtraction and multiplication are
            supported; e.g. QubitOperators, FermionOperators or Scipy sparse
            matrices. 2D Numpy arrays are also supported.

    Raises:
        TypeError: operator_a and operator_b are not of the same type.
    """
    if type(operator_a) != type(operator_b):
        raise TypeError('operator_a and operator_b are not of the same type.')
    if isinstance(operator_a, numpy.ndarray):
        result = operator_a.dot(operator_b)
        result -= operator_b.dot(operator_a)
    else:
        result = operator_a * operator_b
        result -= operator_b * operator_a
    return result


def anticommutator(operator_a, operator_b):
    """Compute the anticommutator of two operators.

    Args:
        operator_a, operator_b: Operators in anticommutator. Any operators
            are accepted so long as implicit addition and multiplication are
            supported; e.g. QubitOperators, FermionOperators or Scipy sparse
            matrices. 2D Numpy arrays are also supported.

    Raises:
        TypeError: operator_a and operator_b are not of the same type.
    """
    if type(operator_a) != type(operator_b):
        raise TypeError('operator_a and operator_b are not of the same type.')
    if isinstance(operator_a, numpy.ndarray):
        result = operator_a.dot(operator_b)
        result += operator_b.dot(operator_a)
    else:
        result = operator_a * operator_b
        result += operator_b * operator_a
    return result


def double_commutator(op1, op2, op3, indices2=None, indices3=None,
                      is_hopping_operator2=None, is_hopping_operator3=None):
    """Return the double commutator [op1, [op2, op3]].

    Args:
        op1, op2, op3 (FermionOperators): operators for the commutator.
        indices2, indices3 (set): The indices op2 and op3 act on.
        is_hopping_operator2 (bool): Whether op2 is a hopping operator.
        is_hopping_operator3 (bool): Whether op3 is a hopping operator.

    Returns:
        The double commutator of the given operators.
    """
    if is_hopping_operator2 and is_hopping_operator3:
        indices2 = set(indices2)
        indices3 = set(indices3)
        # Determine which indices both op2 and op3 act on.
        try:
            intersection, = indices2.intersection(indices3)
        except ValueError:
            return FermionOperator.zero()

        # Remove the intersection from the set of indices, since it will get
        # cancelled out in the final result.
        indices2.remove(intersection)
        indices3.remove(intersection)

        # Find the indices of the final output hopping operator.
        index2, = indices2
        index3, = indices3
        coeff2 = op2.terms[list(op2.terms)[0]]
        coeff3 = op3.terms[list(op3.terms)[0]]
        commutator23 = (
            FermionOperator(((index2, 1), (index3, 0)), coeff2 * coeff3) +
            FermionOperator(((index3, 1), (index2, 0)), -coeff2 * coeff3))
    else:
        commutator23 = normal_ordered(commutator(op2, op3))

    return normal_ordered(commutator(op1, commutator23))


def trivially_double_commutes_dual_basis_using_term_info(
    indices_alpha=None, indices_beta=None, indices_alpha_prime=None,
        is_hopping_operator_alpha=None, is_hopping_operator_beta=None,
        is_hopping_operator_alpha_prime=None, jellium_only=False):
    """Return whether [op_a, [op_b, op_a_prime]] is trivially zero.

    Assumes all the operators are FermionOperators from the dual basis
    Hamiltonian, broken into the form i^j^ i j + c_i*(i^ i) + c_j*(j^ j)
    or i^ j + j^ i, where i and j are modes and c is a constant. For the
    full dual basis Hamiltonian, i^ i and j^ j can have distinct
    coefficients c_i and c_j: for jellium they are necessarily the same.
    If this is the case, jellium_only should be set to True.

    The operators are determined by the indices they act on and by
    whether they are hopping operators (i^ j + j^ i) or number operators
    (i^ j^ i j + c_i*(i^ i) + c_j*(j^ j)). a, b, and a_prime are
    shorthands for alpha, beta, and alpha_prime.

    Args:
        indices_alpha (set): The indices term_alpha acts on.
        indices_beta (set): The indices term_beta acts on.
        indices_alpha_prime (set): The indices term_alpha_prime acts on.
        is_hopping_operator_alpha (bool): Whether term_alpha is a
                                          hopping operator.
        is_hopping_operator_beta (bool): Whether term_beta is a
                                         hopping operator.
        is_hopping_operator_alpha_prime (bool): Whether term_alpha_prime
                                                is a hopping operator.
        jellium_only (bool): Whether the terms are only from the jellium
                             Hamiltonian, i.e. if c_i = c for all number
                             operators i^ i or if it depends on i.

    Returns:
        Whether or not the double commutator is trivially zero.
    """
    # If operator_beta and operator_alpha_prime (in the inner commutator)
    # are number operators, they commute trivially.
    if not (is_hopping_operator_beta or is_hopping_operator_alpha_prime):
        return True

    # The operators in the jellium Hamiltonian (provided they are of the
    # form i^ i + j^ j or i^ j^ i j + c*(i^ i + j^ j), and not both
    # hopping operators) commute if they act on the same modes or if
    # there is no intersection.
    if (jellium_only and (not is_hopping_operator_alpha_prime or
                          not is_hopping_operator_beta) and
            len(indices_beta.intersection(indices_alpha_prime)) != 1):
        return True

    # If the modes operator_alpha acts on are disjoint with the modes
    # operator_beta and operator_alpha_prime act on, they commute.
    if not indices_alpha.intersection(indices_beta.union(indices_alpha_prime)):
        return True

    return False


def trivially_commutes_dual_basis(term_a, term_b):
    """Determine whether the given terms trivially commute.

    Assumes the terms are single-term FermionOperators from the
    plane-wave dual basis Hamiltonian.

    Args:
        term_a, term_b (FermionOperator): Single-term FermionOperators.

    Returns:
        Whether or not the commutator is trivially zero.
    """
    modes_acted_on_by_term_a, = term_a.terms.keys()
    modes_acted_on_by_term_b, = term_b.terms.keys()

    modes_touched_a = [modes_acted_on_by_term_a[0][0],
                       modes_acted_on_by_term_a[1][0]]
    modes_touched_b = [modes_acted_on_by_term_b[0][0],
                       modes_acted_on_by_term_b[1][0]]

    # If there's no intersection between the modes term_a and term_b act
    # on, the commutator is zero.
    if not (modes_touched_a[0] in modes_touched_b or
            modes_touched_a[1] in modes_touched_b):
        return True

    # In the dual basis, possible number operators take the form
    # a^ a or a^ b^ a b. Number operators always commute trivially.
    term_a_is_number_operator = (
        modes_acted_on_by_term_a[0][0] == modes_acted_on_by_term_a[1][0] or
        modes_acted_on_by_term_a[1][1])
    term_b_is_number_operator = (
        modes_acted_on_by_term_b[0][0] == modes_acted_on_by_term_b[1][0] or
        modes_acted_on_by_term_b[1][1])
    if term_a_is_number_operator and term_b_is_number_operator:
        return True

    # If the first commutator's terms are both hopping, and both create
    # or annihilate the same mode, then the result is zero.
    if not (term_a_is_number_operator or term_b_is_number_operator):
        if (modes_acted_on_by_term_a[0][0] == modes_acted_on_by_term_b[0][0] or
            modes_acted_on_by_term_a[1][0] ==
                modes_acted_on_by_term_b[1][0]):
            return True

    # If both terms act on the same operators and are not both hopping
    # operators, then they commute.
    if ((term_a_is_number_operator or term_b_is_number_operator) and
            set(modes_touched_a) == set(modes_touched_b)):
        return True

    return False


def trivially_double_commutes_dual_basis(term_a, term_b, term_c):
    """Check if the double commutator [term_a, [term_b, term_c]] is zero.

    Assumes the terms are single-term FermionOperators from the
    plane-wave dual basis Hamiltonian.

    Args:
        term_a, term_b, term_c: Single-term FermionOperators.

    Notes:
        This function inlines trivially_commutes_dual_basis for terms b
        and c.

    Returns:
        Whether or not the double commutator is trivially zero.
    """
    # Determine the set of modes each term acts on.
    modes_acted_on_by_term_b, = term_b.terms.keys()
    modes_acted_on_by_term_c, = term_c.terms.keys()

    modes_touched_c = [modes_acted_on_by_term_c[0][0],
                       modes_acted_on_by_term_c[1][0]]

    # If there's no intersection between the modes term_b and term_c act
    # on, the commutator is trivially zero.
    if not (modes_acted_on_by_term_b[0][0] in modes_touched_c or
            modes_acted_on_by_term_b[1][0] in modes_touched_c):
        return True

    # In the dual_basis Hamiltonian, possible number operators take the
    # form a^ a or a^ b^ a b. Check for this.
    term_b_is_number_operator = (
        modes_acted_on_by_term_b[0][0] == modes_acted_on_by_term_b[1][0] or
        modes_acted_on_by_term_b[1][1])
    term_c_is_number_operator = (
        modes_acted_on_by_term_c[0][0] == modes_acted_on_by_term_c[1][0] or
        modes_acted_on_by_term_c[1][1])

    # Number operators always commute.
    if term_b_is_number_operator and term_c_is_number_operator:
        return True

    # If the first commutator's terms are both hopping, and both create
    # or annihilate the same mode, then the result is zero.
    if not (term_b_is_number_operator or term_c_is_number_operator):
        if (modes_acted_on_by_term_b[0][0] == modes_acted_on_by_term_c[0][0] or
            modes_acted_on_by_term_b[1][0] ==
                modes_acted_on_by_term_c[1][0]):
            return True

    # The modes term_a acts on are only needed if we reach this stage.
    modes_acted_on_by_term_a, = term_a.terms.keys()
    modes_touched_b = [modes_acted_on_by_term_b[0][0],
                       modes_acted_on_by_term_b[1][0]]
    modes_touched_bc = [
        modes_acted_on_by_term_b[0][0], modes_acted_on_by_term_b[1][0],
        modes_acted_on_by_term_c[0][0], modes_acted_on_by_term_c[1][0]]

    # If the term_a shares no indices with bc, the double commutator is zero.
    if not (modes_acted_on_by_term_a[0][0] in modes_touched_bc or
            modes_acted_on_by_term_a[1][0] in modes_touched_bc):
        return True

    # If term_b and term_c are not both number operators and act on the
    # same modes, the commutator is zero.
    if (sum(1 for i in modes_touched_b if i in modes_touched_c) > 1 and
            (term_b_is_number_operator or term_c_is_number_operator)):
        return True

    # Create a list of all the creation and annihilations performed.
    all_changes = (modes_acted_on_by_term_a + modes_acted_on_by_term_b +
                   modes_acted_on_by_term_c)
    counts = {}
    for operator in all_changes:
        counts[operator[0]] = counts.get(operator[0], 0) + 2 * operator[1] - 1

    # If the final result creates or destroys the same mode twice.
    commutes = max(itervalues(counts)) > 1 or min(itervalues(counts)) < -1

    return commutes
