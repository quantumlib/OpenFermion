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
'''Functions to reorder terms within SymbolicOperators'''

import itertools
import numpy

from openfermion.ops.operators import (BosonOperator, FermionOperator,
                                       QuadOperator)
from openfermion.ops.representations import InteractionOperator


def chemist_ordered(fermion_operator):
    r"""Puts a two-body fermion operator in chemist ordering.

    The normal ordering convention for chemists is different.
    Rather than ordering the two-body term as physicists do, as
    $a^\dagger a^\dagger a a$
    the chemist ordering of the two-body term is
    $a^\dagger a a^\dagger a$

    TODO: This routine can be made more efficient.

    Args:
        fermion_operator (FermionOperator): a fermion operator guarenteed to
            have number conserving one- and two-body fermion terms only.
    Returns:
        chemist_ordered_operator (FermionOperator): the input operator
            ordered in the chemistry convention.
    Raises:
        OperatorSpecificationError: Operator is not two-body number conserving.
    """
    # Make sure we're dealing with a fermion operator from a molecule.
    if not fermion_operator.is_two_body_number_conserving():
        raise TypeError('Operator is not two-body number conserving.')

    # Normal order and begin looping.
    normal_ordered_input = normal_ordered(fermion_operator)
    chemist_ordered_operator = FermionOperator()
    for term, coefficient in normal_ordered_input.terms.items():
        if len(term) == 2 or not len(term):
            chemist_ordered_operator += FermionOperator(term, coefficient)
        else:
            # Possibly add new one-body term.
            if term[1][0] == term[2][0]:
                new_one_body_term = (term[0], term[3])
                chemist_ordered_operator += FermionOperator(
                    new_one_body_term, coefficient)
            # Reorder two-body term.
            new_two_body_term = (term[0], term[2], term[1], term[3])
            chemist_ordered_operator += FermionOperator(new_two_body_term,
                                                        -coefficient)
    return chemist_ordered_operator


def normal_ordered(operator, hbar=1.):
    r"""Compute and return the normal ordered form of a FermionOperator,
    BosonOperator, QuadOperator, or InteractionOperator.

    Due to the canonical commutation/anticommutation relations satisfied
    by these operators, there are multiple forms that the same operator
    can take. Here, we define the normal ordered form of each operator,
    providing a distinct representation for distinct operators.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right). In
    addition:

    * FermionOperators: a^\dagger comes before a
    * BosonOperators: b^\dagger comes before b
    * QuadOperators: q operators come before p operators,

    Args:
        operator: an instance of the FermionOperator, BosonOperator,
            QuadOperator, or InteractionOperator classes.
        hbar (float): the value of hbar used in the definition of the
            commutator [q_i, p_j] = i hbar delta_ij. By default hbar=1.
            This argument only applies when normal ordering QuadOperators.
    """
    kwargs = {}

    if isinstance(operator, FermionOperator):
        ordered_operator = FermionOperator()
        order_fn = normal_ordered_ladder_term
        kwargs['parity'] = -1

    elif isinstance(operator, BosonOperator):
        ordered_operator = BosonOperator()
        order_fn = normal_ordered_ladder_term
        kwargs['parity'] = 1

    elif isinstance(operator, QuadOperator):
        ordered_operator = QuadOperator()
        order_fn = normal_ordered_quad_term
        kwargs['hbar'] = hbar

    elif isinstance(operator, InteractionOperator):
        constant = operator.constant
        n_modes = operator.n_qubits
        one_body_tensor = operator.one_body_tensor.copy()
        two_body_tensor = numpy.zeros_like(operator.two_body_tensor)
        quadratic_index_pairs = (
            (pq, pq) for pq in itertools.combinations(range(n_modes)[::-1], 2))
        cubic_index_pairs = (
            index_pair
            for p, q, r in itertools.combinations(range(n_modes)[::-1], 3)
            for index_pair in [((p, q), (p, r)), ((p, r), (
                p, q)), ((p, q), (q, r)), ((q, r),
                                           (p, q)), ((p, r),
                                                     (q, r)), ((q, r), (p, r))])
        quartic_index_pairs = (
            index_pair
            for p, q, r, s in itertools.combinations(range(n_modes)[::-1], 4)
            for index_pair in [((p, q), (r, s)), ((r, s), (
                p, q)), ((p, r), (q, s)), ((q, s),
                                           (p, r)), ((p, s),
                                                     (q, r)), ((q, r), (p, s))])
        index_pairs = itertools.chain(quadratic_index_pairs, cubic_index_pairs,
                                      quartic_index_pairs)
        for pq, rs in index_pairs:
            two_body_tensor[pq + rs] = sum(
                s * ss * operator.two_body_tensor[pq[::s] + rs[::ss]]
                for s, ss in itertools.product([-1, 1], repeat=2))
        return InteractionOperator(constant, one_body_tensor, two_body_tensor)

    else:
        raise TypeError('Can only normal order FermionOperator, '
                        'BosonOperator, QuadOperator, or InteractionOperator.')

    for term, coefficient in operator.terms.items():
        ordered_operator += order_fn(term, coefficient, **kwargs)

    return ordered_operator


def normal_ordered_ladder_term(term, coefficient, parity=-1):
    """Return a normal ordered FermionOperator or BosonOperator corresponding
    to single term.

    Args:
        term (list or tuple): A sequence of tuples. The first element of each
            tuple is an integer indicating the mode on which a fermion ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient(complex or float): The coefficient of the term.
        parity (int): parity=-1 corresponds to a Fermionic term that should be
            ordered based on the canonical anti-commutation relations.
            parity=1 corresponds to a Bosonic term that should be ordered based
            on the canonical commutation relations.

    Returns:
        ordered_term: a FermionOperator or BosonOperator instance.
            The normal ordered form of the input.
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

    if parity == -1:
        Op = FermionOperator
    elif parity == 1:
        Op = BosonOperator

    ordered_term = Op()

    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if raising on right and lowering on left.
            if right_operator[1] and not left_operator[1]:
                term[j - 1] = right_operator
                term[j] = left_operator
                coefficient *= parity

                # Replace a a^\dagger with 1 + parity*a^\dagger a
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1):]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_ladder_term(
                        tuple(new_term), parity * coefficient, parity)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:

                # If same two Fermionic operators are repeated,
                # evaluate to zero.
                if parity == -1 and right_operator[0] == left_operator[0]:
                    return ordered_term

                # Swap if same ladder type but lower index on left.
                elif right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator
                    coefficient *= parity

    # Add processed term and return.
    ordered_term += Op(tuple(term), coefficient)
    return ordered_term


def normal_ordered_quad_term(term, coefficient, hbar=1.):
    """Return a normal ordered QuadOperator corresponding to single term.

    Args:
        term: A tuple of tuples. The first element of each tuple is
            an integer indicating the mode on which a boson ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient: The coefficient of the term.
        hbar (float): the value of hbar used in the definition of the
            commutator [q_i, p_j] = i hbar delta_ij. By default hbar=1.

    Returns:
        ordered_term (QuadOperator): The normal ordered form of the input.
            Note that this might have more terms.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, q operators come first.
    """
    # Iterate from left to right across operators and reorder to normal
    # form. Swap terms operators into correct position by moving from
    # left to right across ladder operators.
    term = list(term)
    ordered_term = QuadOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if q on right and p on left.
            # p q -> q p
            if right_operator[1] == 'q' and not left_operator[1] == 'q':
                term[j - 1] = right_operator
                term[j] = left_operator

                # Replace p q with i hbar + q p
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1)::]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_quad_term(
                        tuple(new_term), -coefficient * 1j * hbar)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:

                # Swap if same type but lower index on left.
                if right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator

    # Add processed term and return.
    ordered_term += QuadOperator(tuple(term), coefficient)
    return ordered_term


def reorder(operator, order_function, num_modes=None, reverse=False):
    """Changes the ladder operator order of the Hamiltonian based on the
    provided order_function per mode index.

    Args:
        operator (SymbolicOperator): the operator that will be reordered. must
            be a SymbolicOperator or any type of operator that inherits from
            SymbolicOperator.
        order_function (func): a function per mode that is used to map the
            indexing. must have arguments mode index and num_modes.
        num_modes (int): default None. User can provide the number of modes
            assumed for the system. if None, the number of modes will be
            calculated based on the Operator.
        reverse (bool): default False. if set to True, the mode mapping is
            reversed. reverse = True will not revert back to original if
            num_modes calculated differs from original and reverted.

    Note: Every order function must take in a mode_idx and num_modes.
    """

    if num_modes is None:
        num_modes = max(
            [factor[0] for term in operator.terms for factor in term]) + 1

    mode_map = {
        mode_idx: order_function(mode_idx, num_modes)
        for mode_idx in range(num_modes)
    }

    if reverse:
        mode_map = {val: key for key, val in mode_map.items()}

    rotated_hamiltonian = operator.__class__()
    for term, value in operator.terms.items():
        new_term = tuple([(mode_map[op[0]], op[1]) for op in term])
        rotated_hamiltonian += operator.__class__(new_term, value)
    return rotated_hamiltonian
