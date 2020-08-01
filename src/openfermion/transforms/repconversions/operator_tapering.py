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
"""Operations to remove qubits/spin orbitals from operators"""

import copy

from openfermion.ops.operators import (BosonOperator, FermionOperator,
                                       MajoranaOperator, QuadOperator,
                                       QubitOperator)


def freeze_orbitals(fermion_operator, occupied, unoccupied=None, prune=True):
    """Fix some orbitals to be occupied and others unoccupied.

    Removes all operators acting on the specified orbitals, and renumbers the
    remaining orbitals to eliminate unused indices. The sign of each term
    is modified according to the ladder uperator anti-commutation relations in
    order to preserve the expectation value of the operator.

    Args:
        occupied: A list containing the indices of the orbitals that are to be
            assumed to be occupied.
        unoccupied: A list containing the indices of the orbitals that are to
            be assumed to be unoccupied.
    """
    new_operator = fermion_operator
    frozen = [(index, 1) for index in occupied]
    if unoccupied is not None:
        frozen += [(index, 0) for index in unoccupied]

    # Loop over each orbital to be frozen. Within each term, move all
    # ops acting on that orbital to the right side of the term, keeping
    # track of sign flips that come from swapping operators.
    for item in frozen:
        tmp_operator = FermionOperator()
        for term in new_operator.terms:
            new_term = []
            new_coef = new_operator.terms[term]
            current_occupancy = item[1]
            n_ops = 0  # Number of operations on index that have been moved
            n_swaps = 0  # Number of swaps that have been done

            for op in enumerate(reversed(term)):
                if op[1][0] is item[0]:
                    n_ops += 1

                    # Determine number of swaps needed to bring the op in
                    # front of all ops acting on other indices
                    n_swaps += op[0] - n_ops

                    # Check if the op annihilates the current state
                    if current_occupancy == op[1][1]:
                        new_coef = 0

                    # Update current state
                    current_occupancy = (current_occupancy + 1) % 2
                else:
                    new_term.insert(0, op[1])
            if n_swaps % 2:
                new_coef *= -1
            if new_coef and current_occupancy == item[1]:
                tmp_operator += FermionOperator(tuple(new_term), new_coef)
        new_operator = tmp_operator

    # For occupied frozen orbitals, we must also bring together the creation
    # operator from the ket and the annihilation operator from the bra when
    # evaluating expectation values. This can result in an additional minus
    # sign.
    for term in new_operator.terms:
        for index in occupied:
            for op in term:
                if op[0] > index:
                    new_operator.terms[term] *= -1

    # Renumber indices to remove frozen orbitals
    new_operator = prune_unused_indices(new_operator)

    return new_operator


def prune_unused_indices(symbolic_operator):
    """
    Remove indices that do not appear in any terms.

    Indices will be renumbered such that if an index i does not appear in
    any terms, then the next largest index that appears in at least one
    term will be renumbered to i.
    """

    # Determine which indices appear in at least one term
    indices = []
    for term in symbolic_operator.terms:
        for op in term:
            if op[0] not in indices:
                indices.append(op[0])
    indices.sort()

    # Construct a dict that maps the old indices to new ones
    index_map = {}
    for index in enumerate(indices):
        index_map[index[1]] = index[0]

    new_operator = copy.deepcopy(symbolic_operator)
    new_operator.terms.clear()

    # Replace the indices in the terms with the new indices
    for term in symbolic_operator.terms:
        new_term = [(index_map[op[0]], op[1]) for op in term]
        new_operator.terms[tuple(new_term)] = symbolic_operator.terms[term]

    return new_operator
