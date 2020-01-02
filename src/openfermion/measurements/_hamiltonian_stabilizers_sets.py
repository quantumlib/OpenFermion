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

"""Tools to split the Hamiltonian into subsets based on the support of
its stabilizers."""

import numpy
from openfermion.ops import QubitOperator


def _has_overlapping_indices(op1, op2):
    """
    Auxiliary function of get Hamiltonian subsets.

    Check if two QubitOperators have any qubit in common.

    Args:
        op1(QubitOperator): Single Pauli string.
        op2(QubitOperator): Single Pauli string.

    Returns:
        overlap (Boolean): True if the operator and stabilizer overlap.
    """
    indices_in_op1 = {index for (index, _) in list(op1.terms.keys())[0]}
    indices_in_op2 = {index for (index, _) in list(op2.terms.keys())[0]}

    return not indices_in_op1.isdisjoint(indices_in_op2)


def _check_missing_paulis(hamiltonian, subsets):
    """
    Auxiliary function of Hamiltonian subset splitting.

    This function checks which Pauli strings are not included
    in any subset.
    The missing strings overlap with all stabilizers, and will need
    to be measured even when an error occured.

    Args:
        Hamiltonian(QubitOperator): Hamiltonian to be split.
        subsets(list): List of QubitOperators in which the Hamiltonian
            has been split into.
    Return:
        left_paulis(QubitOperator): Only if there are left strings.
            If no missing Paulis, returns an empyt QubitOperator.
    """
    # This functions uses python object sets to compare missing tuples.
    # Hamiltonian keys as a set.
    ham_set_form = set(hamiltonian.terms.keys())

    # Hamiltonian subsets as a single set object.
    aux_set = set()
    for sb in subsets:
        aux_set.update(set(sb.terms.keys()))

    # Find any missing tuple in the Hamiltonian set compared to
    # all subsets.
    left_set = ham_set_form - aux_set

    # Save missing Pauli strings in QubitOperator
    # If no missing Paulis, returns 0
    left_paulis = QubitOperator()
    for p in left_set:
        left_paulis += QubitOperator(p, hamiltonian.terms[p])
    return left_paulis


def get_hamiltonian_subsets(hamiltonian, stabilizers):
    """
    Create subsets of Hamiltonian.

    This function splits a Hamiltonian into subsets of Pauli strings that do
    not overlap with stabilizers.
    The Pauli strings that overlap with all stabilizers will be stored with a
    separate set.
    It can be used to perform SV and not disregarding all data if only a single
    stabilizers signals an error.

    Args:
        hamiltonian(QubitOperator): Hamiltonian to be split.
        stabilizers(list or QubitOperator): List of stabilizers as
            QubitOperators.
    Return:
        hamiltonian_subsets(list): List of QubitOperator which are a subset of
            the Hamiltonian.
            Each QubitOperator does not overlap with the stabilizer at the same
            position in the list.
        rest_paulis(QubitOperator): Operator that contains the Pauli
            strings that overlap with all the stabilizers.
            If there are no paulis left an empty QubitOperator is returned.
    Raises:
        TypeError: Input hamiltonian must be QubitOperator.
        TypeError: Input stabilizer_list must be QubitOperator or list.
    """
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError('Input terms must be QubitOperator.')
    if not isinstance(stabilizers, (QubitOperator, list,
                                    tuple, numpy.ndarray)):
        raise TypeError('Input stabilizers must be QubitOperator or list.')

    hamiltonian_subsets = []
    for stab in list(stabilizers):
        # Initialize subset
        ham_subset = QubitOperator()
        for pauli in hamiltonian:
            if _has_overlapping_indices(pauli, stab) is False:
                ham_subset += pauli

        # Adds the subset to the list.
        hamiltonian_subsets.append(ham_subset)

    # Check if there are missing strings
    rest_paulis = _check_missing_paulis(hamiltonian, hamiltonian_subsets)

    return hamiltonian_subsets, rest_paulis
