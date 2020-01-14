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
import copy
from openfermion.ops import QubitOperator


def _has_disjoint_indices(op1, op2):
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

    return indices_in_op1.isdisjoint(indices_in_op2)


def get_hamiltonian_subsets(hamiltonian, stabilizers):
    """
    Create subsets of Hamiltonian.

    This function splits a Hamiltonian into subsets of Pauli strings that do
    not overlap with stabilizers.
    Pauli strings might be repeated between subsets if they do not share any
    qubit with the stabilizer.
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
        remaining_paulis(QubitOperator): Operator that contains the Pauli
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
    remaining_paulis = copy.deepcopy(hamiltonian)
    for stab in list(stabilizers):
        # Initialize subset
        ham_subset = QubitOperator()
        for pauli in hamiltonian:
            if _has_disjoint_indices(pauli, stab) is True:
                ham_subset += pauli
                # Try if a Pauli string is still in the remaining set
                # otherwise continue.
                try:
                    remaining_paulis.terms.pop(list(pauli.terms.keys())[0])
                except:
                    continue

        # Adds the subset to the list.
        hamiltonian_subsets.append(ham_subset)

    return hamiltonian_subsets, remaining_paulis
