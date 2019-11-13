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


def _check_stabilizer_overlap(pauli_op, stabilizer):
    """
    Auxiliar function of get Hamiltonian subsets.

    Checks if a Pauli string has support on a stabilizer.

    Args:
        pauli_op(QubitOperator): Single Pauli string.
        stabilizer(QubitOperator): Single Pauli string.

    Returns:
        overlap (Boolean): True if the operator and stabilizer overlap.
    """
    # Find qubits involved
    qbts_in_stab = [qbt for (qbt, p) in list(stabilizer.terms.keys())[0]]

    overlap = False
    for qp in list(pauli_op.terms.keys())[0]:
        if qp == ():
            overlap = False
        if qp[0] in qbts_in_stab:
            overlap = True
    return overlap


def _check_missing_paulis(hamiltonian, subsets):
    """
    Auxiliar function of Hamiltonian subset splitting.

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

    stabilizer_list = list(copy.deepcopy(stabilizers))
    ham = copy.deepcopy(hamiltonian)

    hamiltonian_subsets = []
    for stab in stabilizer_list:
        # Initialize subset
        ham_subset = QubitOperator()
        for pauli in ham:
            if _check_stabilizer_overlap(pauli, stab) is False:
                ham_subset += pauli

        # Adds the subset to the list.
        hamiltonian_subsets.append(ham_subset)

    # Check if there are missing strings
    rest_paulis = _check_missing_paulis(ham, hamiltonian_subsets)

    return hamiltonian_subsets, rest_paulis
