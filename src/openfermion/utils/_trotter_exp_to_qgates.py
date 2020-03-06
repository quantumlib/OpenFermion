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
"""Module to perform Trotter-Suzuki decompositions to output as circuits."""

import collections
import copy
import numpy
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits

"""
Description:
    Functions for estimating the exponential of an operator
    composed of Pauli matrices, by Trotterization. Outputs QASM
    format to a python stream.

    Change-of-basis gates:
    H           : Z to X basis (and back)
    Rx(pi/2)    : Z to Y basis .....
    Rx(-pi/2)   : Y to Z basis .....
"""


def _third_order_trotter_helper(op_list):
    """Iteratively find 3rd-order Trotter ordering of a QubitOperator.

    This Trotter ordering is done according to the scheme:
    e^(A+B) = e^(7/24 * A) e^(2/3 * B) e^(3/4 * A) * e^(-2/3 * B)
        * e^(-1/24 * A) e^(B)

    Note:
        See N. Hatano and M. Suzuki Lect. Notes Phys 679, 37-68 (2005)

    Args:
        op_list (list of QubitOperator's): the terms in the
                hamiltonian as a list of QubitOperators
    Returns:
        list of QubitOperators giving the trotterized hamiltonian
    """
    # Create list to hold return value
    ret_val = collections.deque([7.0 / 24.0 * op_list[-2],
                                2.0 / 3.0 * op_list[-1],
                                3.0 / 4.0 * op_list[-2],
                                -2.0 / 3.0 * op_list[-1],
                                -1.0 / 24.0 * op_list[-2],
                                1.0 * op_list[-1]])

    for i in reversed(range(len(op_list)-2)):
        temp_ret_val = copy.deepcopy(ret_val)
        ret_val = collections.deque()
        ret_val.appendleft(7.0 / 24.0 * op_list[i])
        ret_val.extend([2.0 / 3.0 * qubop for qubop in temp_ret_val])
        ret_val.append(3.0 / 4.0 * op_list[i])
        ret_val.extend([-2.0 / 3.0 * qubop for qubop in temp_ret_val])
        ret_val.append(-1.0 / 24.0 * op_list[i])
        ret_val.extend([1.0 * qubop for qubop in temp_ret_val])
    return ret_val


def trotter_operator_grouping(hamiltonian,
                              trotter_number=1,
                              trotter_order=1,
                              term_ordering=None,
                              k_exp=1.0):
    """Trotter-decomposes operators into groups without exponentiating.

    Operators are still Hermitian at the end of this method but have been
        multiplied by k_exp.

    Note:
        The default term_ordering is simply the ordered keys of
        the QubitOperators.terms dict.

    Args:
        hamiltonian (QubitOperator): full hamiltonian
        trotter_number (int): optional number of trotter steps -
            default is 1
        trotter_order (int): optional order of trotterization as
            an integer from 1-3 - default is 1
        term_ordering (list of (tuples of tuples)): optional list
            of QubitOperator terms dictionary keys that specifies
            order of terms when trotterizing
        k_exp (float): optional exponential factor
            to all terms when trotterizing

    Yields:
        QubitOperator generator

    Raises:
        ValueError if order > 3 or order <= 0,
        TypeError for incorrect types
    """
    # Check for default arguments and type errors
    if (trotter_order > 3) or (trotter_order <= 0):
        raise ValueError("Invalid trotter order: " + str(trotter_order))
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError("Hamiltonian must be a QubitOperator.")
    if len(hamiltonian.terms) == 0:
        raise TypeError("Hamiltonian must be a non-empty QubitOperator.")
    if term_ordering is None:
        # To have consistent default behavior, ordering = sorted keys.
        term_ordering = sorted(list(hamiltonian.terms.keys()))
    if len(term_ordering) == 0:
        raise TypeError("term_ordering must None or non-empty list.")

    # Enforce float
    k_exp = float(k_exp)

    # First order trotter
    if trotter_order == 1:
        for _ in range(trotter_number):
            for op in term_ordering:
                yield QubitOperator(
                    op, hamiltonian.terms[op] * k_exp / trotter_number)

    # Second order trotter
    elif trotter_order == 2:
        if len(term_ordering) < 2:
            raise ValueError("Not enough terms in the Hamiltonian to do " +
                             "second order trotterization")
        for _ in range(trotter_number):
            for op in term_ordering[:-1]:
                yield QubitOperator(
                    op, hamiltonian.terms[op] * k_exp / (2.0 * trotter_number))

            yield QubitOperator(
                term_ordering[-1],
                hamiltonian.terms[term_ordering[-1]] * k_exp / trotter_number)

            for op in reversed(term_ordering[:-1]):
                yield QubitOperator(
                    op, hamiltonian.terms[op] * k_exp / (2.0 * trotter_number))

    # Third order trotter
    elif trotter_order == 3:
        if len(term_ordering) < 2:
            raise ValueError("Not enough terms in the Hamiltonian to do " +
                             "third order trotterization")
        # Make sure original hamiltonian is not modified
        ham = hamiltonian * k_exp / float(trotter_number)
        ham_temp = []
        for term in term_ordering:
            ham_temp.append(QubitOperator(term, ham.terms[term]))
        for _ in range(trotter_number):
            for returned_op in _third_order_trotter_helper(ham_temp):
                yield returned_op


def pauli_exp_to_qasm(qubit_operator_list,
                      evolution_time=1.0,
                      qubit_list=None,
                      ancilla=None):
    """Exponentiate a list of QubitOperators to a QASM string generator.

    Exponentiates a list of QubitOperators, and yields string generators in
        QASM format using the formula:  exp(-1.0j * evolution_time * op).

    Args:
        qubit_operator_list (list of QubitOperators): list of single Pauli-term
            QubitOperators to be exponentiated
        evolution_time (float): evolution time of the operators in
            the list
        qubit_list: (list/tuple or None)Specifies the labels for the qubits
            to be output in qasm.
            If a list/tuple, must have length greater than or equal to the
            number of qubits in the QubitOperator. Entries in the
            list must be castable to string.
            If None, qubits are labeled by index (i.e. an integer).
        ancilla (string or None): if any, an ancilla qubit to perform
            the rotation conditional on (for quantum phase estimation)

    Yields:
        string
    """

    num_qubits = max([count_qubits(qubit_operator)
                      for qubit_operator in qubit_operator_list])
    if qubit_list is None:
        qubit_list = list(range(num_qubits))
    else:
        if type(qubit_list) is not tuple and type(qubit_list) is not list:
            raise TypeError('qubit_list must be one of None, tuple, or list.')
        if len(qubit_list) < num_qubits:
            raise TypeError('qubit_list must have an entry for every qubit')

    for qubit_operator in qubit_operator_list:
        # ret_val = ""
        ret_list = []

        for term in qubit_operator.terms:

            term_coeff = qubit_operator.terms[term]

            # Force float
            term_coeff = float(numpy.real(term_coeff))

            # List of operators and list of qubit ids
            ops = []
            qids = []
            string_basis_1 = []  # Basis rotations 1
            string_basis_2 = []  # Basis rotations 2

            for p in term:  # p = single pauli term
                qid = qubit_list[p[0]]
                pop = p[1]  # Pauli op

                qids.append(qid)  # Qubit index
                ops.append(pop)  # Pauli operator

                if pop == 'X':
                    string_basis_1.append("H {}".format(qid))    # Hadamard
                    string_basis_2.append("H {}".format(qid))    # Hadamard
                elif pop == 'Y':
                    string_basis_1.append(
                        "Rx 1.5707963267948966 {}".format(qid))
                    string_basis_2.append(
                        "Rx -1.5707963267948966 {}".format(qid))

            # Prep for CNOTs
            cnot_pairs = numpy.vstack((qids[:-1], qids[1:]))
            cnots1 = []
            cnots2 = []
            for i in range(cnot_pairs.shape[1]):
                pair = cnot_pairs[:, i]
                cnots1.append("CNOT {} {}".format(pair[0], pair[1]))
            for i in numpy.arange(cnot_pairs.shape[1])[::-1]:
                pair = cnot_pairs[:, i]
                cnots2.append("CNOT {} {}".format(pair[0], pair[1]))

            # Exponentiating each Pauli string requires five parts

            # 1. Perform basis rotations
            ret_list = ret_list + string_basis_1

            # 2. First set CNOTs
            ret_list = ret_list + cnots1

            # 3. Rotation (Note kexp & Ntrot)
            if ancilla is not None:
                if len(qids) > 0:
                    ret_list = ret_list + ["C-Phase {} {} {}".format(
                        -2 * term_coeff * evolution_time, ancilla, qids[-1])]
                    ret_list = ret_list + ["Rz {} {}".format(
                        1 * term_coeff * evolution_time, ancilla)]
                else:
                    ret_list = ret_list + ["Rz {} {}".format(
                        1 * term_coeff*evolution_time, ancilla)]
            else:
                if len(qids) > 0:
                    ret_list = ret_list + ["Rz {} {}".format(
                        term_coeff * evolution_time, qids[-1])]

            # 4. Second set of CNOTs
            ret_list = ret_list + cnots2

            # 5. Rotate back to Z basis
            ret_list = ret_list + string_basis_2

            for gate in ret_list:
                yield gate


def trotterize_exp_qubop_to_qasm(hamiltonian,
                                 evolution_time=1,
                                 trotter_number=1,
                                 trotter_order=1,
                                 term_ordering=None,
                                 k_exp=1.0,
                                 qubit_list=None,
                                 ancilla=None):
    """Trotterize a Qubit hamiltonian and write it to QASM format.

    Assumes input hamiltonian is still hermitian and -1.0j has not yet been
    applied. Therefore, signs of coefficients should reflect this. Returns
    a generator which generates a QASM file.

    Args:
        hamiltonian (QubitOperator): hamiltonian
        trotter_number (int): optional number of trotter steps (slices) for
            trotterization as an integer - default = 1
        trotter_order: optional order of trotterization as an integer -
            default = 1
        term_ordering (list of (tuples of tuples)): list of tuples
            (QubitOperator terms dictionary keys) that specifies
            order of terms when trotterizing
        qubit_list: (list/tuple or None)Specifies the labels for the qubits
            to be output in qasm.
            If a list/tuple, must have length greater than or equal to the
            number of qubits in the QubitOperator. Entries in the
            list must be castable to string.
            If None, qubits are labeled by index (i.e. an integer).
        k_exp (float): optional exponential factor to all
            terms when trotterizing

        Yields:
            string generator

    """

    for trotterized_op in trotter_operator_grouping(hamiltonian,
                                                    trotter_number,
                                                    trotter_order,
                                                    term_ordering,
                                                    k_exp):
        for exponentiated_qasm_string in pauli_exp_to_qasm(
                [trotterized_op], evolution_time=evolution_time,
                qubit_list=qubit_list, ancilla=ancilla):
            yield exponentiated_qasm_string
