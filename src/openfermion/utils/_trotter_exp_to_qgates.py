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

from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits
import numpy

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


def third_order_trotter_helper(op_a, op_b):
    """Recursively trotterize a QubitOperator according
        to the scheme: e^(A+B) = e^(7/24 * A) e^(2/3 * B) e^(3/4 * A)
        e^(-2/3 * B) e^(-1/24 * A) e^(B)

    Note:
        See N. Hatano and M. Suzuki Lect. Notes Phys 679, 37-68 (2005)

    Args:
        op_a: (QubitOperator) first term in hamiltonian
        op_b: (list of QubitOperator) the rest of the terms in the
                hamiltonian as a list of QubitOperators
    Returns:
        list of QubitOperators giving the trotterized hamiltonian
    """
    # Create list to hold return value
    ret_val = []

    # e^(7/24 * A)
    ret_val.append(7.0/24.0 * op_a)

    # e^(2/3 * B), recurse if B is not a single operator
    if len(op_b) == 1:
        ret_val.append(2.0 / 3.0 * op_b[0])
    else:
        btemp = []
        for op in op_b[1:]:
            btemp.append(op * 2.0/3.0)
        atemp = op_b[0] * 2.0/3.0
        ret_val.extend(third_order_trotter_helper(atemp, btemp))

    # e^(3/4 * A)
    ret_val.append(3.0/4.0 * op_a)

    # e^(-2/3 * B), recurse if B is not a single operator
    if len(op_b) == 1:
        ret_val.append(-2.0/3.0 * op_b[0])
    else:
        btemp = []
        for op in op_b[1:]:
            btemp.append(op * -2.0/3.0)
        atemp = op_b[0] * -2.0/3.0
        ret_val.extend(third_order_trotter_helper(atemp, btemp))

    # e^(-1/24 * A)
    ret_val.append(-1.0/24.0 * op_a)

    # e^(B), recurse if B is not a single operator
    if len(op_b) == 1:
        ret_val.append(1.0 * op_b[0])
    else:
        btemp = []
        for op in op_b[1:]:
            btemp.append(1.0 * op)
        atemp = op_b[0] * 1.0
        ret_val.extend(third_order_trotter_helper(atemp, btemp))

    return ret_val


def get_trotterized_qubops(
        hamiltonian,
        trotter_number=None,
        trotter_order=None,
        term_ordering=None,
        k_exp=None):
    """Perform Trotter decomposition without exponentiating, to be
    exponentiated later.

    Args:
        hamiltonian(QubitOperator): full hamiltonian
        trotter_number(int): optional number of trotter steps -
            default is 1
        trotter_order(int): optional order of trotterization as
            an integer from 1-3 - default is 1
        term_ordering (list of (tuples of tuples)): optional list
            of QubitOperator terms dictionary keys that specifies
            order of terms when trotterizing
        k_exp (float, int, or double): optional exponential factor
            to all terms when trotterizing

    Note:
        The default term_ordering is simply the ordered keys of
        the QubitOperators.terms dict.

    Returns:
        A list of single-Pauli-string QubitOperators

    Raises:
        ValueError if order > 3 or order <= 0,
        TypeError for incorrect types
    """

    # Check for default arguments and type errors
    if trotter_order is None:
        trotter_order = 1
    elif trotter_order > 3 or trotter_order <= 0:
        raise ValueError("Invalid trotter order: " + str(order))
    if trotter_number is None:
        trotter_number = 1
    if k_exp is None:
        k_exp = 1.0
    elif isinstance(k_exp, int):
        k_exp = float(k_exp)
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError("Hamiltonian must be a QubitOperator.")
    if not hamiltonian.terms:
        raise TypeError("Hamiltonian must be a non-empty QubitOperator.")
    if term_ordering is None:
        # To have consistent default behavior, ordering = sorted keys.
        term_ordering = sorted(list(hamiltonian.terms.keys()))
    if len(term_ordering) == 0:
        raise TypeError("term_ordering must None or non-empty list.")

    ret_val = []
    # First order trotter
    if trotter_order == 1:
        for step in range(trotter_number):
            for op in term_ordering:
                ret_val.append(QubitOperator(op,
                                             hamiltonian.terms[op]
                                             * float(k_exp)
                                             / float(trotter_number)
                                             ))
    # Second order trotter
    elif trotter_order == 2:
        if len(term_ordering) < 2:
            raise ValueError("Not enough terms in the Hamiltonian to do " +
                             "second order trotterization")

        for op in term_ordering[:-1]:
            ret_val.append(QubitOperator(op,
                                         (hamiltonian.terms[op]
                                          * k_exp
                                          / (2.0 * float(trotter_number)))
                                         ))

        ret_val.append(QubitOperator(term_ordering[-1],
                       hamiltonian.terms[term_ordering[-1]] * float(k_exp)
                       / float(trotter_number)))

        for op in reversed(term_ordering[:-1]):
            ret_val.append(QubitOperator(op, hamiltonian.terms[op]
                                         * float(k_exp)
                                         / (2.0 * float(trotter_number))))
        ret_val_copy = []
        ret_val_copy.extend(ret_val)
        for step in range(trotter_number - 1):
            ret_val.extend(ret_val_copy)

    # Third order trotter
    elif trotter_order == 3:
        if len(term_ordering) < 2:
            raise ValueError("Not enough terms in the Hamiltonian to do " +
                             "third order trotterization")
        # Make sure original hamiltonian is not modified
        ham = hamiltonian / float(trotter_number)
        ham *= float(k_exp)
        atemp = QubitOperator(term_ordering[0], ham.terms[term_ordering[0]])
        btemp = []
        for term in term_ordering[1:]:
            btemp.append(QubitOperator(term, ham.terms[term]))
        ret_val = third_order_trotter_helper(atemp, btemp)
        ret_val_copy = []
        ret_val_copy.extend(ret_val)
        for step in range(trotter_number - 1):
            ret_val.extend(ret_val_copy)

    # Return list, each member a single Pauli-string QubitOperator
    return ret_val


def exp_sgl_pauli_string_to_qasm(output_stream,
                                 qubit_operator):
    """Exponentiate a single-Pauli-string QubitOperator
    and print it in QASM format to a supplied output stream

    Args:
        output_stream (file/string/IO stream): QASM output stream
        qubit_operator: single Pauli-term QubitOperator to be exponentiated
            and printed

    Returns:
        None
    """

    for term in qubit_operator.terms:
        term_coeff = qubit_operator.terms[term]

        # List of operators and list of qubit ids
        ops = []
        qids = []
        string_basis_1 = []  # Basis rotations 1
        string_basis_2 = []  # Basis rotations 2

        for p in term:  # p = single pauli term
            qid = p[0]  # Qubit index
            pop = p[1]  # Pauli op

            qids.append(qid)  # Qubit index
            ops.append(pop)  # Pauli operator

            if pop == 'X':
                string_basis_1.append("H {}".format(qid))    # Hadamard
                string_basis_2.append("H {}".format(qid))    # Hadamard
            elif pop == 'Y':
                string_basis_1.append("Rx 1.57079632679 {}".format(qid))
                string_basis_2.append("Rx -1.57079632679 {}".format(qid))

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
        output_stream.write('\n'.join(string_basis_1))
        if not len(string_basis_1) == 0:
            output_stream.write('\n')

        # 2. First set CNOTs
        output_stream.write('\n'.join(cnots1))
        output_stream.write('\n')

        # 3. Rotation (Note kexp & Ntrot)
        output_stream.write("Rz {} {}\n".format(term_coeff, qids[-1]))

        # 4. Second set of CNOTs
        output_stream.write('\n'.join(cnots2))
        output_stream.write('\n')

        # 5. Rotate back to Z basis
        output_stream.write('\n'.join(string_basis_2))
        if not len(string_basis_2) == 0:
            output_stream.write('\n')


def product_exp_pauli_strings_to_qasm(output_stream, op_list):
    """Sends a list of QubitOperators to be exponentiated one by one
    and printed to a QASM stream one at a time. The input is a list of
    Pauli-strings and the output is the ordered product of their
    exponents.

    Args:
    output_stream (file/string/IO stream): QASM output stream
    op_list(list of QubitOperators): list of QubitOperators to be
        exponentiated

    Returns:
        None
    """

    for op in op_list:
        exp_sgl_pauli_string_to_qasm(output_stream, op)


def trotterize_exp_qubop_to_qasm(
                output_stream,
                hamiltonian,
                trotter_number=None,
                trotter_order=None,
                term_ordering=None,
                k_exp=None):
    """Trotterize a Qubit hamiltonian, exponentiate each trotterized term
    and write it to a QASM output stream.

        Params:
            output_stream (file/string/IO stream): QASM output stream
            hamiltonian(QubitOperator): hamiltonian
            trotter_number (int): optional number of trotter steps (slices) for
                trotterization as an integer - default = 1
            trotter_order: optional order of trotterization as an integer -
                default = 1
            term_ordering (list of (tuples of tuples)): list of tuples
                (QubitOperator terms dictionary keys) that specifies
                order of terms when trotterizing
            k_exp (int, float, double): optional exponential factor to all
                terms when trotterizing

        Returns:
            None

    """

    trotterized_ham = get_trotterized_qubops(hamiltonian,
                                             trotter_number,
                                             trotter_order,
                                             term_ordering,
                                             k_exp)
    output_stream.write(str(count_qubits(hamiltonian))+'\n')
    output_stream.write("# ***\n")
    product_exp_pauli_strings_to_qasm(output_stream, trotterized_ham)
