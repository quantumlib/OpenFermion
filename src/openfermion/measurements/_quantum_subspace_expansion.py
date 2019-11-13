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

"""Quantum subspace expansion functions."""

import copy
import numpy

import scipy.linalg

from openfermion.ops import QubitOperator


def get_additional_operators(hamiltonian, expansion_operators):
    """
    Generate an operator with additional Pauli strings.

    Quantum subspace expansion requires additional Pauli strings.
    This function generates a QubitOperator with all the Pauli
    operator to be measured to perform QSE given the expansion
    operator.

    Args:
        hamiltonian(QubitOperator): Hamiltonian.
        stabilizer_list(list): List of QubitOperators.

    Returns:
        additional_ops(QubitOperator): Operator with all terms needed
                for QSE with the given expansion operators.

    Notes:
        Be aware that the coefficient of the additional_ops operator
        are meaningless.
        This function is only useful to know which terms are required
    to be measured to perform QSE.

    Raises:
        TypeError: if hamiltonian is not QubitOperator.
        TypeError: if expansion_operators is not an array-like.
    """
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError('Hamiltonian must be a QubitOperator object.')
    if not isinstance(expansion_operators, (QubitOperator, list,
                                            numpy.ndarray)):
        raise TypeError('List of stabilizers must be an array-like.')
    if isinstance(expansion_operators, QubitOperator):
        expansion_operators = list(expansion_operators)

    ham = copy.deepcopy(hamiltonian)
    additional_ops = QubitOperator()

    for op1 in expansion_operators:
        for op2 in expansion_operators:
            additional_ops += op2 * ham * op1

    additional_ops = additional_ops + ham
    return additional_ops


def calculate_qse_spectrum(hamiltonian, expansion_operators,
                           expectation_values):
    """
    Calculate quantum subspace expansion.

    Quantum subspace expansion (QSE) was described in
    https://arxiv.org/abs/1603.05681.
    An extension to perform error mitigation based on QSE was proposed in
    https://arxiv.org/abs/1807.10050.
    If the expansion operators are symmetries or stabilizers
    S-QSE is performed.

    This function calculates the quantum subspace expansion of a given
    Hamiltonian.
    It is necessary to pass the expectation values of the Hamiltonian and the
    additional operators.

    Args:
        hamiltonian(QubitOperator): Hamiltonian a QubitOperator.
        expansion_operators(list, QubitOperator): List of QubitOperators
        of the expansion.
        expectation_values(QubitOperator): Expectation values as QubitOperator.

    Returns:
        spectrum_qse(list): Energy spectrum after QSE.

    Raises:
        TypeError: if hamiltonian is not a QubitOperator.
        TypeError: if expansion_operators is not an array-like or
                   QubitOperator.
        TypeError: if expectation_values is not a QubitOperator.
        ValueError: if length of expanded operator and length of expectation
                    values is not equal.
    """
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError('Hamiltonian must be a QubitOperator object.')
    if not isinstance(expectation_values, QubitOperator):
        raise TypeError('Hamiltonian must be a QubitOperator object.')
    if not isinstance(expansion_operators, (QubitOperator, list,
                                            numpy.ndarray)):
        raise TypeError('List of stabilizers must be an array-like.')
    # Write expansion operators as a list
    if isinstance(expansion_operators, QubitOperator):
        expansion_operators = list(expansion_operators)

    # Check if the length of additional operator equals the length of
    # expectation values.
    additional_op = get_additional_operators(hamiltonian, expansion_operators)
    if len(additional_op.terms) != len(expectation_values.terms):
        raise ValueError('The number of Pauli strings do not match '
                         'the number of expectation values.')

    # Initialize matrices to store calculated values.
    ham_mat = numpy.zeros((len(expansion_operators),
                           len(expansion_operators)))
    overlap_mat = numpy.zeros((len(expansion_operators),
                               len(expansion_operators)))

    for i in range(len(expansion_operators)):
        for j in range(len(expansion_operators)):
            op_to_trace = expectation_values * \
                expansion_operators[j] * expansion_operators[i]
            hop_to_trace = expectation_values * \
                expansion_operators[j] * hamiltonian * expansion_operators[i]

            # If an entry is 0 it will not be in the operators.
            # An error will arise, hence we use try, expect to
            # set the value in the matrix to 0.0
            try:
                ham_mat[i, j] = numpy.real(hop_to_trace.terms[()])
                overlap_mat[i, j] = numpy.real(op_to_trace.terms[()])
            except:
                ham_mat[i, j] = 0.0
                overlap_mat[i, j] = 0.0

    spectrum_qse = scipy.linalg.eigvalsh(ham_mat, overlap_mat)

    return spectrum_qse
