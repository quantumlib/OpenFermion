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

"""Useful miscelaneous functions to transform QubitOperators
"""

import numpy

from openfermion.ops import QubitOperator


def project_onto_sector(operator, qubits, sectors):
    """
    Remove qubit by projecting onto sector.

    Takes a QubitOperator, and projects out a list
    of qubits, into either the +1 or -1 sector.
    Note - this requires knowledge of which sector
    we wish to project into.

    Args:
        operator: the QubitOperator to work on
        qubits: a list of indices of qubits in
           operator to remove
        sectors: for each qubit, whether to project
            into the 0 subspace (<Z>=1) or the
            1 subspace (<Z>=-1).

    Returns:
        projected_operator: the resultant operator

    Raises:
        TypeError: operator must be a QubitOperator.
        TypeError: qubits and sector must be an array-like.
        ValueError: If qubits and sectors have different length.
        ValueError: If sector are not specified as 0 or 1.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError('Input operator must be a QubitOperator.')
    if not isinstance(qubits, (list, numpy.ndarray)):
        raise TypeError('Qubit input must be an array-like.')
    if not isinstance(sectors, (list, numpy.ndarray)):
        raise TypeError('Sector input must be an array-like.')
    if len(qubits) != len(sectors):
        raise ValueError('Number of qubits and sectors must be equal.')
    for i in sectors:
        if i not in [0, 1]:
            raise ValueError('Sectors must be 0 or 1.')

    projected_operator = QubitOperator()
    for term, factor in operator.terms.items():

        # Any term containing X or Y on the removed
        # qubits has an expectation value of zero
        if [t for t in term if t[0] in qubits and t[1] in ['X', 'Y']]:
            continue

        new_term = tuple((t[0] - len([q for q in qubits if q < t[0]]), t[1])
                         for t in term if t[0] not in qubits)
        new_factor =\
            factor * (-1)**(sum([sectors[qubits.index(t[0])]
                                 for t in term if t[0] in qubits]))
        projected_operator += QubitOperator(new_term, new_factor)

    return projected_operator


def projection_error(operator, qubits, sectors):
    """
    Calculate the error from the project_onto_sector function.

    Args:
        operator: the QubitOperator to work on
        qubits: a list of indices of qubits in
           operator to remove
        sectors: for each qubit, whether to project
            into the 0 subspace (<Z>=1) or the
            1 subspace (<Z>=-1).

    Returns:
        error: the trace norm of the removed term.

    Raises:
        TypeError: operator must be a QubitOperator.
        TypeError: qubits and sector must be an array-like.
        ValueError: If qubits and sectors have different length.
        ValueError: If sector are not specified as 0 or 1.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError('Input operator must be a QubitOperator.')
    if not isinstance(qubits, (list, numpy.ndarray)):
        raise TypeError('Qubit input must be an array-like.')
    if not isinstance(sectors, (list, numpy.ndarray)):
        raise TypeError('Sector input must be an array-like.')
    if len(qubits) != len(sectors):
        raise ValueError('Number of qubits and sectors must be equal.')
    for i in sectors:
        if i not in [0, 1]:
            raise ValueError('Sectors must be 0 or 1.')

    error = 0
    for term, factor in operator.terms.items():

        # Any term containing X or Y on the removed
        # qubits contributes to the error
        if [t for t in term if t[0] in qubits and t[1] in ['X', 'Y']]:
            error += abs(factor)**2

    return numpy.sqrt(error)


def rotate_qubit_by_pauli(qop, pauli, angle):
    r"""
    Rotate qubit operator by exponential of Pauli.

    Perform the rotation e^{-i \theta * P}Qe^{i \theta * P}
    on a qubitoperator Q and a Pauli operator P.

    Args:
        qop: the QubitOperator to be rotated
        pauli: a single Pauli operator - a QubitOperator with
            a single term, and a coefficient equal to 1.
        angle: the angle to be rotated by.

    Returns:
        rotated_op - the rotated QubitOperator following the
            above formula.

    Raises:
        TypeError: qop must be a QubitOperator
        TypeError: pauli must be a Pauli Operator (QubitOperator
            with single term and coefficient equal to 1).
    """
    pvals = list(pauli.terms.values())
    if type(qop) is not QubitOperator:
        raise TypeError('This can only rotate QubitOperators')

    if (type(pauli) is not QubitOperator or
            len(pauli.terms) != 1 or
            pvals[0] != 1):
        raise TypeError('This can only rotate by Pauli operators')

    pqp = pauli * qop * pauli
    even_terms = 0.5 * (qop + pqp)
    odd_terms = 0.5 * (qop - pqp)

    rotated_op = even_terms + numpy.cos(2 * angle) * odd_terms + \
        1j * numpy.sin(2 * angle) * odd_terms * pauli

    return rotated_op
