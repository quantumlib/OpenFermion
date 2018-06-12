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

"""Functions to reduce the number of qubits involved in modeling a given system.
"""

import numpy

from openfermion.ops import QubitOperator


def project_onto_sector(operator, qubits, sectors):
    '''
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
    '''
    if type(operator) is not QubitOperator:
        raise ValueError('''Input operator must be a QubitOperator''')

    projected_operator = QubitOperator()
    for term, factor in operator.terms.items():

        # Any term containing X or Y on the removed
        # qubits has an expectation value of zero
        if [t for t in term if t[0] in qubits
                and t[1] in ['X', 'Y']]:
            continue

        new_term = tuple((t[0]-len([q for q in qubits if q < t[0]]), t[1])
                         for t in term if t[0] not in qubits)
        new_factor =\
            factor * (-1)**(sum([sectors[qubits.index(t[0])]
                                 for t in term if t[0] in qubits]))
        projected_operator += QubitOperator(new_term, new_factor)

    return projected_operator


def projection_error(operator, qubits, sectors):
    '''
    Calculates the error from the project_onto_sector function.

    Args:
        operator: the QubitOperator to work on
        qubits: a list of indices of qubits in
           operator to remove
        sectors: for each qubit, whether to project
            into the 0 subspace (<Z>=1) or the
            1 subspace (<Z>=-1).

    Returns:
        error: the trace norm of the removed term.
    '''
    if type(operator) is not QubitOperator:
        raise ValueError('''Input operator must be a QubitOperator''')

    error = 0
    for term, factor in operator.terms.items():

        # Any term containing X or Y on the removed
        # qubits contributes to the error
        if [t for t in term if t[0] in qubits
                and t[1] in ['X', 'Y']]:
            error += abs(factor)**2

    return numpy.sqrt(error)
