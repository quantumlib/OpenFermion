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

"""Functions for unitary transformations of operators"""
import numpy
from openfermion.ops import QubitOperator


def rotate_qubit_by_pauli(qop, pauli, angle):
    '''
    Performs the rotation e^{-i \theta * P}Qe^{i \theta * P}
    on a qubitoperator Q and a Pauli operator P.

    Args:
        qop: the QubitOperator to be rotated
        pauli: a single Pauli operator - a QubitOperator with
            a single term, and a coefficient equal to 1.
        angle: the angle to be rotated by.

    Returns:
        rotated_op - the rotated QubitOperator following the
            above formula.
    '''
    pvals = list(pauli.terms.values())
    if type(qop) is not QubitOperator:
        raise ValueError('This can only rotate QubitOperators')

    if (type(pauli) is not QubitOperator or
            len(pauli.terms) != 1 or
            pvals[0] != 1):
        raise ValueError('This can only rotate by Pauli operators')

    pqp = pauli * qop * pauli
    even_terms = 0.5 * (qop + pqp)
    odd_terms = 0.5 * (qop - pqp)

    rotated_op = even_terms + numpy.cos(2*angle) * odd_terms + \
        1j * numpy.sin(2*angle) * odd_terms * pauli

    return rotated_op
