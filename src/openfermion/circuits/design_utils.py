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
'''Tools to assist in circuit/primitive design'''


import numpy
from scipy import linalg
import openfermion
from openfermion.config import EQ_TOLERANCE


def check_circuit_implements_trotterized_evolution(circuit, op_list, qubits):
    r'''Checks whether a circuit implements Trotterized evolution

    Takes a circuit that is supposed to implement evolution of the
    form:

    $$\prod_j \exp[iO_j]$$

    and checks whether the implemented unitary is applied.
    Ignores any global phases as part of the implementation of the unitary, as
    these are not kept consistent in cirq (and unphysical).

    Arguments:
        circuit: 'cirq.Circuit' {[type]} -- circuit to be checked
        op_list {List['openfermion.QubitOperator']} -- list of operators $O_j$
            in application order (i.e. op_list[0] is the first operator to be
            applied).
        qubits {List['cirq.Qid']} -- list of qubits in circuit in index order
    '''

    n_qubits = len(qubits)
    hs_dim = 2**n_qubits
    qubit_op = op_list[0]
    op_matrix = openfermion.get_sparse_operator(qubit_op)
    target_unitary = linalg.expm(1j * op_matrix)
    for qubit_op in op_list[1:]:
        op_matrix = openfermion.get_sparse_operator(qubit_op)
        op_unitary = linalg.expm(1j * op_matrix)
        target_unitary = op_unitary.dot(target_unitary)

    actual_unitary = circuit.unitary(qubit_order=qubits)
    overlap = float(numpy.abs(
        numpy.trace(target_unitary.dot(actual_unitary.conj().T)))) / hs_dim
    if 1 - overlap > EQ_TOLERANCE:
        raise ValueError('Unitaries do not agree within tolerance {}. Found '
                         'overlap {} (should be 1).'.format(EQ_TOLERANCE,
                                                            overlap))
    return True
