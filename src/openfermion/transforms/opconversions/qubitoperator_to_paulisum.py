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
from typing import Optional, Sequence
import cirq
from openfermion.ops.operators import QubitOperator
from openfermion.utils.operator_utils import count_qubits


def _qubit_operator_term_to_pauli_string(term: dict, qubits: Sequence[cirq.Qid]
                                        ) -> cirq.PauliString:
    """
    Convert term of QubitOperator to a PauliString.

    Args:
        term (dict): QubitOperator term.
        qubits (list): List of qubit names.
    Returns:
        pauli_string (PauliString): cirq PauliString object.
    """
    ind_ops, coeff = term

    return cirq.PauliString(dict((qubits[ind], op) for ind, op in ind_ops),
                            coeff)


def qubit_operator_to_pauli_sum(operator: QubitOperator,
                                qubits: Optional[Sequence[cirq.Qid]] = None
                               ) -> cirq.PauliSum:
    """
    Convert QubitOperator to a sum of PauliString.

    Args:
        operator (QubitOperator): operator to convert.
        qubits (List): Optional list of qubit names.
            If `None` a list of `cirq.LineQubit` of length number of qubits
            in operator is created.

    Returns:
        pauli_sum (PauliSum): cirq PauliSum object.

    Raises:
        TypeError: if qubit_op is not a QubitOpertor.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError('Input must be a QubitOperator.')

    if qubits is None:
        qubits = cirq.LineQubit.range(count_qubits(operator))

    pauli_sum = cirq.PauliSum()
    for pauli in operator.terms.items():
        pauli_sum += _qubit_operator_term_to_pauli_string(pauli, qubits)

    return pauli_sum
