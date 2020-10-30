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
import pytest
import numpy
import cirq
import openfermion
from openfermion.ops.operators import QubitOperator

from openfermion.transforms.opconversions import (qubit_operator_to_pauli_sum)
from openfermion.transforms.opconversions.qubitoperator_to_paulisum import (
    _qubit_operator_term_to_pauli_string)


def test_function_raises():
    """Test function raises."""
    operator = QubitOperator('X0 X1 X2 X3', 1.0)
    with pytest.raises(TypeError):
        _qubit_operator_term_to_pauli_string(list(operator.terms.items())[0],
                                             qubits=0.0)
    with pytest.raises(TypeError):
        qubit_operator_to_pauli_sum([5.0])


def test_pauli_sum_qubits():
    """Test PauliSum creates qubits."""
    operator = QubitOperator('X0 X1 X2 X3', 1.0)
    pauli_sum = qubit_operator_to_pauli_sum(operator=operator)

    assert list(pauli_sum.qubits) == (cirq.LineQubit.range(4))


def test_identity():
    """Test correct hanlding of Identity."""
    identity_op = QubitOperator(' ', -0.5)
    pau_from_qop = _qubit_operator_term_to_pauli_string(
        term=list(identity_op.terms.items())[0], qubits=cirq.LineQubit.range(2))
    pauli_str = cirq.PauliString() * (-0.5)

    assert pauli_str == pau_from_qop


@pytest.mark.parametrize('qubitop, state_binary',
                         [(QubitOperator('Z0 Z1', -1.0), '00'),
                          (QubitOperator('X0 Y1', 1.0), '10')])
def test_expectation_values(qubitop, state_binary):
    """Test PauliSum and QubitOperator expectation value."""
    n_qubits = openfermion.count_qubits(qubitop)
    state = numpy.zeros(2**n_qubits, dtype='complex64')
    state[int(state_binary, 2)] = 1.0
    qubit_map = {cirq.LineQubit(i): i for i in range(n_qubits)}

    pauli_str = _qubit_operator_term_to_pauli_string(
        term=list(qubitop.terms.items())[0],
        qubits=cirq.LineQubit.range(n_qubits))
    op_mat = openfermion.get_sparse_operator(qubitop, n_qubits)

    expct_qop = openfermion.expectation(op_mat, state)
    expct_pauli = pauli_str.expectation_from_state_vector(state, qubit_map)

    numpy.testing.assert_allclose(expct_qop, expct_pauli)


@pytest.mark.parametrize(
    'qubitop, state_binary',
    [(QubitOperator('Z0 Z1 Z2 Z3', -1.0) + QubitOperator('X0 Y1 Y2 X3', 1.0),
      '1100'),
     (QubitOperator('X0 X3', -1.0) + QubitOperator('Y1 Y2', 1.0), '0000')])
def test_expectation_values_paulisum(qubitop, state_binary):
    """Test PauliSum and QubitOperator expectation value."""
    n_qubits = openfermion.count_qubits(qubitop)
    state = numpy.zeros(2**n_qubits, dtype='complex64')
    state[int(state_binary, 2)] = 1.0
    qubit_map = {cirq.LineQubit(i): i for i in range(n_qubits)}

    pauli_str = qubit_operator_to_pauli_sum(qubitop, list(qubit_map.keys()))
    op_mat = openfermion.get_sparse_operator(qubitop, n_qubits)

    expct_qop = openfermion.expectation(op_mat, state)
    expct_pauli = pauli_str.expectation_from_state_vector(state, qubit_map)

    numpy.testing.assert_allclose(expct_qop, expct_pauli)
