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
'''Tests for circuit_validation.py'''

import cirq
from openfermion import QubitOperator

from openfermion.testing.circuit_validation import validate_trotterized_evolution


def test_checking_passes():
    '''Simple test that circuit checking passes'''
    angle = 0.652
    z_rotation_op = angle * QubitOperator('Z0')
    qubits = [cirq.GridQubit(0, 0)]
    circuit = cirq.Circuit([cirq.rz(-2 * angle).on(qubits[0])])
    res = validate_trotterized_evolution(circuit, [z_rotation_op], qubits)
    assert res is True


def test_checking_passes_twoops():
    '''Simple test that circuit checking passes'''
    anglez = 0.652
    anglex = 0.334
    z_rotation_op = anglez * QubitOperator('Z0')
    x_rotation_op = anglex * QubitOperator('X0')
    qubits = [cirq.GridQubit(0, 0)]
    circuit = cirq.Circuit([cirq.rz(-2 * anglez).on(qubits[0]), cirq.rx(-2 * anglex).on(qubits[0])])
    res = validate_trotterized_evolution(circuit, [z_rotation_op, x_rotation_op], qubits)
    assert res is True


def test_checking_fails():
    '''Simple test that circuit checking fails'''
    angle = 0.652
    z_rotation_op = angle * QubitOperator('Z0')
    qubits = [cirq.GridQubit(0, 0)]
    circuit = cirq.Circuit([cirq.rz(angle).on(qubits[0])])
    res = validate_trotterized_evolution(circuit, [z_rotation_op], qubits)
    assert res is False
