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
"""Tests for vpe_circuits.py"""

import numpy
import cirq

from openfermion.measurements import get_phase_function

from .vpe_circuits import vpe_single_circuit, vpe_circuits_single_timestep


def test_single_circuit():
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    qubits = reversed([q0, q1])
    prep = cirq.Circuit([cirq.FSimGate(theta=numpy.pi / 4, phi=0).on(q0, q1)])
    evolve = cirq.Circuit([cirq.rz(numpy.pi / 2).on(q0), cirq.rz(numpy.pi / 2).on(q1)])
    initial_rotation = cirq.ry(numpy.pi / 2).on(q0)
    final_rotation = cirq.rx(-numpy.pi / 2).on(q0)
    circuit = vpe_single_circuit(qubits, prep, evolve, initial_rotation, final_rotation)
    assert len(circuit) == 6
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    data_counts = result.data['msmt'].value_counts()
    assert data_counts[1] == 100


def test_single_timestep():
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    qubits = [q0, q1]
    prep = cirq.Circuit([cirq.FSimGate(theta=numpy.pi / 4, phi=0).on(q0, q1)])
    evolve = cirq.Circuit([cirq.rz(numpy.pi / 2).on(q0), cirq.rz(numpy.pi / 2).on(q1)])
    target_qubit = q0
    circuits = vpe_circuits_single_timestep(qubits, prep, evolve, target_qubit)
    results = []
    simulator = cirq.Simulator()
    for circuit in circuits:
        this_res = simulator.run(circuit, repetitions=10000)
        results.append(this_res)
    pf_estimation = get_phase_function(results, qubits, 0)
    # Loose bound - standard deviation on this measurement should be 0.005, so
    # 95% of times this should pass. If it fails, check the absolute number.
    assert numpy.abs(pf_estimation - 1j) < 1e-2
