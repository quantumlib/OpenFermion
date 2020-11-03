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
"""Circuit generation functions for verified phase estimation (2010.02538)"""

from typing import Sequence, Optional
import numpy
import cirq


def vpe_single_circuit(qubits: Sequence[cirq.Qid], prep: cirq.Circuit,
                       evolve: cirq.Circuit, initial_rotation: cirq.Gate,
                       final_rotation: cirq.Gate) -> cirq.Circuit:
    """
    Combines the different parts that make up a VPE circuit

    The protocol for VPE requires combining preparation, evolution, and
    measurement circuits for different values of time in order to estimate
    the phase function. This function takes these parts and combines them.

    Note that we need not specify the time of evolution as this is contained
    already within evolve.

    Arguments:
        prep [cirq.Circuit] -- The circuit to prepare the initial state
            (|psi_s>+|psi_r>) from |0>+|1>
        evolve [cirq.Circuit] -- The circuit to evolve for time t
        initial_rotation [cirq.Gate] -- The initial rotation on the target qubit
            (Note that the gate should already be targeting the qubit)
        final_rotation [cirq.Gate] -- The final rotation on the target qubit
            (Note that the gate should already be targeting the qubit)
    """
    circuit = cirq.Circuit()
    circuit.append(initial_rotation)
    circuit.append(prep)
    circuit.append(evolve)
    circuit.append(cirq.inverse(prep))
    circuit.append(final_rotation)
    circuit.append(cirq.measure(*qubits, key='msmt'))
    return circuit


# Turning off yapf here as its formatting suggestion is bad.
# yapf: disable
standard_vpe_rotation_set = [
    [0.25, cirq.ry(numpy.pi / 2), cirq.ry(-numpy.pi / 2)],
    [-0.25, cirq.ry(numpy.pi / 2), cirq.ry(numpy.pi / 2)],
    [-0.25j, cirq.ry(numpy.pi / 2), cirq.rx(-numpy.pi / 2)],
    [0.25j, cirq.ry(numpy.pi / 2), cirq.rx(numpy.pi / 2)],
    [0.25, cirq.rx(numpy.pi / 2), cirq.rx(-numpy.pi / 2)],
    [-0.25, cirq.rx(numpy.pi / 2), cirq.rx(numpy.pi / 2)],
    [0.25j, cirq.rx(numpy.pi / 2), cirq.ry(-numpy.pi / 2)],
    [-0.25j, cirq.rx(numpy.pi / 2), cirq.ry(numpy.pi / 2)],
]
# yapf: enable


def vpe_circuits_single_timestep(qubits: Sequence[cirq.Qid],
                                 prep: cirq.Circuit,
                                 evolve: cirq.Circuit,
                                 target_qubit: cirq.Qid,
                                 rotation_set: Optional[Sequence] = None
                                ) -> Sequence[cirq.Circuit]:
    """Prepares the circuits to perform VPE at a fixed time

    Puts together the set of pre- and post-rotations to implement
    VPE at for a given state preparation and time evolution.

    [description]

    Arguments:
        prep [cirq.Circuit] -- The circuit to prepare the target state
            (|psi_s>+|psi_r>) from |0>+|1>
        evolve [cirq.Circuit] -- The circuit to evolve for time t
        target_qubit [cirq.Qid] -- The qubit on which the phase
            function is encoded
        rotation_set [Sequence] -- A set of initial and final rotations for the
            target qubit. We average the phase function estimation over multiple
            such rotations to cancel out readout noise, final T1 decay, etc.
            The standard rotation set is typically sufficient for these
            purposes. The first element of each gate is the multiplier to get
            the phase function; we do not need this for this function.

            If rotation_set is set to None, the 'standard rotation set' of all
            possible X and Y rotations before and after the circuit is used.
    """
    if rotation_set is None:
        rotation_set = standard_vpe_rotation_set
    circuits = [
        vpe_single_circuit(qubits, prep, evolve, rdata[1].on(target_qubit),
                           rdata[2].on(target_qubit)) for rdata in rotation_set
    ]
    return circuits
