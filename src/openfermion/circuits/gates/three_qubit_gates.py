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
"""Common gates that target three qubits."""
import numpy as np
import cirq


def rot111(rads: float) -> cirq.CCZPowGate:
    """Phases the |111> state of three qubits by e^{i rads}."""
    return cirq.CCZ ** (rads / np.pi)


def CRxxyy(rads: float) -> cirq.ControlledGate:
    """Controlled version of openfermion.Rxxyy"""
    return cirq.ControlledGate(cirq.ISwapPowGate(exponent=-2 * rads / np.pi))


def CRyxxy(rads: float) -> cirq.ControlledGate:
    """Controlled version of openfermion.Ryxxy"""
    return cirq.ControlledGate(cirq.PhasedISwapPowGate(exponent=2 * rads / np.pi))
