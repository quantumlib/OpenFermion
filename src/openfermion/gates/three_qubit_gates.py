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

from typing import Optional
import warnings

import numpy as np
import cirq
import deprecation

from openfermioncirq.gates import common_gates


def rot111(rads: float) -> cirq.CCZPowGate:
    """Phases the |111> state of three qubits by e^{i rads}."""
    return cirq.CCZ**(rads / np.pi)


class CXXYYPowGate(cirq.EigenGate, cirq.ThreeQubitGate):
    """Controlled XX + YY interaction."""

    @deprecation.deprecated(
        deprecated_in='v0.4.0',
        removed_in='v0.5.0',
        details=('Use cirq.ControlledGate and cirq.ISwapPowGate with '
                 'negated exponent, instead.'))
    def __init__(self, *args, **kwargs):
        super(CXXYYPowGate, self).__init__(*args, **kwargs)

    def _apply_unitary_(self,
                        args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        return cirq.apply_unitary(cirq.ControlledGate(
            common_gates.XXYY**self.exponent),
                                  args,
                                  default=None)

    def _eigen_components(self):
        minus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]), np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.diag([0]))
        plus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]), np.array([[0.5, -0.5], [-0.5, 0.5]]),
            np.diag([0]))

        return [(0, np.diag([1, 1, 1, 1, 1, 0, 0, 1])),
                (-0.5, minus_half_component), (0.5, plus_half_component)]

    def _decompose_(self, qubits):
        control, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.H(a)
        yield cirq.CCZ(control, a, b)**self.exponent
        # Note: Clifford optimization would merge this CZ into the CCZ decomp.
        yield cirq.CZ(control, b)**(-self.exponent / 2)
        yield cirq.H(a)
        yield cirq.CNOT(a, b)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=('@', 'XXYY', 'XXYY'),
                                       exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'CXXYY'
        return 'CXXYY**{!r}'.format(self.exponent)


class CYXXYPowGate(cirq.EigenGate, cirq.ThreeQubitGate):
    """Controlled YX - XY interaction."""

    @deprecation.deprecated(
        deprecated_in='v0.4.0',
        removed_in='v0.5.0',
        details=('Use cirq.ControlledGate and cirq.PhasedISwapPowGate, '
                 'instead.'))
    def __init__(self, *args, **kwargs):
        super(CYXXYPowGate, self).__init__(*args, **kwargs)

    def _apply_unitary_(self,
                        args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        return cirq.apply_unitary(cirq.ControlledGate(
            common_gates.YXXY**self.exponent),
                                  args,
                                  default=None)

    def _eigen_components(self):
        minus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]), np.array([[0.5, -0.5j], [0.5j, 0.5]]),
            np.diag([0]))
        plus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
            np.diag([0]))

        return [(0, np.diag([1, 1, 1, 1, 1, 0, 0, 1])),
                (-0.5, minus_half_component), (0.5, plus_half_component)]

    def _decompose_(self, qubits):
        control, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.X(a)**0.5
        yield cirq.CCZ(control, a, b)**self.exponent
        # Note: Clifford optimization would merge this CZ into the CCZ decomp.
        yield cirq.CZ(control, b)**(-self.exponent / 2)
        yield cirq.X(a)**-0.5
        yield cirq.CNOT(a, b)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=('@', 'YXXY', '#2'),
                                       exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'CYXXY'
        return 'CYXXY**{!r}'.format(self.exponent)


def CRxxyy(rads: float) -> cirq.ControlledGate:
    """Controlled version of ofc.Rxxyy"""
    return cirq.ControlledGate(cirq.ISwapPowGate(exponent=-2 * rads / np.pi))


def CRyxxy(rads: float) -> cirq.ControlledGate:
    """Controlled version of ofc.Ryxxy"""
    return cirq.ControlledGate(
        cirq.PhasedISwapPowGate(exponent=2 * rads / np.pi))


# Deprecated
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    CXXYY = CXXYYPowGate()
    CYXXY = CYXXYPowGate()
