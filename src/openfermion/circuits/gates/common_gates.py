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
"""Gates that are commonly used for quantum simulation of fermions."""

from typing import Optional

import numpy as np
import sympy
import cirq


class FSwapPowGate(cirq.EigenGate, cirq.InterchangeableQubitsGate):
    """The FSWAP gate, possibly raised to a power.

    FSwapPowGate()**t = FSwapPowGate(exponent=t) and acts on two qubits in the
    computational basis as the matrix:

        [[1, 0, 0, 0],
         [0, g·c, -i·g·s, 0],
         [0, -i·g·s, g·c, 0],
         [0, 0, 0, p]]

    where:

        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2)
        p = exp(i·π·t).

    `openfermion.FSWAP` is an instance of this gate at exponent=1. It swaps
    adjacent fermionic modes under the Jordan-Wigner Transform.
    """

    def num_qubits(self):
        return 2

    def _eigen_components(self):
        # yapf: disable
        return [
            (0,
             np.array([[1, 0, 0, 0],
                       [0, 0.5, 0.5, 0],
                       [0, 0.5, 0.5, 0],
                       [0, 0, 0, 0]])),
            (1,
             np.array([[0, 0, 0, 0],
                       [0, 0.5, -0.5, 0],
                       [0, -0.5, 0.5, 0],
                       [0, 0, 0, 1]])),
        ]
        # yapf: enable

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if self.exponent != 1:
            return NotImplemented

        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        ii = args.subspace_index(0b11)
        args.available_buffer[oi] = args.target_tensor[oi]
        args.target_tensor[oi] = args.target_tensor[io]
        args.target_tensor[io] = args.available_buffer[oi]
        args.target_tensor[ii] *= -1
        return args.target_tensor

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = '×ᶠ', '×ᶠ'
        else:
            symbols = 'fswap', 'fswap'
        return cirq.CircuitDiagramInfo(wire_symbols=symbols, exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'FSWAP'
        return 'FSWAP**{!r}'.format(self.exponent)

    def __repr__(self) -> str:
        if self.exponent == 1:
            return 'openfermion.FSWAP'
        return '(openfermion.FSWAP**{!r})'.format(self.exponent)


def Rxxyy(rads: float) -> cirq.ISwapPowGate:
    """Returns a gate with the matrix exp(-i rads (X⊗X + Y⊗Y) / 2)."""
    pi = sympy.pi if isinstance(rads, sympy.Basic) else np.pi
    return cirq.ISwapPowGate(exponent=-2 * rads / pi)


def Ryxxy(rads: float) -> cirq.PhasedISwapPowGate:
    """Returns a gate with the matrix exp(-i rads (Y⊗X - X⊗Y) / 2)."""
    pi = sympy.pi if isinstance(rads, sympy.Basic) else np.pi
    return cirq.PhasedISwapPowGate(exponent=2 * rads / pi)


def Rzz(rads: float) -> cirq.ZZPowGate:
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    pi = sympy.pi if isinstance(rads, sympy.Basic) else np.pi
    return cirq.ZZPowGate(exponent=2 * rads / pi, global_shift=-0.5)


def rot11(rads: float) -> cirq.CZPowGate:
    """Phases the |11> state of two qubits by e^{i rads}."""
    pi = sympy.pi if isinstance(rads, sympy.Basic) else np.pi
    return cirq.CZ ** (rads / pi)


FSWAP = FSwapPowGate()
