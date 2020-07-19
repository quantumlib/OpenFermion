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
import warnings

import numpy as np
import sympy
import cirq
import deprecation


class FSwapPowGate(cirq.EigenGate, cirq.InterchangeableQubitsGate,
                   cirq.TwoQubitGate):
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

    `ofc.FSWAP` is an instance of this gate at exponent=1. It swaps adjacent
    fermionic modes under the Jordan-Wigner Transform.
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
        #yapf: enable

    def _apply_unitary_(self,
                        args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if self.exponent != 1:
            return None

        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        ii = args.subspace_index(0b11)
        args.available_buffer[oi] = args.target_tensor[oi]
        args.target_tensor[oi] = args.target_tensor[io]
        args.target_tensor[io] = args.available_buffer[oi]
        args.target_tensor[ii] *= -1
        return args.target_tensor

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                              ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = '×ᶠ', '×ᶠ'
        else:
            symbols = 'fswap', 'fswap'
        return cirq.CircuitDiagramInfo(wire_symbols=symbols,
                                       exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'FSWAP'
        return 'FSWAP**{!r}'.format(self.exponent)

    def __repr__(self) -> str:
        if self.exponent == 1:
            return 'ofc.FSWAP'
        return '(ofc.FSWAP**{!r})'.format(self.exponent)


class XXYYPowGate(cirq.EigenGate, cirq.InterchangeableQubitsGate,
                  cirq.TwoQubitGate):
    """XX + YY interaction.

    When exponent=1, swaps the two qubits and phases |01⟩ and |10⟩ by -i. More
    generally, this gate's matrix is defined as follows:

        XXYY**t ≡ exp(-i π t (X⊗X + Y⊗Y) / 4)

    which is given by the matrix:

        [[1, 0, 0, 0],
         [0, c, -i·s, 0],
         [0, -i·s, c, 0],
         [0, 0, 0, 1]]

    where:

        c = cos(π·t/2)
        s = sin(π·t/2).

    `ofc.XXYY` is an instance of this gate at exponent=1.
    """

    @deprecation.deprecated(
        deprecated_in='v0.4.0',
        removed_in='v0.5.0',
        details='Use cirq.ISwapPowGate with negated exponent, instead.')
    def __init__(self, *args, **kwargs):
        super(XXYYPowGate, self).__init__(*args, **kwargs)

    def num_qubits(self):
        return 2

    def _eigen_components(self):
        # yapf: disable
        return [(0, np.diag([1, 0, 0, 1])),
                (-0.5,
                 np.array([[0, 0, 0, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 0, 0]])),
                (+0.5,
                 np.array([[0, 0, 0, 0],
                           [0, 0.5, -0.5, 0],
                           [0, -0.5, 0.5, 0],
                           [0, 0, 0, 0]]))]
        # yapf: enable

    def _apply_unitary_(self,
                        args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        inner_matrix = cirq.unitary(cirq.rx(self.exponent * np.pi))
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        return cirq.apply_matrix_to_slices(args.target_tensor,
                                           inner_matrix,
                                           slices=[oi, io],
                                           out=args.available_buffer)

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.X(a)**0.5
        yield cirq.CNOT(a, b)
        yield cirq.XPowGate(exponent=self.exponent / 2,
                            global_shift=self._global_shift - 0.5).on(a)
        yield cirq.YPowGate(exponent=self.exponent / 2,
                            global_shift=self._global_shift - 0.5).on(b)
        yield cirq.CNOT(a, b)
        yield cirq.X(a)**-0.5

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                              ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=('XXYY', 'XXYY'),
                                       exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'XXYY'
        return 'XXYY**{!r}'.format(self.exponent)


class YXXYPowGate(cirq.EigenGate, cirq.TwoQubitGate):
    """YX - XY interaction.

    This gate's matrix is defined as follows:

        YXXY**t ≡ exp(-i π t (Y⊗X - X⊗Y) / 4)

    which is given by the matrix:

        [[1, 0, 0, 0],
         [0, c, -s, 0],
         [0, s, c, 0],
         [0, 0, 0, 1]]

    where
        c = cos(π·t/2)
        s = sin(π·t/2).

    `ofc.YXXY` is an instance of this gate at exponent=1.
    """

    @deprecation.deprecated(deprecated_in='v0.4.0',
                            removed_in='v0.5.0',
                            details='Use cirq.PhasedISwapPowGate, instead.')
    def __init__(self, *args, **kwargs):
        super(YXXYPowGate, self).__init__(*args, **kwargs)

    def num_qubits(self):
        return 2

    def _eigen_components(self):
        # yapf: disable
        return [(0, np.diag([1, 0, 0, 1])),
                (-0.5,
                 np.array([[0, 0, 0, 0],
                           [0, 0.5, -0.5j, 0],
                           [0, 0.5j, 0.5, 0],
                           [0, 0, 0, 0]])),
                (0.5,
                 np.array([[0, 0, 0, 0],
                           [0, 0.5, 0.5j, 0],
                           [0, -0.5j, 0.5, 0],
                           [0, 0, 0, 0]]))]
        # yapf: enable

    def _apply_unitary_(self,
                        args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        inner_matrix = cirq.unitary(cirq.ry(-self.exponent * np.pi))
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        return cirq.apply_matrix_to_slices(args.target_tensor,
                                           inner_matrix,
                                           slices=[oi, io],
                                           out=args.available_buffer)

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z(a)**-0.5
        yield XXYY(a, b)**self.exponent
        yield cirq.Z(a)**0.5

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                              ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=('YXXY', '#2'),
                                       exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'YXXY'
        return 'YXXY**{!r}'.format(self.exponent)


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
    return cirq.CZ**(rads / pi)


FSWAP = FSwapPowGate()

# Deprecated
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    XXYY = XXYYPowGate()
    YXXY = YXXYPowGate()
