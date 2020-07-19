# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gates useful for simulating fermions."""

from openfermioncirq.gates.common_gates import (
    FSWAP,
    FSwapPowGate,
    Rxxyy,
    Ryxxy,
    Rzz,
    rot11,
    XXYY,
    XXYYPowGate,
    YXXY,
    YXXYPowGate,
)

from openfermioncirq.gates.three_qubit_gates import (
    CRxxyy,
    CRyxxy,
    CXXYY,
    CYXXY,
    CXXYYPowGate,
    CYXXYPowGate,
    rot111,
)

from openfermioncirq.gates.fermionic_simulation import (
    fermionic_simulation_gates_from_interaction_operator,
    ParityPreservingFermionicGate,
    QuadraticFermionicSimulationGate,
    CubicFermionicSimulationGate,
    QuarticFermionicSimulationGate,
)

from openfermioncirq.gates.four_qubit_gates import (
    DoubleExcitation,
    DoubleExcitationGate,
)

# Deprecated
# pylint: disable=wrong-import-order
import sys as _sys
from openfermioncirq._compat import wrap_module as _wrap_module
_deprecated_constants = {
    'XXYY': ('v0.5.0', 'Use cirq.ISWAP with negated exponent, instead'),
    'YXXY': ('v0.5.0', 'Use cirq.PhasedISwapPowGate, instead.'),
    'CXXYY': ('v0.5.0', 'Use cirq.ControlledGate and cirq.ISWAP with '
              'negated exponent, instead'),
    'CYXXY': ('v0.5.0', 'Use cirq.ControlledGate and '
              'cirq.PhasedISwapPowGate, instead.'),
}
_sys.modules[__name__] = _wrap_module(_sys.modules[__name__],
                                      _deprecated_constants)
