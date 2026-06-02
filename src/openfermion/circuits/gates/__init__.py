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

from .common_gates import FSWAP, FSwapPowGate, Rxxyy, Ryxxy, Rzz, rot11

from .fermionic_simulation import (
    state_swap_eigen_component,
    fermionic_simulation_gates_from_interaction_operator,
    sum_of_interaction_operator_gate_generators,
    ParityPreservingFermionicGate,
    InteractionOperatorFermionicGate,
    QuadraticFermionicSimulationGate,
    CubicFermionicSimulationGate,
    QuarticFermionicSimulationGate,
)

from .four_qubit_gates import DoubleExcitation, DoubleExcitationGate

from .three_qubit_gates import rot111, CRxxyy, CRyxxy
