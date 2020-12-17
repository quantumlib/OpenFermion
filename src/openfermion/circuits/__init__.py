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

from .gates import (
    FSWAP,
    FSwapPowGate,
    Rxxyy,
    Ryxxy,
    Rzz,
    rot11,
    state_swap_eigen_component,
    fermionic_simulation_gates_from_interaction_operator,
    sum_of_interaction_operator_gate_generators,
    ParityPreservingFermionicGate,
    InteractionOperatorFermionicGate,
    QuadraticFermionicSimulationGate,
    CubicFermionicSimulationGate,
    QuarticFermionicSimulationGate,
    DoubleExcitation,
    DoubleExcitationGate,
    rot111,
    CRxxyy,
    CRyxxy,
)

from .lcu_util import (
    preprocess_lcu_coefficients_for_reversible_sampling,
    lambda_norm,
)

from .low_rank import (
    get_chemist_two_body_coefficients,
    low_rank_two_body_decomposition,
    prepare_one_body_squared_evolution,
)

from .primitives import (
    bogoliubov_transform,
    ffft,
    optimal_givens_decomposition,
    prepare_gaussian_state,
    prepare_slater_determinant,
    swap_network,
)

from .slater_determinants import (
    gaussian_state_preparation_circuit,
    slater_determinant_preparation_circuit,
    jw_get_gaussian_state,
    jw_slater_determinant,
)

from .trotter_exp_to_qgates import (
    trotter_operator_grouping,
    pauli_exp_to_qasm,
    trotterize_exp_qubop_to_qasm,
)

from .unitary_cc import (
    uccsd_generator,
    uccsd_convert_amplitude_format,
    uccsd_singlet_paramsize,
    uccsd_singlet_get_packed_amplitudes,
    uccsd_singlet_generator,
)

from .trotter import (
    LINEAR_SWAP_NETWORK,
    LinearSwapNetworkTrotterAlgorithm,
    LOW_RANK,
    LowRankTrotterAlgorithm,
    SplitOperatorTrotterAlgorithm,
    SplitOperatorTrotterStep,
    SymmetricSplitOperatorTrotterStep,
    ControlledAsymmetricSplitOperatorTrotterStep,
    AsymmetricSplitOperatorTrotterStep,
    ControlledSymmetricSplitOperatorTrotterStep,
    SPLIT_OPERATOR,
    diagonal_coulomb_potential_and_kinetic_terms_as_arrays,
    bit_mask_of_modes_acted_on_by_fermionic_terms,
    split_operator_trotter_error_operator_diagonal_two_body,
    fermionic_swap_trotter_error_operator_diagonal_two_body,
    simulation_ordered_grouped_hubbard_terms_with_info,
    low_depth_second_order_trotter_error_operator,
    low_depth_second_order_trotter_error_bound,
    simulation_ordered_grouped_low_depth_terms_with_info,
    stagger_with_info,
    simulate_trotter,
    TrotterAlgorithm,
    TrotterStep,
    error_bound,
    error_operator,
    trotter_steps_required,
)

from .vpe_circuits import (
    vpe_single_circuit,
    vpe_circuits_single_timestep,
    standard_vpe_rotation_set,
)
