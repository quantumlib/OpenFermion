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

from ._bch_expansion import bch_expand

from ._channel_state import (amplitude_damping_channel, dephasing_channel,
                             depolarizing_channel)

from ._commutators import anticommutator, commutator, double_commutator

from ._grid import Grid

from ._lattice import (HubbardSquareLattice, SpinPairs, Spin)

from ._lcu_util import (lambda_norm,
                        preprocess_lcu_coefficients_for_reversible_sampling)

from ._operator_utils import (chemist_ordered, count_qubits,
                              eigenspectrum, fourier_transform,
                              freeze_orbitals, get_file_path,
                              hermitian_conjugated, inline_sum,
                              inverse_fourier_transform,
                              is_hermitian, is_identity,
                              normal_ordered, prune_unused_indices,
                              reorder, up_then_down,
                              load_operator, save_operator,
                              group_into_tensor_product_basis_sets)

from ._qubit_tapering_from_stabilizer import (reduce_number_of_terms,
                                              taper_off_qubits)

from ._rdm_mapping_functions import (kronecker_delta,
                                     map_two_pdm_to_two_hole_dm,
                                     map_two_pdm_to_one_pdm,
                                     map_one_pdm_to_one_hole_dm,
                                     map_one_hole_dm_to_one_pdm,
                                     map_two_pdm_to_particle_hole_dm,
                                     map_two_hole_dm_to_two_pdm,
                                     map_two_hole_dm_to_one_hole_dm,
                                     map_particle_hole_dm_to_one_pdm,
                                     map_particle_hole_dm_to_two_pdm)

from ._slater_determinants import (gaussian_state_preparation_circuit,
                                   slater_determinant_preparation_circuit)

from ._special_operators import (majorana_operator, number_operator,
                                 s_minus_operator, s_plus_operator,
                                 s_squared_operator,
                                 sx_operator, sy_operator, sz_operator)

from ._testing_utils import (haar_random_vector,
                             random_antisymmetric_matrix,
                             random_diagonal_coulomb_hamiltonian,
                             random_hermitian_matrix,
                             random_interaction_operator,
                             random_quadratic_hamiltonian,
                             random_qubit_operator,
                             random_unitary_matrix,
                             module_importable)

from ._trotter_error import error_bound, error_operator

from ._trotter_exp_to_qgates import (pauli_exp_to_qasm,
                                     trotterize_exp_qubop_to_qasm,
                                     trotter_operator_grouping)

from ._unitary_cc import (uccsd_convert_amplitude_format,
                          uccsd_generator,
                          uccsd_singlet_generator,
                          uccsd_singlet_get_packed_amplitudes,
                          uccsd_singlet_paramsize)

from ._wedge_product import (generate_parity_permutations,
                             wedge)


# Imports out of alphabetical order to avoid circular dependency.
from ._jellium_hf_state import hartree_fock_state_jellium

from ._low_rank import (get_chemist_two_body_coefficients,
                        low_rank_two_body_decomposition,
                        prepare_one_body_squared_evolution)

from ._low_depth_trotter_error import (
    low_depth_second_order_trotter_error_bound,
    low_depth_second_order_trotter_error_operator)

from ._sparse_tools import (boson_ladder_sparse,
                            boson_operator_sparse,
                            expectation,
                            expectation_computational_basis_state,
                            get_density_matrix,
                            get_gap,
                            get_ground_state,
                            get_linear_qubit_operator_diagonal,
                            inner_product,
                            jordan_wigner_sparse,
                            jw_configuration_state,
                            jw_hartree_fock_state,
                            jw_get_gaussian_state,
                            jw_get_ground_state_at_particle_number,
                            jw_number_restrict_operator,
                            jw_number_restrict_state,
                            jw_slater_determinant,
                            jw_sz_restrict_operator,
                            jw_sz_restrict_state,
                            qubit_operator_sparse,
                            sparse_eigenspectrum,
                            variance)

from ._davidson import Davidson, DavidsonOptions, QubitDavidson, SparseDavidson
from ._linear_qubit_operator import (
    LinearQubitOperator,
    LinearQubitOperatorOptions,
    ParallelLinearQubitOperator,
    generate_linear_qubit_operator,
)

from ._pubchem import geometry_from_pubchem
