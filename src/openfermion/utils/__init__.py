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

from ._channel_state import (amplitude_damping_channel, dephasing_channel,
                             depolarizing_channel)

from ._grid import Grid

from ._operator_utils import (commutator, count_qubits,
                              eigenspectrum, fourier_transform,
                              get_file_path, inverse_fourier_transform,
                              is_identity, load_operator, save_operator)

from ._slater_determinants import (gaussian_state_preparation_circuit,
                                   slater_determinant_preparation_circuit)

from ._sparse_tools import (expectation,
                            expectation_computational_basis_state,
                            get_density_matrix,
                            get_gap,
                            get_ground_state,
                            is_hermitian,
                            jordan_wigner_sparse,
                            jw_hartree_fock_state,
                            jw_get_gaussian_state,
                            jw_get_ground_states_by_particle_number,
                            jw_number_restrict_operator,
                            jw_slater_determinant,
                            qubit_operator_sparse,
                            sparse_eigenspectrum)

from ._trotter_error import error_bound, error_operator

from ._trotter_exp_to_qgates import (pauli_exp_to_qasm,
									 trotterize_exp_qubop_to_qasm,
									 trotter_operator_grouping)

from ._unitary_cc import (uccsd_convert_amplitude_format,
                          uccsd_operator,
                          uccsd_singlet_operator,
                          uccsd_singlet_paramsize)

# Imports out of alphabetical order to avoid circular dependancy.
from ._dual_basis_trotter_error import (dual_basis_error_bound,
                                        dual_basis_error_operator)

from ._jellium_hf_state import hartree_fock_state_jellium
