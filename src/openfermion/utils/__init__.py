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

from ._grid import Grid

from ._operator_utils import (commutator, count_qubits,
                              eigenspectrum, fourier_transform,
                              get_file_path, inverse_fourier_transform,
                              is_identity, load_operator, save_operator)

from ._slater_determinants import (givens_decomposition,
                                   fermionic_gaussian_decomposition)

from ._sparse_tools import (expectation,
                            expectation_computational_basis_state,
                            get_density_matrix,
                            get_gap,
                            get_ground_state,
                            is_hermitian,
                            jordan_wigner_sparse,
                            jw_hartree_fock_state,
                            jw_number_restrict_operator,
                            qubit_operator_sparse,
                            sparse_eigenspectrum)

from ._trotter_error import error_bound, error_operator

from ._unitary_cc import (uccsd_convert_amplitude_format,
                          uccsd_operator,
                          uccsd_singlet_operator,
                          uccsd_singlet_paramsize)

# Imports out of alphabetical order to avoid circular dependancy.
from ._dual_basis_trotter_error import (dual_basis_error_bound,
                                        dual_basis_error_operator)

from ._jellium_hf_state import hartree_fock_state_jellium
