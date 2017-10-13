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

from ._chemical_series import (make_atomic_ring,
                               make_atomic_lattice,
                               make_atom)

from ._givens_rotations import givens_decomposition

from ._grid import Grid

from ._hubbard import fermi_hubbard

from ._jellium import (dual_basis_kinetic,
                       dual_basis_potential,
                       dual_basis_jellium_model,
                       jellium_model,
                       jordan_wigner_dual_basis_jellium,
                       plane_wave_kinetic,
                       plane_wave_potential)

from ._meanfield import meanfield_dwave

from ._dual_basis_trotter_error import (dual_basis_error_bound,
                                        dual_basis_error_operator)

from ._molecular_data import MolecularData, periodic_table

from ._operator_utils import (commutator, count_qubits,
                              eigenspectrum, get_file_path, is_identity,
                              load_operator, save_operator)

from ._plane_wave_hamiltonian import (dual_basis_external_potential,
                                      fourier_transform,
                                      inverse_fourier_transform,
                                      plane_wave_external_potential,
                                      plane_wave_hamiltonian,
                                      jordan_wigner_dual_basis_hamiltonian,
                                      wigner_seitz_length_scale)

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

# Imports out of alphabetical order to avoid circular dependancy.
from ._jellium_hf_state import hartree_fock_state_jellium
