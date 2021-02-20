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
from .davidson import (
    Davidson,
    DavidsonOptions,
    DavidsonError,
    QubitDavidson,
    SparseDavidson,
    generate_random_vectors,
    append_random_vectors,
    orthonormalize,
)

from .erpa import (
    erpa_eom_hamiltonian,
    singlet_erpa,
)

from .givens_rotations import (
    givens_decomposition,
    givens_rotate,
    double_givens_rotate,
    givens_decomposition_square,
    givens_matrix_elements,
    fermionic_gaussian_decomposition,
    swap_rows,
    swap_columns,
)

from .linear_qubit_operator import (
    generate_linear_qubit_operator,
    LinearQubitOperator,
    LinearQubitOperatorOptions,
    ParallelLinearQubitOperator,
)

from .rdm_reconstruction import (
    valdemoro_reconstruction,)

from .sparse_tools import (
    wrapped_kronecker,
    kronecker_operators,
    jordan_wigner_ladder_sparse,
    jordan_wigner_sparse,
    qubit_operator_sparse,
    eigenspectrum,
    get_linear_qubit_operator_diagonal,
    jw_configuration_state,
    jw_hartree_fock_state,
    jw_number_indices,
    jw_sz_indices,
    jw_number_restrict_operator,
    jw_sz_restrict_operator,
    jw_number_restrict_state,
    jw_sz_restrict_state,
    jw_get_ground_state_at_particle_number,
    jw_sparse_givens_rotation,
    jw_sparse_particle_hole_transformation_last_mode,
    get_density_matrix,
    get_ground_state,
    sparse_eigenspectrum,
    expectation,
    variance,
    expectation_computational_basis_state,
    expectation_db_operator_with_pw_basis_state,
    expectation_one_body_db_operator_computational_basis_state,
    expectation_two_body_db_operator_computational_basis_state,
    expectation_three_body_db_operator_computational_basis_state,
    get_gap,
    inner_product,
    boson_ladder_sparse,
    single_quad_op_sparse,
    boson_operator_sparse,
    get_sparse_operator,
    get_number_preserving_sparse_operator,
)

from .wave_fitting import (
    fit_known_frequencies,
    prony,
)

from .wedge_product import (
    generate_parity_permutations,
    wedge,
)
