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

from ._binary_code_transform import (
    binary_code_transform,
    dissolve,
)

from ._bksf import (
    bravyi_kitaev_fast,)

from ._binary_codes import (
    bravyi_kitaev_code,
    checksum_code,
    interleaved_code,
    jordan_wigner_code,
    linearize_decoder,
    parity_code,
    weight_one_binary_addressing_code,
    weight_one_segment_code,
    weight_two_segment_code,
)

from ._bravyi_kitaev import (
    bravyi_kitaev,)

from ._bravyi_kitaev_tree import (
    bravyi_kitaev_tree,)

from ._conversion import (
    get_boson_operator,
    get_diagonal_coulomb_hamiltonian,
    get_fermion_operator,
    get_interaction_rdm,
    get_interaction_operator,
    get_quadratic_hamiltonian,
    get_majorana_operator,
    get_molecular_data,
    get_number_preserving_sparse_operator,
    get_sparse_operator,
    get_quad_operator,
)

from ._jordan_wigner import (
    jordan_wigner,)

from ._qubit_operator_transforms import (
    project_onto_sector,
    projection_error,
    rotate_qubit_by_pauli,
)

from ._reverse_jordan_wigner import (
    reverse_jordan_wigner,)

from ._verstraete_cirac import (
    verstraete_cirac_2d_square,)

from ._weyl_ordering import (
    symmetric_ordering,
    weyl_polynomial_quantization,
)

from ._remove_symmetry_qubits import (
    symmetry_conserving_bravyi_kitaev,
    edit_hamiltonian_for_spin,
)
