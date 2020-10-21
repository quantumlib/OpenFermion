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

from .bch_expansion import bch_expand

from .channel_state import (
    amplitude_damping_channel,
    dephasing_channel,
    depolarizing_channel,
)

# Imports out of alphabetical order to avoid circular dependency.
from .lattice import (
    HubbardSquareLattice,
    HubbardLattice,
    SpinPairs,
    Spin,
)

from .commutators import (
    anticommutator,
    commutator,
    double_commutator,
    trivially_double_commutes_dual_basis_using_term_info,
    trivially_commutes_dual_basis,
    trivially_double_commutes_dual_basis,
)

from .grid import Grid

from .indexing import (
    up_index,
    down_index,
    up_then_down,
)

from .operator_utils import (
    count_qubits,
    get_file_path,
    hermitian_conjugated,
    is_hermitian,
    is_identity,
    load_operator,
    save_operator,
    OperatorUtilsError,
    OperatorSpecificationError,
)

from .rdm_mapping_functions import (
    kronecker_delta,
    map_two_pdm_to_two_hole_dm,
    map_two_pdm_to_one_pdm,
    map_one_pdm_to_one_hole_dm,
    map_one_hole_dm_to_one_pdm,
    map_two_pdm_to_particle_hole_dm,
    map_two_hole_dm_to_two_pdm,
    map_two_hole_dm_to_one_hole_dm,
    map_particle_hole_dm_to_one_pdm,
    map_particle_hole_dm_to_two_pdm,
)
