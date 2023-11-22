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

from .general_hubbard import (
    FermiHubbardModel,
    number_operator,
    interaction_operator,
    tunneling_operator,
    number_difference_operator,
)

from .hartree_fock import (
    rhf_func_generator,
    rhf_minimization,
    HartreeFockFunctional,
    rhf_params_to_matrix,
    get_matrix_of_eigs,
    generate_hamiltonian,
)

from .hubbard import bose_hubbard, fermi_hubbard

from .jellium import (
    dual_basis_kinetic,
    dual_basis_potential,
    dual_basis_jellium_model,
    jellium_model,
    jordan_wigner_dual_basis_jellium,
    hypercube_grid_with_given_wigner_seitz_radius_and_filling,
    plane_wave_kinetic,
    plane_wave_potential,
    wigner_seitz_length_scale,
)

from .jellium_hf_state import hartree_fock_state_jellium, lowest_single_particle_energy_states

from .mean_field_dwave import mean_field_dwave

from .richardson_gaudin import RichardsonGaudin

from .plane_wave_hamiltonian import (
    dual_basis_external_potential,
    plane_wave_external_potential,
    plane_wave_hamiltonian,
    jordan_wigner_dual_basis_hamiltonian,
)

from .special_operators import (
    s_plus_operator,
    s_squared_operator,
    sx_operator,
    sy_operator,
    sz_operator,
    majorana_operator,
    number_operator,
)
