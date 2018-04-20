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

from ._equality_constraint_projection import (apply_constraints,
                                              constraint_matrix,
                                              linearize_term,
                                              unlinearize_term)

from ._rdm_equality_constraints import (one_body_fermion_constraints,
                                        two_body_fermion_constraints)

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
