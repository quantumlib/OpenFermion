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

from .equality_constraint_projection import (
    apply_constraints,
    constraint_matrix,
    linearize_term,
    unlinearize_term,
    operator_to_vector,
    vector_to_operator,
)

from .qubit_partitioning import (
    binary_partition_iterator,
    group_into_tensor_product_basis_sets,
    partition_iterator,
    pauli_string_iterator,
)

from .fermion_partitioning import (
    pair_within,
    pair_between,
    pair_within_simultaneously,
    pair_within_simultaneously_binned,
    pair_within_simultaneously_symmetric,
)

from .get_interaction_rdm import get_interaction_rdm

from .rdm_equality_constraints import one_body_fermion_constraints, two_body_fermion_constraints

from .vpe_estimators import PhaseFitEstimator, get_phase_function
