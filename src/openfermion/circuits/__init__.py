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
from .gates import *

from .lcu_util import (preprocess_lcu_coefficients_for_reversible_sampling,
                       lambda_norm)

from .low_rank import (get_chemist_two_body_coefficients,
                       low_rank_two_body_decomposition,
                       prepare_one_body_squared_evolution)

from .primitives import *

from .slater_determinants import (gaussian_state_preparation_circuit,
                                  slater_determinant_preparation_circuit,
                                  jw_get_gaussian_state, jw_slater_determinant)

from .trotter_exp_to_qgates import (trotter_operator_grouping,
                                    pauli_exp_to_qasm,
                                    trotterize_exp_qubop_to_qasm)

from .unitary_cc import (uccsd_generator, uccsd_convert_amplitude_format,
                         uccsd_singlet_paramsize,
                         uccsd_singlet_get_packed_amplitudes,
                         uccsd_singlet_generator)

from .trotter import *

from .translator import *
