#coverage:ignore
#   Copyright 2020 Google LLC

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
import pytest

try:
    import jax, pybtas
    from .spacetime import qubit_vs_toffoli
    from .compute_lambda_thc import compute_lambda
    from .compute_cost_thc import compute_cost
    from .factorize_thc import thc_via_cp3 as factorize
    from .generate_costing_table_thc import generate_costing_table
except ImportError:
    pytest.skip('Need jax and pybtas for THC', allow_module_level=True)
