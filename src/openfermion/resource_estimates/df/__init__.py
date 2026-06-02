# coverage:ignore
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

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from .compute_cost_df import compute_cost
    from .compute_lambda_df import compute_lambda
    from .factorize_df import factorize
    from .generate_costing_table_df import generate_costing_table
