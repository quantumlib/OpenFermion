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
    from .adagrad import (
        adagrad,
        constant,
        exponential_decay,
        inverse_time_decay,
        make_schedule,
        piecewise_constant,
        polynomial_decay,
    )
    from .thc_factorization import (
        adagrad_opt_thc,
        lbfgsb_opt_cholesky,
        lbfgsb_opt_thc,
        lbfgsb_opt_thc_l2reg,
    )
    from .thc_objectives import (
        cp_ls_cholesky_factor_objective,
        thc_objective,
        thc_objective_and_grad,
        thc_objective_grad,
        thc_objective_grad_jax,
        thc_objective_jax,
        thc_objective_regularized,
    )
