#coverage:ignore
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
    from .pyscf_utils import (avas_active_space, cas_to_pyscf, ccsd_t,
                              factorized_ccsd_t, get_num_active_alpha_beta,
                              load_casfile_to_pyscf, localize, open_shell_t1_d1,
                              pyscf_to_cas, save_pyscf_to_casfile, stability)
