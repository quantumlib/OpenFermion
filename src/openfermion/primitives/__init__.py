# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Building blocks of algorithms for quantum simulation."""

from openfermioncirq.primitives.bogoliubov_transform import bogoliubov_transform

from openfermioncirq.primitives.ffft import ffft

from openfermioncirq.primitives.optimal_givens_decomposition import (
    optimal_givens_decomposition)

from openfermioncirq.primitives.state_preparation import (
    prepare_gaussian_state,
    prepare_slater_determinant)

from openfermioncirq.primitives.swap_network import swap_network
