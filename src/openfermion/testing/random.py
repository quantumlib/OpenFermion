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

from typing import TYPE_CHECKING

import itertools
from typing import Optional

from openfermion import ops
from openfermion.testing.testing_utils import random_interaction_operator

if TYPE_CHECKING:
    import openfermion


def random_interaction_operator_term(
    order: int, real: bool = True, seed: Optional[int] = None
) -> 'openfermion.InteractionOperator':
    """Generates a random interaction operator with non-zero coefficients only
    on terms corresponding to the given number of unique orbitals.

    The number of orbitals is equal to the given order.

    Args:
        order: How many unique orbitals the non-zero terms should correspond to.
        real: Whether or not the coefficients should be real. Defaults to True.
        seed: The seed. If None (default), uses np.random.
    """

    n_orbitals = order

    if order > 4:
        return ops.InteractionOperator.zero(order)

    operator = random_interaction_operator(n_orbitals, real=real, seed=seed)
    operator.constant = 0

    for indices in itertools.product(range(n_orbitals), repeat=2):
        if len(set(indices)) != order:
            operator.one_body_tensor[indices] = 0

    for indices in itertools.product(range(n_orbitals), repeat=4):
        if len(set(indices)) != order:
            operator.two_body_tensor[indices] = 0

    return operator
