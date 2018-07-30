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

"""IsingOperator stores an Ising-type Hamiltonian, i.e. a sum of products of
Zs."""

from collections import defaultdict
from itertools import product, chain

from openfermion.ops import QubitOperator

class IsingOperator(QubitOperator):
    """The IsingOperator class provides an analytic representation of an
    Ising-type Hamiltonian, i.e. a sum of product of Zs.

    IsingOperator is a subclass of SymbolicOperator. Importantly, it has
    attributes set as follows:

        actions = ('Z')
        action_strings = ('Z')
        action_before_index = True
        different_indices_commute = True

    See the documentation of SymbolicOperator for more details.
    """

    actions = ('Z')
    action_strings = ('Z')
    action_before_index = True
    different_indices_commute = True
