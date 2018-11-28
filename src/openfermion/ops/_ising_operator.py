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

from openfermion.ops import SymbolicOperator

class IsingOperator(SymbolicOperator):
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

    @property
    def actions(self):
        """The allowed actions."""
        return ('Z',)

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('Z',)

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return True

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return True

    def _simplify(self, term, coefficient=1.0):
        powers = defaultdict(int)
        for factor in term:
            powers[factor[0]] += 1
        odd_powers = sorted(i for i, p in powers.items() if p % 2)
        new_term = tuple((i, 'Z') for i in odd_powers)
        return coefficient, new_term
