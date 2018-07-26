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

SCALARS = (int, float, complex)


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

    actions = ('Z')
    action_strings = ('Z')
    action_before_index = True
    different_indices_commute = True

    def __imul__(self, multiplier):
        """
        Override in-place multiply of SymbolicOperator.

        Args:
            multiplier(int, complex, float, or IsingOperator): multiplier
        """

        # Handle scalars
        if isinstance(multiplier, SCALARS):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle non-scalar multipliers other than IsingOperators
        if not isinstance(multiplier, self.__class__):
            raise TypeError('Cannot in-place multiply IsingOperator by '
                            'multiplier of type {}.'.format(type(multiplier)))

        product_terms = dict()
        term_pairs = product(self.terms.items(), multiplier.terms.items())
        for ((left_factors, left_coefficient),
             (right_factors, right_coefficient)) in term_pairs:
            powers = defaultdict(int)
            for factor in chain(left_factors, right_factors):
                powers[factor[0]] += 1
            product_factors = tuple((i, 'Z') for i in
                                    sorted(i for i, p in
                                           powers.items() if p % 2))
            product_coefficient = left_coefficient * right_coefficient
            if product_factors in product_terms:
                product_terms[product_factors] += product_coefficient
            else:
                product_terms[product_factors] = product_coefficient
        self.terms = product_terms
        return self
