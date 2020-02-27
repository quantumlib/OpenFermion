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

import unittest

from openfermion.utils._testing_utils import EqualsTester

from openfermion.ops._ising_operator import IsingOperator

def test_properties():
    operator = IsingOperator()
    assert operator.actions == ('Z',)
    assert operator.action_strings == ('Z',)
    assert operator.action_before_index
    assert operator.different_indices_commute


class GeneralTest(unittest.TestCase):
    """General tests."""

    def test_ising_operator(self):
        equals_tester = EqualsTester(self)

        group = [IsingOperator('Z0 Z3'), IsingOperator([(0, 'Z'), (3, 'Z')])]

        equals_tester.add_equality_group(*group)
