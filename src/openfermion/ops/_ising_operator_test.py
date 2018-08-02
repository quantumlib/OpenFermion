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
from openfermion.ops._qubit_operator import QubitOperator

class GeneralTest(unittest.TestCase):
    """General tests."""

    def test_ising_operator(self):
        equals_tester = EqualsTester(self)

        group_1 = [IsingOperator('Z0 Z3'), 
                   IsingOperator([(0, 'Z'), (3, 'Z')])]
        group_2 = [IsingOperator('Z0', 0.2), QubitOperator('Z0', 0.2)]

        equals_tester.add_equality_group(*group_1)
        equals_tester.add_equality_group(*group_2)
