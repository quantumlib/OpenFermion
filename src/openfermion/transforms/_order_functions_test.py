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

""" tests for order functions"""

import unittest
from openfermion.transforms._order_functions import *

class OrderFunctionTest(unittest.TestCase):
    def test_up_then_down(self):
        self.assertEqual(up_then_down(6,8),3)
        self.assertEqual(up_then_down(3,8),5)

