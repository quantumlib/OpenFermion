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
'''Tests for the projection module'''
from __future__ import absolute_import

import unittest

from openfermion.ops import QubitOperator
from openfermion.transforms import project_onto_sector, projection_error
from openfermion.utils import count_qubits


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_projection(self):
        coefficient = 0.5
        opstring = ((0, 'X'), (1, 'X'), (2, 'Z'))
        opstring2 = ((0, 'X'), (2, 'Z'), (3, 'Z'))
        operator = QubitOperator(opstring, coefficient)
        operator += QubitOperator(opstring2, coefficient)
        new_operator = project_onto_sector(
                operator, qubits=[2, 3], sectors=[0, 1])
        error = projection_error(operator, qubits=[2, 3], sectors=[0, 1])
        self.assertEqual(count_qubits(new_operator), 2)
        self.assertEqual(error, 0)
        self.assertTrue(((0, 'X'), (1, 'X')) in new_operator.terms)
        self.assertEqual(new_operator.terms[((0, 'X'), (1, 'X'))], 0.5)
        self.assertTrue(((0, 'X'),) in new_operator.terms)
        self.assertEqual(new_operator.terms[((0, 'X'),)], -0.5)

    def test_projection_error(self):
        coefficient = 0.5
        opstring = ((0, 'X'), (1, 'X'), (2, 'Z'))
        opstring2 = ((0, 'X'), (2, 'Z'), (3, 'Z'))
        operator = QubitOperator(opstring, coefficient)
        operator += QubitOperator(opstring2, coefficient)
        new_operator = project_onto_sector(operator, qubits=[1], sectors=[0])
        error = projection_error(operator, qubits=[1], sectors=[0])
        self.assertEqual(count_qubits(new_operator), 3)
        self.assertTrue(((0, 'X'), (1, 'Z'), (2, 'Z')) in new_operator.terms)
        self.assertEqual(new_operator.terms[((0, 'X'), (1, 'Z'), (2, 'Z'))],
                         0.5)
        self.assertEqual(error, 0.5)
