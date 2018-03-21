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

from openfermion.ops import BinaryCode, FermionOperator, QubitOperator
from openfermion.transforms import binary_code_transform, dissolve


class CodeTransformTest(unittest.TestCase):
    def test_transform(self):
        code = BinaryCode([[1, 0, 0], [0, 1, 0]], ['W0', 'W1', '1 + W0 + W1'])
        hamiltonian = FermionOperator('0^ 2', 0.5) + FermionOperator('2^ 0',
                                                                     0.5)
        transform = binary_code_transform(hamiltonian, code)
        correct_op = QubitOperator('X0 Z1',0.25) + QubitOperator('X0',0.25)
        self.assertTrue(transform == correct_op)

        with self.assertRaises(TypeError):
            binary_code_transform('0^ 2', code)
        with self.assertRaises(TypeError):
            binary_code_transform(hamiltonian,
                                  ([[1, 0], [0, 1]], ['w0', 'w1']))

    def test_dissolve(self):
        code = BinaryCode([[1, 0, 0], [0, 1, 0]], ['W0', 'W1', '1 + W0 W1'])
        hamiltonian = FermionOperator('0^ 2', 0.5) + FermionOperator('2^ 0',
                                                                     0.5)
        transform = binary_code_transform(hamiltonian, code)

        correct_op = QubitOperator('X0 Z1', 0.375) + \
                     QubitOperator('X0', -0.125) + \
                     QubitOperator('Y0', -0.125j) + \
                     QubitOperator('Y0 Z1', -0.125j)
        self.assertTrue(transform == correct_op)

        with self.assertRaises(ValueError):
            dissolve(((1, '1'),))
