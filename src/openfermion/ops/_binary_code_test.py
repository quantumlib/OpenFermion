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

from openfermion.ops import SymbolicBinary
from openfermion.ops._binary_code import (BinaryCode,
                                          BinaryCodeError,
                                          linearize_decoder,
                                          shift_decoder)


class CodeOperatorTest(unittest.TestCase):
    def test_init_errors(self):
        with self.assertRaises(TypeError):
            BinaryCode(1,
                       [SymbolicBinary(' w1 + w0 '), SymbolicBinary('w0 + 1')])
        with self.assertRaises(TypeError):
            BinaryCode([[0, 1], [1, 0]], '1+w1')
        with self.assertRaises(BinaryCodeError):
            BinaryCode([[0, 1], [1, 0]], [SymbolicBinary(' w1 + w0 ')])
        with self.assertRaises(TypeError):
            BinaryCode([[0, 1, 1], [1, 0, 0]], ['1 + w1',
                                                SymbolicBinary('1 + w0'),
                                                2.0])
        with self.assertRaises(BinaryCodeError):
            BinaryCode([[0, 1], [1, 0]], [SymbolicBinary(' w0 '),
                                          SymbolicBinary('w0 + 1')])
        with self.assertRaises(BinaryCodeError):
            BinaryCode([[0, 1], [1, 0]], [SymbolicBinary(' w5 '),
                                          SymbolicBinary('w0 + 1')])

    def test_addition(self):
        a = BinaryCode([[0, 1, 0], [1, 0, 1]],
                       [SymbolicBinary(' w1 + w0 '), SymbolicBinary('w0 + 1'),
                        SymbolicBinary('w1')])
        d = BinaryCode([[0, 1], [1, 0]],
                       [SymbolicBinary(' w0 '), SymbolicBinary('w0 w1')])
        summation = a + d
        self.assertEqual(str(summation), "[[[0, 1, 0, 0, 0],"
                                         " [1, 0, 1, 0, 0],"
                                         " [0, 0, 0, 0, 1],"
                                         " [0, 0, 0, 1, 0]],"
                                         " '[[W1] + [W0],[W0] + [1],"
                                         "[W1],[W2],[W2 W3]]']")
        with self.assertRaises(TypeError):
            summation += 5

    def test_multiplication(self):
        a = BinaryCode([[0, 1, 0], [1, 0, 1]],
                       [SymbolicBinary(' w1 + w0 '), SymbolicBinary('w0 + 1'),
                        SymbolicBinary('w1')])
        d = BinaryCode([[0, 1], [1, 0]],
                       [SymbolicBinary(' w0 '), SymbolicBinary('w0 w1')])

        b = a * d
        self.assertEqual(b.__repr__(), "[[[1, 0, 1], [0, 1, 0]],"
                                       " '[[W0] + [W0 W1],[1] +"
                                       " [W0],[W0 W1]]']")
        b = 2 * d
        self.assertEqual(str(b), "[[[0, 1, 0, 0], [1, 0, 0, 0], "
                                 "[0, 0, 0, 1], [0, 0, 1, 0]], "
                                 "'[[W0],[W0 W1],[W2],[W2 W3]]']")
        with self.assertRaises(BinaryCodeError):
            d * a
        with self.assertRaises(TypeError):
            2.0 * a
        with self.assertRaises(TypeError):
            a *= 2.0
        with self.assertRaises(ValueError):
            b = 0 * a

    def test_linearize(self):
        a = linearize_decoder([[0, 1, 1], [1, 0, 0]])
        self.assertListEqual([str(a[0]), str(a[1])], ['[W1] + [W2]', '[W0]'])

    def test_shift(self):
        decoder = [SymbolicBinary('1'), SymbolicBinary('1 + w1 w0')]
        with self.assertRaises(TypeError):
            shift_decoder(decoder, 2.5)
