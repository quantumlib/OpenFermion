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
"""Tests  _symbolic_operator.py."""

import unittest

import numpy

from openfermion.ops._binary_polynomial import (BinaryPolynomial,
                                                BinaryPolynomialError,
                                                _SYMBOLIC_ONE)


class BinaryPolynomialTest(unittest.TestCase):
    def test_init_long_string(self):
        operator1 = BinaryPolynomial('w1 w2 1 + 11')
        self.assertEqual(operator1.terms, [(1, 2), (_SYMBOLIC_ONE,)])
        operator1 = BinaryPolynomial('a1 b2 1 + 1')
        self.assertEqual(operator1.terms, [(1, 2), (_SYMBOLIC_ONE,)])
        with self.assertRaises(ValueError):
            BinaryPolynomial('1 + wx')

    def test_init_string(self):
        operator1 = BinaryPolynomial('w1')
        self.assertEqual(operator1.terms, [(1,)])
        operator1 = BinaryPolynomial('9 w1 w2 + 5')
        self.assertEqual(str(operator1), '[W1 W2] + [1]')
        operator1 = BinaryPolynomial(u'9 w1 w2 + 5')
        self.assertEqual(str(operator1), '[W1 W2] + [1]')

    def test_none_init(self):
        operator1 = BinaryPolynomial()
        self.assertEqual(operator1.terms, [])
        operator1 = BinaryPolynomial([])
        self.assertEqual(operator1.terms, [])
        with self.assertRaises(ValueError):
            operator1 = BinaryPolynomial(12.0)

    def test_int_init(self):
        operator1 = BinaryPolynomial(3)
        self.assertEqual(operator1.terms, [(_SYMBOLIC_ONE,)])

    def test_init_list(self):
        operator1 = BinaryPolynomial([(3, 4, _SYMBOLIC_ONE)])
        self.assertEqual(operator1.terms, [(3, 4)])
        operator1 = BinaryPolynomial([(4, 3, _SYMBOLIC_ONE)])
        self.assertEqual(operator1.terms, [(3, 4)])
        operator1 = BinaryPolynomial(((1, 2), (1, 2)))
        self.assertEqual(operator1.terms, [])
        with self.assertRaises(ValueError):
            operator1 = BinaryPolynomial(((1, -2),))
        with self.assertRaises(ValueError):
            operator1 = BinaryPolynomial(((1, -2.0),))

    def test_multiplication(self):
        operator1 = BinaryPolynomial('1 + w1 w2')
        operator2 = BinaryPolynomial([(3, 4, _SYMBOLIC_ONE)])
        multiplication = operator1 * operator2
        self.assertEqual(multiplication.terms, [(3, 4), (1, 2, 3, 4)])
        operator1 = BinaryPolynomial([(_SYMBOLIC_ONE,)])
        operator1 *= operator1
        self.assertEqual(str(operator1), '[1]')
        operator1 = 1 * operator1
        self.assertEqual(str(operator1), '[1]')
        for idx in numpy.arange(3):
            operator1 = idx * operator1
        with self.assertRaises(TypeError):
            operator1 *= 4.3
        with self.assertRaises(TypeError):
            _ = 4.3 * operator1
        with self.assertRaises(TypeError):
            _ = operator1 * 4.3

    def test_addition(self):
        operator1 = BinaryPolynomial('w1 w2')
        operator2 = BinaryPolynomial('1 + w1 w2')
        addition = operator1 + operator2
        self.assertEqual(addition.terms, [(_SYMBOLIC_ONE,)])
        addition = addition + 1
        self.assertEqual(addition.terms, [])
        addition = addition + 1
        self.assertEqual(addition.terms, [(_SYMBOLIC_ONE,)])
        with self.assertRaises(TypeError):
            _ = 4.3 + operator1
        with self.assertRaises(TypeError):
            operator1 += 4.3

    def test_string_output(self):
        operator1 = BinaryPolynomial('w15')
        self.assertEqual(str(operator1), '[W15]')
        operator1 = BinaryPolynomial()
        self.assertEqual(operator1.__repr__(), '0')

    def test_power(self):
        operator1 = BinaryPolynomial('1 + w1 w2 + w3 w4')
        pow_loc = operator1 ** 2
        self.assertEqual(pow_loc.terms, [(_SYMBOLIC_ONE,), (1, 2),
                                         (3, 4)])
        with self.assertRaises(TypeError):
            _ = operator1 ** 4.3
        with self.assertRaises(TypeError):
            _ = operator1 ** (-1)

    def test_init_binary_rule(self):
        operator1 = BinaryPolynomial('1 + w2 w2 + w2')
        self.assertEqual(operator1.terms, [(_SYMBOLIC_ONE,)])

    def test_multiply_by_one(self):
        operator1 = BinaryPolynomial('1 w1 w3')
        self.assertEqual(operator1.terms, [(1, 3)])

    def test_multiply_by_zero(self):
        operator1 = BinaryPolynomial('w1 w3 0')
        self.assertEqual(operator1.terms, [])
        operator1 = BinaryPolynomial('w1 w3')
        operator1 *= 4
        self.assertEqual(operator1.terms, [])

    def test_ordering(self):
        operator1 = BinaryPolynomial('w3 w2 w1 w4')
        self.assertEqual(operator1.terms, [(1, 2, 3, 4)])
        operator1 = BinaryPolynomial('1 + w1 w2 + w2 w1')
        self.assertEqual(operator1.terms, [(_SYMBOLIC_ONE,)])

    def test_rmul(self):
        operator1 = BinaryPolynomial('1 + w1 w2')
        operator1 *= BinaryPolynomial('w1 w2')
        self.assertEqual(operator1.terms, [])

    def test_add_integer(self):
        operator1 = BinaryPolynomial('1 + w1 w2')
        operator1 += 1
        self.assertEqual(operator1.terms, [(1, 2)])

    def test_add_integer2(self):
        operator1 = BinaryPolynomial('1 + w1 w2')
        operator1 += 2
        self.assertEqual(operator1.terms, [(_SYMBOLIC_ONE,), (1, 2)])

    def test_shift(self):
        operator1 = BinaryPolynomial('1 + w1 w2')
        operator1.shift(3)
        self.assertEqual(operator1.terms, [(_SYMBOLIC_ONE,), (4, 5)])
        with self.assertRaises(TypeError):
            operator1.shift(3.5)

    def test_count_qubits(self):
        operator1 = BinaryPolynomial('1 + w0 w2 w5')
        qubits = operator1.enumerate_qubits()
        self.assertEqual(qubits, [0, 2, 5])

    def test_evaluate(self):
        operator1 = BinaryPolynomial()
        self.assertEqual(operator1.evaluate('1111'), 0)
        operator1 = BinaryPolynomial(1)
        self.assertEqual(operator1.evaluate('1111'), 1)
        operator1 = BinaryPolynomial('1 + w0 w2 w1 + w0 w1 + w0 w2')
        a = operator1.evaluate([0, 0, 1])
        self.assertEqual(a, 1.0)
        a = operator1.evaluate([1, 0, 1])
        self.assertEqual(a, 0.0)
        a = operator1.evaluate([1, 1, 1])
        self.assertEqual(a, 0.0)
        a = operator1.evaluate('1111')
        self.assertEqual(a, 0.0)
        with self.assertRaises(BinaryPolynomialError):
            operator1.evaluate([1, 1])

    def test_init_binary_rule2(self):
        operator1 = BinaryPolynomial('w1 w1 + 1')
        self.assertEqual(operator1.terms, [(1,), (_SYMBOLIC_ONE,)])

    def test_power_null(self):
        operator1 = BinaryPolynomial('w1 w1 + 1')
        is_one = operator1 ** 0
        self.assertEqual(is_one.terms, [(_SYMBOLIC_ONE,), ])

    def test_addition_evaluations(self):
        operator1 = BinaryPolynomial('w1 w1 + 1')
        operator2 = operator1 + 1
        self.assertEqual(operator1.terms, [(1,), (_SYMBOLIC_ONE,)])
        self.assertEqual(operator2.terms, [(1,)])

        operator2 = 1 + operator1
        self.assertEqual(operator1.terms, [(1,), (_SYMBOLIC_ONE,)])
        self.assertEqual(operator2.terms, [(1,)])

        operator2 += operator1
        self.assertEqual(operator1.terms, [(1,), (_SYMBOLIC_ONE,)])
        self.assertEqual(operator2.terms, [(_SYMBOLIC_ONE,)])
