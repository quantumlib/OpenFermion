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

"""Tests  _jordan_wigner.py."""
import unittest

from openfermion.ops import BosonOperator, FermionOperator, QuadOperator
from openfermion.utils import is_hermitian

from openfermion.transforms._weyl_ordering import (
    mccoy, symmetric_ordering, weyl_polynomial_quantization)


class McCoyTest(unittest.TestCase):
    def setUp(self):
        self.mode = 0
        self.op_a = 'q'
        self.op_b = 'p'

    def test_identity(self):
        m = 0
        n = 0
        res = mccoy(self.mode, self.op_a, self.op_b, m, n)
        expected = {tuple(): 1.0}
        self.assertEqual(res, expected)

    def test_single_operator(self):
        m = 1
        n = 0
        res = mccoy(self.mode, self.op_a, self.op_b, m, n)
        expected = {((0, self.op_a),): 1.0}
        self.assertEqual(res, expected)

        m = 0
        n = 1
        res = mccoy(self.mode, self.op_a, self.op_b, m, n)
        expected = {((0, self.op_b),): 1.0}
        self.assertEqual(res, expected)

    def test_squared(self):
        m = 2
        n = 0
        res = mccoy(self.mode, self.op_a, self.op_b, m, n)
        expected = {((0, self.op_a), (0, self.op_a)): 1.0}
        self.assertEqual(res, expected)

        m = 0
        n = 2
        res = mccoy(self.mode, self.op_a, self.op_b, m, n)
        expected = {((0, self.op_b), (0, self.op_b)): 1.0}
        self.assertEqual(res, expected)

    def test_mixed(self):
        m = 1
        n = 1
        res = mccoy(self.mode, self.op_a, self.op_b, m, n)
        expected = {((0, self.op_a), (0, self.op_b)): 0.5,
                    ((0, self.op_b), (0, self.op_a)): 0.5}
        self.assertEqual(res, expected)


class WeylQuantizationTest(unittest.TestCase):

    def test_weyl_empty(self):
        res = weyl_polynomial_quantization('')
        self.assertTrue(res == QuadOperator.zero())

    def test_weyl_one_term(self):
        op = QuadOperator('q0')
        res = weyl_polynomial_quantization('q0')
        self.assertTrue(res == op)

    def test_weyl_one_term_multimode(self):
        op = QuadOperator('q0 q1 p2 p3')
        res = weyl_polynomial_quantization('q0 q1 p2 p3')
        self.assertTrue(res == op)

    def test_weyl_two_term_same(self):
        op = QuadOperator('q0 q0')
        res = weyl_polynomial_quantization('q0^2')
        self.assertTrue(res == op)

    def test_weyl_non_hermitian(self):
        res = weyl_polynomial_quantization('q0 p0')
        expected = QuadOperator('q0 p0', 0.5) \
            + QuadOperator('p0 q0', 0.5)
        self.assertTrue(res == expected)
        self.assertTrue(is_hermitian(res))

        res = weyl_polynomial_quantization('q0^2 p0')
        expected = QuadOperator('q0 q0 p0', 0.5) \
            + QuadOperator('p0 q0 q0', 0.5)
        self.assertTrue(res == expected)
        self.assertTrue(is_hermitian(res))


class SymmetricOrderingTest(unittest.TestCase):

    def test_invalid_op(self):
        op = FermionOperator()
        with self.assertRaises(TypeError):
            _ = symmetric_ordering(op)

    def test_symmetric_empty(self):
        for op in (BosonOperator, QuadOperator):
            res = symmetric_ordering(op())
            self.assertTrue(res == op().zero())

    def test_symmetric_identity(self):
        for op in (BosonOperator, QuadOperator):
            res = symmetric_ordering(op(''), ignore_identity=False)
            self.assertTrue(res == op().identity())

        for op in (BosonOperator, QuadOperator):
            res = symmetric_ordering(op(''), ignore_identity=True)
            self.assertTrue(res == op().zero())

    def test_symmetric_one_term(self):
        op = BosonOperator('0^')
        res = symmetric_ordering(op)
        self.assertTrue(res == op)

        op = QuadOperator('q0')
        res = symmetric_ordering(op)
        self.assertTrue(res == op)

    def test_symmetric_one_term_multimode(self):
        op = BosonOperator('0^ 1^ 2 3')
        res = symmetric_ordering(op)
        self.assertTrue(res == op)

        op = QuadOperator('q0 q1 p2 p3')
        res = symmetric_ordering(op)
        self.assertTrue(res == op)

    def test_symmetric_two_term_same(self):
        op = BosonOperator('0^ 0^')
        res = symmetric_ordering(op)
        self.assertTrue(res == op)

        op = QuadOperator('q0 q0')
        res = symmetric_ordering(op)
        self.assertTrue(res == op)

    def test_symmetric_non_hermitian(self):
        op = BosonOperator('0^ 0')
        res = symmetric_ordering(op)
        expected = BosonOperator('0^ 0', 0.5) \
            + BosonOperator('0 0^', 0.5)
        self.assertTrue(res == expected)
        self.assertTrue(is_hermitian(res))

        op = QuadOperator('q0 p0')
        res = symmetric_ordering(op)
        expected = QuadOperator('q0 p0', 0.5) \
            + QuadOperator('p0 q0', 0.5)
        self.assertTrue(res == expected)
        self.assertTrue(is_hermitian(res))

    def test_symmetric_non_hermitian_order(self):
        op1 = QuadOperator('q0 p0 q0')
        op2 = QuadOperator('q0 q0 p0')
        op3 = QuadOperator('p0 q0 q0')

        w1 = symmetric_ordering(op1)
        w2 = symmetric_ordering(op2)
        w3 = symmetric_ordering(op3)

        self.assertTrue(is_hermitian(w1))
        self.assertTrue(is_hermitian(w2))
        self.assertTrue(is_hermitian(w3))

        expected = QuadOperator('q0 q0 p0', 0.5) \
            + QuadOperator('p0 q0 q0', 0.5)
        self.assertTrue(w1 == expected)
        self.assertTrue(w2 == expected)
        self.assertTrue(w3 == expected)

    def test_symmetric_coefficient(self):
        coeff = 0.5+0.6j
        op = QuadOperator('q0 p0', coeff)
        res = symmetric_ordering(op, ignore_coeff=False)
        expected = QuadOperator('q0 p0', 0.5) \
            + QuadOperator('p0 q0', 0.5)
        self.assertTrue(res == coeff*expected)
        self.assertFalse(is_hermitian(res))
