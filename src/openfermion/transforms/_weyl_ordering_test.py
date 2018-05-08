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
from __future__ import absolute_import
import os
import unittest

import numpy

from openfermion.ops import (BosonOperator, QuadOperator,
                             normal_ordered_boson,
                             normal_ordered_quad)
from openfermion.utils import hermitian_conjugated, is_hermitian

from openfermion.transforms._weyl_ordering import (
    mccoy, weyl_ordering)


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

class WeylOrderingTest(unittest.TestCase):

    def test_weyl_empty(self):
        for op in (BosonOperator, QuadOperator):
            res = weyl_ordering(op())
            self.assertTrue(res == op().zero())

    def test_weyl_identity(self):
        for op in (BosonOperator, QuadOperator):
            res = weyl_ordering(op(''))
            self.assertTrue(res == op().identity())

    def test_weyl_one_term(self):
        op = BosonOperator('0^')
        res = weyl_ordering(op)
        self.assertTrue(res == op)

        op = QuadOperator('q0')
        res = weyl_ordering(op)
        self.assertTrue(res == op)

    def test_weyl_one_term_multimode(self):
        op = BosonOperator('0^ 1^ 2 3')
        res = weyl_ordering(op)
        self.assertTrue(res == op)

        op = QuadOperator('q0 q1 p2 p3')
        res = weyl_ordering(op)
        self.assertTrue(res == op)

    def test_weyl_two_term_same(self):
        op = BosonOperator('0^ 0^')
        res = weyl_ordering(op)
        self.assertTrue(res == op)

        op = QuadOperator('q0 q0')
        res = weyl_ordering(op)
        self.assertTrue(res == op)

    def test_weyl_non_hermitian(self):
        op = BosonOperator('0^ 0')
        res = weyl_ordering(op)
        expected = BosonOperator('0^ 0', 0.5) \
                    + BosonOperator('0 0^', 0.5)
        self.assertTrue(res == expected)
        self.assertTrue(is_hermitian(res))

        op = QuadOperator('q0 p0')
        res = weyl_ordering(op)
        expected = QuadOperator('q0 p0', 0.5) \
                    + QuadOperator('p0 q0', 0.5)
        self.assertTrue(res == expected)
        self.assertTrue(is_hermitian(res))

    def test_weyl_non_hermitian_order(self):
        op1 = QuadOperator('q0 p0 q0')
        op2 = QuadOperator('q0 q0 p0')
        op3 = QuadOperator('p0 q0 q0')

        w1 = weyl_ordering(op1)
        w2 = weyl_ordering(op2)
        w3 = weyl_ordering(op3)

        self.assertTrue(is_hermitian(w1))
        self.assertTrue(is_hermitian(w2))
        self.assertTrue(is_hermitian(w3))

        expected = QuadOperator('q0 q0 p0', 0.5) \
                    + QuadOperator('p0 q0 q0', 0.5)
        self.assertTrue(w1 == expected)
        self.assertTrue(w2 == expected)
        self.assertTrue(w3 == expected)

    def test_weyl_coefficient(self):
        coeff = 0.5+0.6j
        op = coeff*QuadOperator('q0 p0')
        res = weyl_ordering(op)
        expected = QuadOperator('q0 p0', 0.5) \
                    + QuadOperator('p0 q0', 0.5)
        self.assertTrue(res == coeff*expected)
        self.assertFalse(is_hermitian(res))
