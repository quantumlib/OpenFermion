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

"""Tests for QuadOperator."""

import unittest

from openfermion.ops._quad_operator import QuadOperator


class QuadOperatorTest(unittest.TestCase):

    def test_is_normal_ordered_empty(self):
        op = QuadOperator() * 2
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_number(self):
        op = QuadOperator('q2 p2') * -1j
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_reversed(self):
        self.assertFalse(QuadOperator('p2 q2').is_normal_ordered())

    def test_is_normal_ordered_q(self):
        self.assertTrue(QuadOperator('q11').is_normal_ordered())

    def test_is_normal_ordered_p(self):
        self.assertTrue(QuadOperator('p0').is_normal_ordered())

    def test_is_normal_ordered_different_indices(self):
        self.assertTrue(QuadOperator('p0 q5 q3 q2 q1').is_normal_ordered())

    def test_is_normal_ordered_multi(self):
        op = QuadOperator('p4 p3 q2 p2') + QuadOperator('p1 p2')
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_multiorder(self):
        op = QuadOperator('p4 p3 p2 p1') + QuadOperator('p3 p2')
        self.assertTrue(op.is_normal_ordered())

    def test_is_gaussian_QuadOperator(self):
        op = QuadOperator()
        self.assertTrue(op.is_gaussian())

        op = QuadOperator('')
        self.assertTrue(op.is_gaussian())

        op1 = QuadOperator('q0')
        self.assertTrue(op1.is_gaussian())

        op2 = QuadOperator('q0 q0')
        self.assertTrue(op2.is_gaussian())

        op3 = QuadOperator('p0')
        self.assertTrue(op3.is_gaussian())

        op4 = QuadOperator('p0 p0')
        self.assertTrue(op4.is_gaussian())

        op5 = QuadOperator('q0 p0')
        self.assertTrue(op5.is_gaussian())

        op6 = QuadOperator('q0 q0 q0')
        self.assertFalse(op6.is_gaussian())

        op = op1 + op2 + op3 + op4 + op5
        self.assertTrue(op.is_gaussian())
        op += op6
        self.assertFalse(op.is_gaussian())
