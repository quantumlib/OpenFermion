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

"""Tests  _fermion_operator.py."""
import copy
import numpy
import unittest

from openfermion.ops._quad_operator import (QuadOperator,
                                            QuadOperatorError,
                                            normal_ordered)


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

    def test_normal_ordered_single_term(self):
        op = QuadOperator('p4 p3 p2 p1') + QuadOperator('p3 p2')
        self.assertTrue(op == normal_ordered(op))

        op = QuadOperator('q0 p0') - QuadOperator('p0 q0')
        expected = QuadOperator('', 2.j)
        self.assertTrue(expected == normal_ordered(op, hbar=2.))

    def test_normal_ordered_two_term(self):
        op_b = QuadOperator('p0 q0 p3', 88.)
        normal_ordered_b = normal_ordered(op_b, hbar=2)
        expected = QuadOperator('p3', -88.*2j) + QuadOperator('q0 p0 p3', 88.0)
        self.assertTrue(normal_ordered_b == expected)

    def test_normal_ordered_offsite(self):
        op = QuadOperator(((3, 'p'), (2, 'q')))
        self.assertTrue(op == normal_ordered(op))

    def test_normal_ordered_offsite_reversed(self):
        op = QuadOperator(((3, 'q'), (2, 'p')))
        expected = QuadOperator(((2, 'p'), (3, 'q')))
        self.assertTrue(expected == normal_ordered(op))

    def test_normal_ordered_triple(self):
        op_132 = QuadOperator(((1, 'p'), (3, 'q'), (2, 'q')))
        op_123 = QuadOperator(((1, 'p'), (2, 'q'), (3, 'q')))
        op_321 = QuadOperator(((3, 'q'), (2, 'q'), (1, 'p')))

        self.assertTrue(op_132 == normal_ordered(op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_is_linear_QuadOperator(self):
        op = QuadOperator()
        self.assertTrue(op.is_linear())

        op = QuadOperator('q0 p0 p1')
        self.assertTrue(op.is_linear())

        op = QuadOperator('q0 p0 q0 p0 p1')
        self.assertTrue(op.is_linear())

        op = QuadOperator('q0 p0 q0 p0 q0 p1')
        self.assertFalse(op.is_linear())