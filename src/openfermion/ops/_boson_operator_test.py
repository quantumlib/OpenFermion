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

"""Tests for BosonOperator."""

import unittest

from openfermion.ops._boson_operator import BosonOperator
from openfermion.utils import number_operator


class BosonOperatorTest(unittest.TestCase):

    def test_is_normal_ordered_empty(self):
        op = BosonOperator() * 2
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_number(self):
        op = BosonOperator('2^ 2') * -1j
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_reversed(self):
        self.assertFalse(BosonOperator('2 2^').is_normal_ordered())

    def test_is_normal_ordered_create(self):
        self.assertTrue(BosonOperator('11^').is_normal_ordered())

    def test_is_normal_ordered_annihilate(self):
        self.assertTrue(BosonOperator('0').is_normal_ordered())

    def test_is_normal_ordered_long_not(self):
        self.assertTrue(BosonOperator('0 5^ 3^ 2^ 1^').is_normal_ordered())

    def test_is_normal_ordered_outoforder(self):
        self.assertTrue(BosonOperator('0 1').is_normal_ordered())

    def test_is_normal_ordered_long_descending(self):
        self.assertTrue(BosonOperator('5^ 3^ 2^ 1^ 0').is_normal_ordered())

    def test_is_normal_ordered_multi(self):
        op = BosonOperator('4 3 2^ 2') + BosonOperator('1 2')
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_multiorder(self):
        op = BosonOperator('4 3 2 1') + BosonOperator('3 2')
        self.assertTrue(op.is_normal_ordered())

    def test_is_boson_preserving_BosonOperator(self):
        op = BosonOperator()
        self.assertTrue(op.is_boson_preserving())

    def test_is_boson_preserving_number(self):
        op = number_operator(n_modes=5, mode=3, parity=1)
        self.assertTrue(op.is_boson_preserving())

    def test_is_boson_preserving_three(self):
        op = BosonOperator(((0, 1), (2, 1), (4, 0)))
        self.assertFalse(op.is_boson_preserving())

    def test_is_boson_preserving_out_of_order(self):
        op = BosonOperator(((0, 1), (2, 0), (1, 1), (3, 0)))
        self.assertTrue(op.is_boson_preserving())
