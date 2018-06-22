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
import unittest

from openfermion.ops._fermion_operator import FermionOperator
from openfermion.utils import number_operator


class FermionOperatorTest(unittest.TestCase):

    def test_is_normal_ordered_empty(self):
        op = FermionOperator() * 2
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_number(self):
        op = FermionOperator('2^ 2') * -1j
        self.assertTrue(op.is_normal_ordered())

    def test_is_normal_ordered_reversed(self):
        self.assertFalse(FermionOperator('2 2^').is_normal_ordered())

    def test_is_normal_ordered_create(self):
        self.assertTrue(FermionOperator('11^').is_normal_ordered())

    def test_is_normal_ordered_annihilate(self):
        self.assertTrue(FermionOperator('0').is_normal_ordered())

    def test_is_normal_ordered_long_not(self):
        self.assertFalse(FermionOperator('0 5^ 3^ 2^ 1^').is_normal_ordered())

    def test_is_normal_ordered_outoforder(self):
        self.assertFalse(FermionOperator('0 1').is_normal_ordered())

    def test_is_normal_ordered_long_descending(self):
        self.assertTrue(FermionOperator('5^ 3^ 2^ 1^ 0').is_normal_ordered())

    def test_is_normal_ordered_multi(self):
        op = FermionOperator('4 3 2^ 2') + FermionOperator('1 2')
        self.assertFalse(op.is_normal_ordered())

    def test_is_normal_ordered_multiorder(self):
        op = FermionOperator('4 3 2 1') + FermionOperator('3 2')
        self.assertTrue(op.is_normal_ordered())

    def test_is_two_body_number_conserving_FermionOperator(self):
        op = FermionOperator()
        self.assertTrue(op.is_two_body_number_conserving())

    def test_is_two_body_number_conserving_number(self):
        op = number_operator(5, 3)
        self.assertTrue(op.is_two_body_number_conserving())

    def test_is_two_body_number_conserving_updown(self):
        op = FermionOperator(((2, 1), (4, 0)))
        self.assertTrue(op.is_two_body_number_conserving())

    def test_is_two_body_number_conserving_downup(self):
        op = FermionOperator(((2, 0), (4, 1)))
        self.assertTrue(op.is_two_body_number_conserving())

    def test_is_two_body_number_conserving_downup_badspin(self):
        op = FermionOperator(((2, 0), (3, 1)))
        self.assertFalse(op.is_two_body_number_conserving(True))

    def test_is_two_body_number_conserving_three(self):
        op = FermionOperator(((0, 1), (2, 1), (4, 0)))
        self.assertFalse(op.is_two_body_number_conserving())

    def test_is_two_body_number_conserving_out_of_order(self):
        op = FermionOperator(((0, 1), (2, 0), (1, 1), (3, 0)))
        self.assertTrue(op.is_two_body_number_conserving())
