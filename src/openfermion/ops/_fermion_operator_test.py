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

from openfermion.ops._fermion_operator import (FermionOperator,
                                               FermionOperatorError,
                                               normal_ordered,
                                               freeze_orbitals)
from openfermion.utils import number_operator


class FermionOperatorTest(unittest.TestCase):

    def test_number_operator_site(self):
        op = number_operator(3, 2, 1j)
        self.assertTrue(op.isclose(FermionOperator(((2, 1), (2, 0))) * 1j))

    def test_number_operator_nosite(self):
        op = number_operator(4)
        expected = (FermionOperator(((0, 1), (0, 0))) +
                    FermionOperator(((1, 1), (1, 0))) +
                    FermionOperator(((2, 1), (2, 0))) +
                    FermionOperator(((3, 1), (3, 0))))
        self.assertTrue(op.isclose(expected))

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

    def test_normal_ordered_single_term(self):
        op = FermionOperator('4 3 2 1') + FermionOperator('3 2')
        self.assertTrue(op.isclose(normal_ordered(op)))

    def test_normal_ordered_two_term(self):
        op_b = FermionOperator(((2, 0), (4, 0), (2, 1)), -88.)
        normal_ordered_b = normal_ordered(op_b)
        expected = (FermionOperator(((4, 0),), 88.) +
                    FermionOperator(((2, 1), (4, 0), (2, 0)), 88.))
        self.assertTrue(normal_ordered_b.isclose(expected))

    def test_normal_ordered_number(self):
        number_op2 = FermionOperator(((2, 1), (2, 0)))
        self.assertTrue(number_op2.isclose(normal_ordered(number_op2)))

    def test_normal_ordered_number_reversed(self):
        n_term_rev2 = FermionOperator(((2, 0), (2, 1)))
        number_op2 = number_operator(3, 2)
        expected = FermionOperator(()) - number_op2
        self.assertTrue(normal_ordered(n_term_rev2).isclose(expected))

    def test_normal_ordered_offsite(self):
        op = FermionOperator(((3, 1), (2, 0)))
        self.assertTrue(op.isclose(normal_ordered(op)))

    def test_normal_ordered_offsite_reversed(self):
        op = FermionOperator(((3, 0), (2, 1)))
        expected = -FermionOperator(((2, 1), (3, 0)))
        self.assertTrue(expected.isclose(normal_ordered(op)))

    def test_normal_ordered_double_create(self):
        op = FermionOperator(((2, 0), (3, 1), (3, 1)))
        expected = FermionOperator((), 0.0)
        self.assertTrue(expected.isclose(normal_ordered(op)))

    def test_normal_ordered_double_create_separated(self):
        op = FermionOperator(((3, 1), (2, 0), (3, 1)))
        expected = FermionOperator((), 0.0)
        self.assertTrue(expected.isclose(normal_ordered(op)))

    def test_normal_ordered_multi(self):
        op = FermionOperator(((2, 0), (1, 1), (2, 1)))
        expected = (-FermionOperator(((2, 1), (1, 1), (2, 0))) -
                    FermionOperator(((1, 1),)))
        self.assertTrue(expected.isclose(normal_ordered(op)))

    def test_normal_ordered_triple(self):
        op_132 = FermionOperator(((1, 1), (3, 0), (2, 0)))
        op_123 = FermionOperator(((1, 1), (2, 0), (3, 0)))
        op_321 = FermionOperator(((3, 0), (2, 0), (1, 1)))

        self.assertTrue(op_132.isclose(normal_ordered(-op_123)))
        self.assertTrue(op_132.isclose(normal_ordered(op_132)))
        self.assertTrue(op_132.isclose(normal_ordered(op_321)))

    def test_is_molecular_term_FermionOperator(self):
        op = FermionOperator()
        self.assertTrue(op.is_molecular_term())

    def test_is_molecular_term_number(self):
        op = number_operator(n_orbitals=5, orbital=3)
        self.assertTrue(op.is_molecular_term())

    def test_is_molecular_term_updown(self):
        op = FermionOperator(((2, 1), (4, 0)))
        self.assertTrue(op.is_molecular_term())

    def test_is_molecular_term_downup(self):
        op = FermionOperator(((2, 0), (4, 1)))
        self.assertTrue(op.is_molecular_term())

    def test_is_molecular_term_downup_badspin(self):
        op = FermionOperator(((2, 0), (3, 1)))
        self.assertFalse(op.is_molecular_term())

    def test_is_molecular_term_three(self):
        op = FermionOperator(((0, 1), (2, 1), (4, 0)))
        self.assertFalse(op.is_molecular_term())

    def test_is_molecular_term_out_of_order(self):
        op = FermionOperator(((0, 1), (2, 0), (1, 1), (3, 0)))
        self.assertTrue(op.is_molecular_term())

    def test_freeze_orbitals_nonvanishing(self):
        op = FermionOperator(((1, 1), (1, 0), (0, 1), (2, 0)))
        op_frozen = freeze_orbitals(op,[1])
        expected = FermionOperator(((0, 1), (1, 0)), -1)
        self.assertTrue(op_frozen.isclose(expected))

    def test_freeze_orbitals_vanishing(self):
        op = FermionOperator(((1, 1), (2, 0)))
        op_frozen = freeze_orbitals(op, [], [2])
        self.assertEquals(len(op_frozen.terms), 0)
