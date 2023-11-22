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
"""Tests for term_reordering."""

import unittest
import itertools
import numpy

from openfermion.hamiltonians import number_operator
from openfermion.ops.operators import FermionOperator, BosonOperator, QuadOperator
from openfermion.transforms.opconversions import jordan_wigner, get_fermion_operator
from openfermion.testing.testing_utils import random_interaction_operator
from openfermion.utils import up_then_down

from openfermion.transforms.opconversions.term_reordering import (
    normal_ordered,
    chemist_ordered,
    reorder,
)


class ChemistOrderingTest(unittest.TestCase):
    def test_convert_forward_back(self):
        n_qubits = 6
        random_operator = get_fermion_operator(random_interaction_operator(n_qubits))
        chemist_operator = chemist_ordered(random_operator)
        normalized_chemist = normal_ordered(chemist_operator)
        difference = normalized_chemist - normal_ordered(random_operator)
        self.assertAlmostEqual(0.0, difference.induced_norm())

    def test_exception(self):
        n_qubits = 6
        random_operator = get_fermion_operator(random_interaction_operator(n_qubits))
        bad_term = ((2, 1), (3, 1))
        random_operator += FermionOperator(bad_term)
        with self.assertRaises(TypeError):
            chemist_ordered(random_operator)

    def test_form(self):
        n_qubits = 6
        random_operator = get_fermion_operator(random_interaction_operator(n_qubits))
        chemist_operator = chemist_ordered(random_operator)
        for term, _ in chemist_operator.terms.items():
            if len(term) == 2 or not len(term):
                pass
            else:
                self.assertTrue(term[0][1])
                self.assertTrue(term[2][1])
                self.assertFalse(term[1][1])
                self.assertFalse(term[3][1])
                self.assertTrue(term[0][0] > term[2][0])
                self.assertTrue(term[1][0] > term[3][0])


class TestNormalOrdering(unittest.TestCase):
    def test_boson_single_term(self):
        op = BosonOperator('4 3 2 1') + BosonOperator('3 2')
        self.assertTrue(op == normal_ordered(op))

    def test_boson_two_term(self):
        op_b = BosonOperator(((2, 0), (4, 0), (2, 1)), 88.0)
        normal_ordered_b = normal_ordered(op_b)
        expected = BosonOperator(((4, 0),), 88.0) + BosonOperator(((2, 1), (4, 0), (2, 0)), 88.0)
        self.assertTrue(normal_ordered_b == expected)

    def test_boson_number(self):
        number_op2 = BosonOperator(((2, 1), (2, 0)))
        self.assertTrue(number_op2 == normal_ordered(number_op2))

    def test_boson_number_reversed(self):
        n_term_rev2 = BosonOperator(((2, 0), (2, 1)))
        number_op2 = number_operator(3, 2, parity=1)
        expected = BosonOperator(()) + number_op2
        self.assertTrue(normal_ordered(n_term_rev2) == expected)

    def test_boson_offsite(self):
        op = BosonOperator(((3, 1), (2, 0)))
        self.assertTrue(op == normal_ordered(op))

    def test_boson_offsite_reversed(self):
        op = BosonOperator(((3, 0), (2, 1)))
        expected = BosonOperator(((2, 1), (3, 0)))
        self.assertTrue(expected == normal_ordered(op))

    def test_boson_multi(self):
        op = BosonOperator(((2, 0), (1, 1), (2, 1)))
        expected = BosonOperator(((2, 1), (1, 1), (2, 0))) + BosonOperator(((1, 1),))
        self.assertTrue(expected == normal_ordered(op))

    def test_boson_triple(self):
        op_132 = BosonOperator(((1, 1), (3, 0), (2, 0)))
        op_123 = BosonOperator(((1, 1), (2, 0), (3, 0)))
        op_321 = BosonOperator(((3, 0), (2, 0), (1, 1)))

        self.assertTrue(op_132 == normal_ordered(op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_fermion_single_term(self):
        op = FermionOperator('4 3 2 1') + FermionOperator('3 2')
        self.assertTrue(op == normal_ordered(op))

    def test_fermion_two_term(self):
        op_b = FermionOperator(((2, 0), (4, 0), (2, 1)), -88.0)
        normal_ordered_b = normal_ordered(op_b)
        expected = FermionOperator(((4, 0),), 88.0) + FermionOperator(
            ((2, 1), (4, 0), (2, 0)), 88.0
        )
        self.assertTrue(normal_ordered_b == expected)

    def test_fermion_number(self):
        number_op2 = FermionOperator(((2, 1), (2, 0)))
        self.assertTrue(number_op2 == normal_ordered(number_op2))

    def test_fermion_number_reversed(self):
        n_term_rev2 = FermionOperator(((2, 0), (2, 1)))
        number_op2 = number_operator(3, 2)
        expected = FermionOperator(()) - number_op2
        self.assertTrue(normal_ordered(n_term_rev2) == expected)

    def test_fermion_offsite(self):
        op = FermionOperator(((3, 1), (2, 0)))
        self.assertTrue(op == normal_ordered(op))

    def test_fermion_offsite_reversed(self):
        op = FermionOperator(((3, 0), (2, 1)))
        expected = -FermionOperator(((2, 1), (3, 0)))
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_double_create(self):
        op = FermionOperator(((2, 0), (3, 1), (3, 1)))
        expected = FermionOperator((), 0.0)
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_double_create_separated(self):
        op = FermionOperator(((3, 1), (2, 0), (3, 1)))
        expected = FermionOperator((), 0.0)
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_multi(self):
        op = FermionOperator(((2, 0), (1, 1), (2, 1)))
        expected = -FermionOperator(((2, 1), (1, 1), (2, 0))) - FermionOperator(((1, 1),))
        self.assertTrue(expected == normal_ordered(op))

    def test_fermion_triple(self):
        op_132 = FermionOperator(((1, 1), (3, 0), (2, 0)))
        op_123 = FermionOperator(((1, 1), (2, 0), (3, 0)))
        op_321 = FermionOperator(((3, 0), (2, 0), (1, 1)))

        self.assertTrue(op_132 == normal_ordered(-op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_quad_single_term(self):
        op = QuadOperator('p4 p3 p2 p1') + QuadOperator('p3 p2')
        self.assertTrue(op == normal_ordered(op))

        op = QuadOperator('q0 p0') - QuadOperator('p0 q0')
        expected = QuadOperator('', 2.0j)
        self.assertTrue(expected == normal_ordered(op, hbar=2.0))

    def test_quad_two_term(self):
        op_b = QuadOperator('p0 q0 p3', 88.0)
        normal_ordered_b = normal_ordered(op_b, hbar=2)
        expected = QuadOperator('p3', -88.0 * 2j) + QuadOperator('q0 p0 p3', 88.0)
        self.assertTrue(normal_ordered_b == expected)

    def test_quad_offsite(self):
        op = QuadOperator(((3, 'p'), (2, 'q')))
        self.assertTrue(op == normal_ordered(op))

    def test_quad_offsite_reversed(self):
        op = QuadOperator(((3, 'q'), (2, 'p')))
        expected = QuadOperator(((2, 'p'), (3, 'q')))
        self.assertTrue(expected == normal_ordered(op))

    def test_quad_triple(self):
        op_132 = QuadOperator(((1, 'p'), (3, 'q'), (2, 'q')))
        op_123 = QuadOperator(((1, 'p'), (2, 'q'), (3, 'q')))
        op_321 = QuadOperator(((3, 'q'), (2, 'q'), (1, 'p')))

        self.assertTrue(op_132 == normal_ordered(op_123))
        self.assertTrue(op_132 == normal_ordered(op_132))
        self.assertTrue(op_132 == normal_ordered(op_321))

    def test_interaction_operator(self):
        for n_orbitals, real, _ in itertools.product((1, 2, 5), (True, False), range(5)):
            operator = random_interaction_operator(n_orbitals, real=real)
            normal_ordered_operator = normal_ordered(operator)
            expected_qubit_operator = jordan_wigner(operator)
            actual_qubit_operator = jordan_wigner(normal_ordered_operator)
            assert expected_qubit_operator == actual_qubit_operator
            two_body_tensor = normal_ordered_operator.two_body_tensor
            n_orbitals = len(two_body_tensor)
            ones = numpy.ones((n_orbitals,) * 2)
            triu = numpy.triu(ones, 1)
            shape = (n_orbitals**2, 1)
            mask = (
                triu.reshape(shape) * ones.reshape(shape[::-1])
                + ones.reshape(shape) * triu.reshape(shape[::-1])
            ).reshape((n_orbitals,) * 4)
            assert numpy.allclose(mask * two_body_tensor, numpy.zeros((n_orbitals,) * 4))
            for term in normal_ordered_operator:
                order = len(term) // 2
                left_term, right_term = term[:order], term[order:]
                assert all(i[1] == 1 for i in left_term)
                assert all(i[1] == 0 for i in right_term)
                assert left_term == tuple(sorted(left_term, reverse=True))
                assert right_term == tuple(sorted(right_term, reverse=True))

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            _ = normal_ordered(1)


class TestReorder(unittest.TestCase):
    def test_reorder(self):
        def shift_by_one(x, y):
            return (x + 1) % y

        operator = FermionOperator('1^ 2^ 3 4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms, {((2, 1), (3, 1), (4, 0), (0, 0)): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms, {((0, 1), (1, 1), (2, 0), (3, 0)): -3.17})

    def test_reorder_boson(self):
        shift_by_one = lambda x, y: (x + 1) % y
        operator = BosonOperator('1^ 2^ 3 4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms, {((0, 0), (2, 1), (3, 1), (4, 0)): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms, {((0, 1), (1, 1), (2, 0), (3, 0)): -3.17})

    def test_reorder_quad(self):
        shift_by_one = lambda x, y: (x + 1) % y
        operator = QuadOperator('q1 q2 p3 p4', -3.17)
        reordered = reorder(operator, shift_by_one)
        self.assertEqual(reordered.terms, {((0, 'p'), (2, 'q'), (3, 'q'), (4, 'p')): -3.17})
        reordered = reorder(operator, shift_by_one, reverse=True)
        self.assertEqual(reordered.terms, {((0, 'q'), (1, 'q'), (2, 'p'), (3, 'p')): -3.17})

    def test_up_then_down(self):
        for LadderOp in (FermionOperator, BosonOperator):
            operator = LadderOp('1^ 2^ 3 4', -3.17)
            reordered = reorder(operator, up_then_down)
            reordered = reorder(reordered, up_then_down, reverse=True)

            self.assertEqual(reordered.terms, operator.terms)
            self.assertEqual(up_then_down(6, 8), 3)
            self.assertEqual(up_then_down(3, 8), 5)
