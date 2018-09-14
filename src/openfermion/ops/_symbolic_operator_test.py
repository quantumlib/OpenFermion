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
import copy
import unittest
import warnings

import numpy
from openfermion.config import EQ_TOLERANCE
from openfermion.utils._testing_utils import EqualsTester

from openfermion.ops._symbolic_operator import SymbolicOperator


class DummyOperator1(SymbolicOperator):
    """Subclass of SymbolicOperator created for testing purposes."""

    @property
    def actions(self):
        """The allowed actions."""
        return (1, 0)

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('^', '')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return False

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return False


class DummyOperator2(SymbolicOperator):
    """Subclass of SymbolicOperator created for testing purposes."""

    @property
    def actions(self):
        """The allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return True

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return True


class GeneralTest(unittest.TestCase):
    """General tests."""

    def test_symbolic_operator_is_abstract_cant_instantiate(self):
        with self.assertRaises(TypeError):
            _ = SymbolicOperator()

    def test_symbolic_operator_constant(self):
        op = DummyOperator1((), 1.723)
        self.assertEqual(op.constant, 1.723)

        op = DummyOperator1('1^ 4', 0.182)
        self.assertEqual(op.constant, 0.0)

    def test_init_single_factor(self):
        """Test initialization of the form DummyOperator((index, action))."""
        equals_tester = EqualsTester(self)

        group_1 = [DummyOperator1((3, 0)), DummyOperator1(((3, 0),))]
        group_2 = [DummyOperator2((5, 'X')), DummyOperator2(((5, 'X'),))]
        group_3 = [DummyOperator2((5, 'X'), .5),
                   DummyOperator2(((5, 'X'),), .5)]

        equals_tester.add_equality_group(*group_1)
        equals_tester.add_equality_group(*group_2)
        equals_tester.add_equality_group(*group_3)

    def test_eq_and_ne(self):
        """Test == and !=."""
        equals_tester = EqualsTester(self)

        zeros_1 = [DummyOperator1(),
                   DummyOperator1('1^ 0', 0.),
                   DummyOperator1('1^ 0', -1j) * 0]

        zeros_2 = [DummyOperator2(),
                   DummyOperator2(((1, 'Y'), (0, 'X')), 0.),
                   DummyOperator2(((1, 'Y'), (0, 'X')), -1j) * 0]

        different_ops_1 = [DummyOperator1(((1, 0),), -0.1j),
                           DummyOperator1(((1, 1),), -0.1j),
                           (DummyOperator1(((1, 0),), -0.1j) +
                            DummyOperator1(((1, 1),), -0.1j))]

        different_ops_2 = [DummyOperator2(((1, 'Y'),), -0.1j),
                           DummyOperator2(((1, 'X'),), -0.1j),
                           (DummyOperator2(((1, 'Y'),), -0.1j) +
                            DummyOperator2(((2, 'Y'),), -0.1j))]

        equals_tester.add_equality_group(*zeros_1)
        equals_tester.add_equality_group(*zeros_2)
        for op in different_ops_1:
            equals_tester.add_equality_group(op)
        for op in different_ops_2:
            equals_tester.add_equality_group(op)

    def test_many_body_order(self):
        """Test computing the many-body order."""
        zero = DummyOperator1()
        identity = DummyOperator2(())

        op1 = DummyOperator1('0^ 3 5^ 6')
        op2 = op1 + DummyOperator1('8^ 3')
        op3 = op2 + DummyOperator1(u'1^ 2 3^ 4 5 ')

        op4 = DummyOperator2('X0 X1 Y3')
        op5 = op4 - DummyOperator2('Z0')
        op6 = op5 - DummyOperator2('Z1 Z2 Y3 Y4 Y9 Y10')
        op7 = op5 - DummyOperator2('Z1 Z2 Y3 Y4 Y9 Y10', EQ_TOLERANCE / 2.)

        self.assertEqual(zero.many_body_order(), 0)
        self.assertEqual(identity.many_body_order(), 0)
        self.assertEqual(op1.many_body_order(), 4)
        self.assertEqual(op2.many_body_order(), 4)
        self.assertEqual(op3.many_body_order(), 5)
        self.assertEqual(op4.many_body_order(), 3)
        self.assertEqual(op5.many_body_order(), 3)
        self.assertEqual(op6.many_body_order(), 6)
        self.assertEqual(op7.many_body_order(), 3)

    def test_iter(self):
        op1 = DummyOperator1('0^ 3 5^ 6')
        op2 = DummyOperator1('8^ 3')
        opsum = op1 + op2
        op_list = []
        for op_term in opsum:
            op_list.append(op_term)

        self.assertEqual(len(op_list), 2)
        self.assertEqual(op_list[0], op1)
        self.assertEqual(op_list[1], op2)


class SymbolicOperatorTest1(unittest.TestCase):
    """Test the subclass DummyOperator1."""

    def test_init_defaults(self):
        loc_op = DummyOperator1()
        self.assertEqual(len(loc_op.terms), 0)

    def test_init_tuple_real_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = 0.5
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_tuple_complex_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = 0.6j
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_tuple_npfloat64_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = numpy.float64(2.303)
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_tuple_npcomplex128_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = numpy.complex128(-1.123j + 43.7)
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_list_real_coefficient(self):
        loc_op = [(0, 1), (5, 0), (6, 1)]
        coefficient = 1. / 3
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_list_complex_coefficient(self):
        loc_op = [(0, 1), (5, 0), (6, 1)]
        coefficient = 2j / 3.
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_list_npfloat64_coefficient(self):
        loc_op = [(0, 1), (5, 0), (6, 1)]
        coefficient = numpy.float64(2.3037)
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_init_list_npcomplex128_coefficient(self):
        loc_op = [(0, 1), (5, 0), (6, 1)]
        coefficient = numpy.complex128(-1.1237j + 43.37)
        fermion_op = DummyOperator1(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[tuple(loc_op)], coefficient)

    def test_identity_is_multiplicative_identity(self):
        u = DummyOperator1.identity()
        f = DummyOperator1(((0, 1), (5, 0), (6, 1)), 0.6j)
        g = DummyOperator1(((0, 0), (5, 0), (6, 1)), 0.3j)
        h = f + g
        self.assertTrue(f == u * f)
        self.assertTrue(f == f * u)
        self.assertTrue(g == u * g)
        self.assertTrue(g == g * u)
        self.assertTrue(h == u * h)
        self.assertTrue(h == h * u)

        u *= h
        self.assertTrue(h == u)
        self.assertFalse(f == u)

        # Method always returns new instances.
        self.assertFalse(DummyOperator1.identity() == u)

    def test_zero_is_additive_identity(self):
        o = DummyOperator1.zero()
        f = DummyOperator1(((0, 1), (5, 0), (6, 1)), 0.6j)
        g = DummyOperator1(((0, 0), (5, 0), (6, 1)), 0.3j)
        h = f + g
        self.assertTrue(f == o + f)
        self.assertTrue(f == f + o)
        self.assertTrue(g == o + g)
        self.assertTrue(g == g + o)
        self.assertTrue(h == o + h)
        self.assertTrue(h == h + o)

        o += h
        self.assertTrue(h == o)
        self.assertFalse(f == o)

        # Method always returns new instances.
        self.assertFalse(DummyOperator1.zero() == o)

    def test_zero_is_multiplicative_nil(self):
        o = DummyOperator1.zero()
        u = DummyOperator1.identity()
        f = DummyOperator1(((0, 1), (5, 0), (6, 1)), 0.6j)
        g = DummyOperator1(((0, 0), (5, 0), (6, 1)), 0.3j)
        self.assertTrue(o == o * u)
        self.assertTrue(o == o * f)
        self.assertTrue(o == o * g)
        self.assertTrue(o == o * (f + g))

    def test_init_str(self):
        fermion_op = DummyOperator1('0^ 5 12^', -1.)
        correct = ((0, 1), (5, 0), (12, 1))
        self.assertIn(correct, fermion_op.terms)
        self.assertEqual(fermion_op.terms[correct], -1.0)

    def test_init_long_str(self):
        fermion_op = DummyOperator1(
                '(-2.0+3.0j) [0^ 1] +\n\n -1.0[ 2^ 3 ] - []', -1.)
        correct = \
            DummyOperator1('0^ 1', complex(2., -3.)) + \
            DummyOperator1('2^ 3', 1.) + \
            DummyOperator1('', 1.)
        self.assertEqual(len((fermion_op-correct).terms), 0)
        reparsed_op = DummyOperator1(str(fermion_op))
        self.assertEqual(len((fermion_op-reparsed_op).terms), 0)

        fermion_op = DummyOperator1('1.7 [3^ 2] - 8 [4^]')
        correct = DummyOperator1('3^ 2', 1.7) + DummyOperator1('4^', -8.)
        self.assertEqual(len((fermion_op-correct).terms), 0)

        fermion_op = DummyOperator1('-(2.3 + 1.7j) [3^ 2]')
        correct = DummyOperator1('3^ 2', complex(-2.3, -1.7))
        self.assertEqual(len((fermion_op-correct).terms), 0)

    def test_merges_multiple_whitespace(self):
        fermion_op = DummyOperator1('        \n ')
        self.assertEqual(fermion_op.terms, {(): 1})

    def test_init_str_identity(self):
        fermion_op = DummyOperator1('')
        self.assertIn((), fermion_op.terms)

    def test_init_bad_term(self):
        with self.assertRaises(ValueError):
            DummyOperator1(2)

    def test_init_bad_coefficient(self):
        with self.assertRaises(ValueError):
            DummyOperator1('0^', "0.5")

    def test_init_bad_action_str(self):
        with self.assertRaises(ValueError):
            DummyOperator1('0-')

    def test_init_bad_action_tuple(self):
        with self.assertRaises(ValueError):
            DummyOperator1(((0, 2),))

    def test_init_bad_tuple(self):
        with self.assertRaises(ValueError):
            DummyOperator1(((0, 1, 1),))

    def test_init_bad_str(self):
        with self.assertRaises(ValueError):
            DummyOperator1('^')

    def test_init_bad_mode_num(self):
        with self.assertRaises(ValueError):
            DummyOperator1('-1^')

    def test_init_invalid_tensor_factor(self):
        with self.assertRaises(ValueError):
            DummyOperator1(((-2, 1), (1, 0)))

    def test_DummyOperator1(self):
        op = DummyOperator1((), 3.)
        self.assertTrue(op == DummyOperator1(()) * 3.)

    def test_imul_inplace(self):
        fermion_op = DummyOperator1("1^")
        prev_id = id(fermion_op)
        fermion_op *= 3.
        self.assertEqual(id(fermion_op), prev_id)
        self.assertEqual(fermion_op.terms[((1, 1),)], 3.)

    def test_imul_scalar_real(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = 0.5
        fermion_op = DummyOperator1(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_scalar_complex(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = 0.6j
        fermion_op = DummyOperator1(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_scalar_npfloat64(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = numpy.float64(2.303)
        fermion_op = DummyOperator1(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_scalar_npcomplex128(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = numpy.complex128(-1.123j + 1.7911)
        fermion_op = DummyOperator1(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_fermion_op(self):
        op1 = DummyOperator1(((0, 1), (3, 0), (8, 1), (8, 0), (11, 1)), 3.j)
        op2 = DummyOperator1(((1, 1), (3, 1), (8, 0)), 0.5)
        op1 *= op2
        correct_term = ((0, 1), (3, 0), (8, 1), (8, 0), (11, 1),
                        (1, 1), (3, 1), (8, 0))
        self.assertEqual(len(op1.terms), 1)
        self.assertIn(correct_term, op1.terms)

    def test_imul_fermion_op_2(self):
        op3 = DummyOperator1(((1, 1), (0, 0)), -1j)
        op4 = DummyOperator1(((1, 0), (0, 1), (2, 1)), -1.5)
        op3 *= op4
        op4 *= op3
        self.assertIn(((1, 1), (0, 0), (1, 0), (0, 1), (2, 1)), op3.terms)
        self.assertEqual(op3.terms[((1, 1), (0, 0), (1, 0), (0, 1), (2, 1))],
                         1.5j)

    def test_imul_fermion_op_duplicate_term(self):
        op1 = DummyOperator1('1 2 3')
        op1 += DummyOperator1('1 2')
        op1 += DummyOperator1('1')

        op2 = DummyOperator1('3')
        op2 += DummyOperator1('2 3')

        op1 *= op2
        self.assertAlmostEqual(op1.terms[((1, 0), (2, 0), (3, 0))], 2.)

    def test_imul_bidir(self):
        op_a = DummyOperator1(((1, 1), (0, 0)), -1j)
        op_b = DummyOperator1(((1, 1), (0, 1), (2, 1)), -1.5)
        op_a *= op_b
        op_b *= op_a
        self.assertIn(((1, 1), (0, 0), (1, 1), (0, 1), (2, 1)), op_a.terms)
        self.assertEqual(op_a.terms[((1, 1), (0, 0), (1, 1), (0, 1), (2, 1))],
                         1.5j)
        self.assertIn(((1, 1), (0, 1), (2, 1),
                       (1, 1), (0, 0), (1, 1), (0, 1), (2, 1)), op_b.terms)
        self.assertEqual(op_b.terms[((1, 1), (0, 1), (2, 1),
                                     (1, 1), (0, 0),
                                     (1, 1), (0, 1), (2, 1))], -2.25j)

    def test_imul_bad_multiplier(self):
        op = DummyOperator1(((1, 1), (0, 1)), -1j)
        with self.assertRaises(TypeError):
            op *= "1"

    def test_mul_by_scalarzero(self):
        op = DummyOperator1(((1, 1), (0, 1)), -1j) * 0
        self.assertNotIn(((0, 1), (1, 1)), op.terms)
        self.assertIn(((1, 1), (0, 1)), op.terms)
        self.assertEqual(op.terms[((1, 1), (0, 1))], 0.0)

    def test_mul_bad_multiplier(self):
        op = DummyOperator1(((1, 1), (0, 1)), -1j)
        with self.assertRaises(TypeError):
            op = op * "0.5"

    def test_mul_out_of_place(self):
        op1 = DummyOperator1(((0, 1), (3, 1), (3, 0), (11, 1)), 3.j)
        op2 = DummyOperator1(((1, 1), (3, 1), (8, 0)), 0.5)
        op3 = op1 * op2
        correct_coefficient = 3.0j * 0.5
        correct_term = ((0, 1), (3, 1), (3, 0), (11, 1),
                        (1, 1), (3, 1), (8, 0))
        self.assertTrue(op1 == DummyOperator1(
            ((0, 1), (3, 1), (3, 0), (11, 1)), 3.j))
        self.assertTrue(op2 == DummyOperator1(((1, 1), (3, 1), (8, 0)), 0.5))
        self.assertTrue(op3 == DummyOperator1(correct_term,
                                              correct_coefficient))

    def test_mul_npfloat64(self):
        op = DummyOperator1(((1, 0), (3, 1)), 0.5)
        res = op * numpy.float64(0.5)
        self.assertTrue(res == DummyOperator1(((1, 0), (3, 1)), 0.5 * 0.5))

    def test_mul_multiple_terms(self):
        op = DummyOperator1(((1, 0), (8, 1)), 0.5)
        op += DummyOperator1(((1, 1), (9, 1)), 1.4j)
        res = op * op
        correct = DummyOperator1(((1, 0), (8, 1), (1, 0), (8, 1)), 0.5 ** 2)
        correct += (DummyOperator1(((1, 0), (8, 1), (1, 1), (9, 1)), 0.7j) +
                    DummyOperator1(((1, 1), (9, 1), (1, 0), (8, 1)), 0.7j))
        correct += DummyOperator1(((1, 1), (9, 1), (1, 1), (9, 1)), 1.4j ** 2)
        self.assertTrue(res == correct)

    def test_rmul_scalar_real(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = 0.5
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1 == res2)

    def test_rmul_scalar_complex(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = 0.6j
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1 == res2)

    def test_rmul_scalar_npfloat64(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = numpy.float64(2.303)
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1 == res2)

    def test_rmul_scalar_npcomplex128(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = numpy.complex128(-1.5j + 7.7)
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1 == res2)

    def test_rmul_bad_multiplier(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        with self.assertRaises(TypeError):
            op = "0.5" * op

    def test_truediv_and_div_real(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.5
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res == correct)
        # Test if done out of place
        self.assertTrue(op == original)

    def test_truediv_and_div_complex(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.6j
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res == correct)
        # Test if done out of place
        self.assertTrue(op == original)

    def test_truediv_and_div_npfloat64(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.float64(2.303)
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res == correct)
        # Test if done out of place
        self.assertTrue(op == original)

    def test_truediv_and_div_npcomplex128(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.complex128(566.4j + 0.3)
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res == correct)
        # Test if done out of place
        self.assertTrue(op == original)

    def test_truediv_bad_divisor(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        with self.assertRaises(TypeError):
            op = op / "0.5"

    def test_itruediv_and_idiv_real(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.5
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op == correct)
        # Test if done in-place
        self.assertFalse(op == original)

    def test_itruediv_and_idiv_complex(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.6j
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op == correct)
        # Test if done in-place
        self.assertFalse(op == original)

    def test_itruediv_and_idiv_npfloat64(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.float64(2.3030)
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op == correct)
        # Test if done in-place
        self.assertFalse(op == original)

    def test_itruediv_and_idiv_npcomplex128(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.complex128(12.3 + 7.4j)
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op == correct)
        # Test if done in-place
        self.assertFalse(op == original)

    def test_itruediv_bad_divisor(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        with self.assertRaises(TypeError):
            op /= "0.5"

    def test_iadd_different_term(self):
        term_a = ((1, 1), (3, 0), (8, 1))
        term_b = ((1, 1), (3, 1), (8, 0))
        a = DummyOperator1(term_a, 1.0)
        a += DummyOperator1(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], 0.5)
        a += DummyOperator1(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], 1.0)

    def test_iadd_bad_addend(self):
        op = DummyOperator1((), 1.0)
        with self.assertRaises(TypeError):
            op += "0.5"

    def test_add(self):
        term_a = ((1, 1), (3, 0), (8, 1))
        term_b = ((1, 0), (3, 0), (8, 1))
        a = DummyOperator1(term_a, 1.0)
        b = DummyOperator1(term_b, 0.5)
        res = a + b + b
        self.assertEqual(len(res.terms), 2)
        self.assertEqual(res.terms[term_a], 1.0)
        self.assertEqual(res.terms[term_b], 1.0)
        # Test out of place
        self.assertTrue(a == DummyOperator1(term_a, 1.0))
        self.assertTrue(b == DummyOperator1(term_b, 0.5))

    def test_add_bad_addend(self):
        op = DummyOperator1((), 1.0)
        with self.assertRaises(TypeError):
            _ = op + "0.5"

    def test_sub(self):
        term_a = ((1, 1), (3, 1), (8, 1))
        term_b = ((1, 0), (3, 1), (8, 1))
        a = DummyOperator1(term_a, 1.0)
        b = DummyOperator1(term_b, 0.5)
        res = a - b
        self.assertEqual(len(res.terms), 2)
        self.assertEqual(res.terms[term_a], 1.0)
        self.assertEqual(res.terms[term_b], -0.5)
        res2 = b - a
        self.assertEqual(len(res2.terms), 2)
        self.assertEqual(res2.terms[term_a], -1.0)
        self.assertEqual(res2.terms[term_b], 0.5)

    def test_sub_bad_subtrahend(self):
        op = DummyOperator1((), 1.0)
        with self.assertRaises(TypeError):
            _ = op - "0.5"

    def test_isub_different_term(self):
        term_a = ((1, 1), (3, 1), (8, 0))
        term_b = ((1, 0), (3, 1), (8, 1))
        a = DummyOperator1(term_a, 1.0)
        a -= DummyOperator1(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], -0.5)
        a -= DummyOperator1(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], -1.0)

    def test_isub_bad_addend(self):
        op = DummyOperator1((), 1.0)
        with self.assertRaises(TypeError):
            op -= "0.5"

    def test_neg(self):
        op = DummyOperator1(((1, 1), (3, 1), (8, 1)), 0.5)
        _ = -op
        # out of place
        self.assertTrue(op == DummyOperator1(((1, 1), (3, 1), (8, 1)), 0.5))
        correct = -1.0 * op
        self.assertTrue(correct == -op)

    def test_pow_square_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = DummyOperator1(ops, coeff)
        squared = term ** 2
        expected = DummyOperator1(ops + ops, coeff ** 2)
        self.assertTrue(squared == term * term)
        self.assertTrue(squared == expected)

    def test_pow_zero_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = DummyOperator1(ops, coeff)
        zerod = term ** 0
        expected = DummyOperator1(())
        self.assertTrue(expected == zerod)

    def test_pow_one_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = DummyOperator1(ops, coeff)
        self.assertTrue(term == term ** 1)

    def test_pow_high_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = DummyOperator1(ops, coeff)
        high = term ** 10
        expected = DummyOperator1(ops * 10, coeff ** 10)
        self.assertTrue(expected == high)

    def test_pow_neg_error(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator1() ** -1

    def test_pow_nonint_error(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator1('3 2^') ** 0.5

    def test_compress_terms(self):
        op = (DummyOperator1('3^ 1', 0.3 + 3e-11j) +
              DummyOperator1('2^ 3', 5e-10) +
              DummyOperator1('1^ 3', 1e-3))
        op_compressed = (DummyOperator1('3^ 1', 0.3) +
                         DummyOperator1('1^ 3', 1e-3))
        op.compress(1e-7)
        self.assertTrue(op_compressed == op)

    def test_str(self):

        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        self.assertEqual(str(op), "0.5 [1^ 3 8^]")

        op = DummyOperator1((), 2)
        self.assertEqual(str(op), "2 []")

        op = DummyOperator1()
        self.assertEqual(str(op), "0")

        op = (DummyOperator1(((3, 1), (4, 1), (5, 0)), 1.0) +
              DummyOperator1(((3, 1), (4, 1), (4, 0)), 2.0) +
              DummyOperator1(((2, 1), (4, 1), (5, 0)), 1.0) +
              DummyOperator1(((3, 0), (2, 1), (1, 1)), 2.0) +
              DummyOperator1(((3, 0), (2, 0), (1, 1)), 2.0))
        self.assertEqual(str(op).strip(), """
1.0 [2^ 4^ 5] +
2.0 [3 2 1^] +
2.0 [3 2^ 1^] +
2.0 [3^ 4^ 4] +
1.0 [3^ 4^ 5]
""".strip())

        op = (DummyOperator1(((3, 1), (4, 1), (5, 0)), 0.0) +
              DummyOperator1(((3, 1), (4, 1), (4, 0)), 2.0))
        self.assertEqual(str(op).strip(), """
2.0 [3^ 4^ 4]
""".strip())

    def test_rep(self):
        op = DummyOperator1(((1, 1), (3, 0), (8, 1)), 0.5)
        # Not necessary, repr could do something in addition
        self.assertEqual(repr(op), str(op))


class SymbolicOperatorTest2(unittest.TestCase):
    """Test the subclass DummyOperator2."""
    def test_init_defaults(self):
        loc_op = DummyOperator2()
        self.assertTrue(len(loc_op.terms) == 0)

    def test_init_tuple(self):
        coefficient = 0.5
        loc_op = ((0, 'X'), (5, 'Y'), (6, 'Z'))
        qubit_op = DummyOperator2(loc_op, coefficient)
        self.assertTrue(len(qubit_op.terms) == 1)
        self.assertTrue(qubit_op.terms[loc_op] == coefficient)

    def test_init_list(self):
        coefficient = 0.6j
        loc_op = [(0, 'X'), (5, 'Y'), (6, 'Z')]
        qubit_op = DummyOperator2(loc_op, coefficient)
        self.assertTrue(len(qubit_op.terms) == 1)
        self.assertTrue(qubit_op.terms[tuple(loc_op)] == coefficient)

    def test_init_str(self):
        qubit_op = DummyOperator2('X0 Y5 Z12', -1.)
        correct = ((0, 'X'), (5, 'Y'), (12, 'Z'))
        self.assertTrue(correct in qubit_op.terms)
        self.assertTrue(qubit_op.terms[correct] == -1.0)

    def test_init_long_str(self):
        qubit_op = DummyOperator2(
                '(-2.0+3.0j) [X0 Y1] +\n\n -1.0[ X2 Y3 ] - []', -1.)
        correct = \
            DummyOperator2('X0 Y1', complex(2., -3.)) + \
            DummyOperator2('X2 Y3', 1.) + \
            DummyOperator2('', 1.)
        self.assertEqual(len((qubit_op-correct).terms), 0)
        reparsed_op = DummyOperator2(str(qubit_op))
        self.assertEqual(len((qubit_op-reparsed_op).terms), 0)

        qubit_op = DummyOperator2('[X0 X1] + [Y0 Y1]')
        correct = DummyOperator2('X0 X1') + DummyOperator2('Y0 Y1')
        self.assertTrue(qubit_op == correct)
        self.assertTrue(qubit_op == DummyOperator2(str(qubit_op)))

    def test_init_str_identity(self):
        qubit_op = DummyOperator2('', 2.)
        self.assertTrue(len(qubit_op.terms) == 1)
        self.assertTrue(() in qubit_op.terms)
        self.assertAlmostEqual(qubit_op.terms[()], 2.)

    def test_init_bad_term(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2(2)

    def test_init_bad_coefficient(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2('X0', "0.5")

    def test_init_bad_action(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2('Q0')

    def test_init_bad_action_in_tuple(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2(((1, 'Q'),))

    def test_init_bad_qubit_num_in_tuple(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2((("1", 'X'),))

    def test_init_bad_tuple(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2(((0, 1, 'X'),))

    def test_init_bad_str(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2('X')

    def test_init_bad_qubit_num(self):
        with self.assertRaises(ValueError):
            _ = DummyOperator2('X-1')

    def test_compress(self):
        a = DummyOperator2('X0', .9e-12)
        self.assertTrue(len(a.terms) == 1)
        a.compress()
        self.assertTrue(len(a.terms) == 0)
        a = DummyOperator2('X0', 1. + 1j)
        a.compress(.5)
        self.assertTrue(len(a.terms) == 1)
        for term in a.terms:
            self.assertTrue(a.terms[term] == 1. + 1j)
        a = DummyOperator2('X0', 1.1 + 1j)
        a.compress(1.)
        self.assertTrue(len(a.terms) == 1)
        for term in a.terms:
            self.assertTrue(a.terms[term] == 1.1)
        a = DummyOperator2('X0', 1.1 + 1j) + DummyOperator2('X1', 1.e-6j)
        a.compress()
        self.assertTrue(len(a.terms) == 2)
        for term in a.terms:
            self.assertTrue(isinstance(a.terms[term], complex))
        a.compress(1.e-5)
        self.assertTrue(len(a.terms) == 1)
        for term in a.terms:
            self.assertTrue(isinstance(a.terms[term], complex))
        a.compress(1.)
        self.assertTrue(len(a.terms) == 1)
        for term in a.terms:
            self.assertTrue(isinstance(a.terms[term], float))

    def test_rmul_scalar(self):
        multiplier = 0.5
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1 == res2)

    def test_rmul_bad_multiplier(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        with self.assertRaises(TypeError):
            op = "0.5" * op

    def test_truediv_and_div(self):
        divisor = 0.6j
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        op2 = copy.deepcopy(op)
        original = copy.deepcopy(op)
        res = op / divisor
        res2 = op2.__div__(divisor)  # To test python 2 version as well
        correct = op * (1. / divisor)
        self.assertTrue(res == correct)
        self.assertTrue(res2 == correct)
        # Test if done out of place
        self.assertTrue(op == original)
        self.assertTrue(op2 == original)

    def test_truediv_bad_divisor(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        with self.assertRaises(TypeError):
            op = op / "0.5"

    def test_itruediv_and_idiv(self):
        divisor = 2
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        op2 = copy.deepcopy(op)
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        op2.__idiv__(divisor)  # To test python 2 version as well
        self.assertTrue(op == correct)
        self.assertTrue(op2 == correct)
        # Test if done in-place
        self.assertTrue(not op == original)
        self.assertTrue(not op2 == original)

    def test_itruediv_bad_divisor(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        with self.assertRaises(TypeError):
            op /= "0.5"

    def test_iadd_cancellation(self):
        term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        term_b = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        a = DummyOperator2(term_a, 1.0)
        a += DummyOperator2(term_b, -1.0)
        self.assertTrue(len(a.terms) == 0)

    def test_iadd_different_term(self):
        term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
        a = DummyOperator2(term_a, 1.0)
        a += DummyOperator2(term_b, 0.5)
        self.assertTrue(len(a.terms) == 2)
        self.assertAlmostEqual(a.terms[term_a], 1.0)
        self.assertAlmostEqual(a.terms[term_b], 0.5)
        a += DummyOperator2(term_b, 0.5)
        self.assertTrue(len(a.terms) == 2)
        self.assertAlmostEqual(a.terms[term_a], 1.0)
        self.assertAlmostEqual(a.terms[term_b], 1.0)

    def test_iadd_bad_addend(self):
        op = DummyOperator2((), 1.0)
        with self.assertRaises(TypeError):
            op += "0.5"

    def test_add(self):
        term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
        a = DummyOperator2(term_a, 1.0)
        b = DummyOperator2(term_b, 0.5)
        res = a + b + b
        self.assertTrue(len(res.terms) == 2)
        self.assertAlmostEqual(res.terms[term_a], 1.0)
        self.assertAlmostEqual(res.terms[term_b], 1.0)
        # Test out of place
        self.assertTrue(a == DummyOperator2(term_a, 1.0))
        self.assertTrue(b == DummyOperator2(term_b, 0.5))

    def test_add_bad_addend(self):
        op = DummyOperator2((), 1.0)
        with self.assertRaises(TypeError):
            op = op + "0.5"

    def test_sub(self):
        term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
        a = DummyOperator2(term_a, 1.0)
        b = DummyOperator2(term_b, 0.5)
        res = a - b
        self.assertTrue(len(res.terms) == 2)
        self.assertAlmostEqual(res.terms[term_a], 1.0)
        self.assertAlmostEqual(res.terms[term_b], -0.5)
        res2 = b - a
        self.assertTrue(len(res2.terms) == 2)
        self.assertAlmostEqual(res2.terms[term_a], -1.0)
        self.assertAlmostEqual(res2.terms[term_b], 0.5)

    def test_sub_bad_subtrahend(self):
        op = DummyOperator2((), 1.0)
        with self.assertRaises(TypeError):
            op = op - "0.5"

    def test_isub_different_term(self):
        term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
        a = DummyOperator2(term_a, 1.0)
        a -= DummyOperator2(term_b, 0.5)
        self.assertTrue(len(a.terms) == 2)
        self.assertAlmostEqual(a.terms[term_a], 1.0)
        self.assertAlmostEqual(a.terms[term_b], -0.5)
        a -= DummyOperator2(term_b, 0.5)
        self.assertTrue(len(a.terms) == 2)
        self.assertAlmostEqual(a.terms[term_a], 1.0)
        self.assertAlmostEqual(a.terms[term_b], -1.0)

    def test_isub_bad_addend(self):
        op = DummyOperator2((), 1.0)
        with self.assertRaises(TypeError):
            op -= "0.5"

    def test_neg(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        # out of place
        self.assertTrue(op == DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')),
                                             0.5))
        correct = -1.0 * op
        self.assertTrue(correct == -op)

    def test_str(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        self.assertEqual(str(op), "0.5 [X1 Y3 Z8]")
        op2 = DummyOperator2((), 2)
        self.assertEqual(str(op2), "2 []")
        op3 =  (DummyOperator2(((3, 'X'), (4, 'Z'), (5, 'Y')), 3.0) +
                DummyOperator2(((1, 'X'), (4, 'Z'), (4, 'Z')), 2.0) +
                DummyOperator2(((2, 'Z'), (4, 'Z'), (5, 'X')), 1.0) +
                DummyOperator2(((3, 'Y'), (2, 'Y'), (1, 'Z')), 2.0) +
                DummyOperator2(((3, 'Y'), (2, 'Y'), (1, 'Y')), 2.0))
        self.assertEqual(str(op3).strip(), """
2.0 [X1 Z4 Z4] +
2.0 [Y1 Y2 Y3] +
2.0 [Z1 Y2 Y3] +
1.0 [Z2 Z4 X5] +
3.0 [X3 Z4 Y5]
""".strip())

    def test_str_empty(self):
        op = DummyOperator2()
        self.assertEqual(str(op), '0')

    def test_str_out_of_order(self):
        op = DummyOperator2(((3, 'Y'), (1, 'X'), (8, 'Z')), 0.5)
        self.assertEqual(str(op), '0.5 [X1 Y3 Z8]')

    def test_str_multiple_terms(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        op += DummyOperator2(((1, 'Y'), (3, 'Y'), (8, 'Z')), 0.6)
        self.assertTrue((str(op) == "0.5 [X1 Y3 Z8] +\n0.6 [Y1 Y3 Z8]" or
                         str(op) == "0.6 [Y1 Y3 Z8] +\n0.5 [X1 Y3 Z8]"))
        op2 = DummyOperator2((), 2)
        self.assertEqual(str(op2), "2 []")

    def test_rep(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        # Not necessary, repr could do something in addition
        self.assertEqual(repr(op), str(op))

    def test_norm(self):
        op = DummyOperator2(((1, 'X'), (3, 'Y'), (8, 'Z')), 1)
        op += DummyOperator2(((2, 'Z'), (3, 'Y')), 1)
        self.assertAlmostEqual(op.induced_norm(2), numpy.sqrt(2.))

    def test_tracenorm_zero(self):
        op = DummyOperator2()
        self.assertFalse(op.induced_norm())


class DeprecatedFunctionsTest(unittest.TestCase):
    """Tests for deprecated functions."""

    def test_warnings(self):
        """Test that warnings are raised appropriately."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            op1 = DummyOperator1()
            op1.isclose(op1)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertTrue("deprecated" in str(w[0].message))

    def test_isclose_zero_terms_1(self):
        op = DummyOperator1('1^ 0', -1j) * 0
        self.assertTrue(op == DummyOperator1())
        self.assertTrue(DummyOperator1() == op)

    def test_isclose_different_terms_1(self):
        a = DummyOperator1(((1, 0),), -0.1j)
        b = DummyOperator1(((1, 1),), -0.1j)
        self.assertFalse(a == b)

    def test_isclose_different_num_terms_1(self):
        a = DummyOperator1(((1, 0),), -0.1j)
        a += DummyOperator1(((1, 1),), -0.1j)
        b = DummyOperator1(((1, 0),), -0.1j)
        self.assertFalse(b == a)
        self.assertFalse(a == b)

    def test_isclose_zero_terms_2(self):
        op = DummyOperator2(((1, 'Y'), (0, 'X')), -1j) * 0
        self.assertTrue(op == DummyOperator2())
        self.assertTrue(DummyOperator2((), 0.0) == op)

    def test_isclose_different_terms_2(self):
        a = DummyOperator2(((1, 'Y'),), -0.1j)
        b = DummyOperator2(((1, 'X'),), -0.1j)
        self.assertTrue(not a == b)
        self.assertTrue(not b == a)

    def test_isclose_different_num_terms_2(self):
        a = DummyOperator2(((1, 'Y'),), -0.1j)
        a += DummyOperator2(((2, 'Y'),), -0.1j)
        b = DummyOperator2(((1, 'X'),), -0.1j)
        self.assertTrue(not b == a)
        self.assertTrue(not a == b)
