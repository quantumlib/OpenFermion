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
                                               hermitian_conjugated,
                                               normal_ordered,
                                               number_operator)


class FermionOperatorTest(unittest.TestCase):

    def test_init_defaults(self):
        loc_op = FermionOperator()
        self.assertEqual(len(loc_op.terms), 0)

    def test_init_tuple_real_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = 0.5
        fermion_op = FermionOperator(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[loc_op], coefficient)

    def test_init_tuple_complex_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = 0.6j
        fermion_op = FermionOperator(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[loc_op], coefficient)

    def test_init_tuple_npfloat64_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = numpy.float64(2.303)
        fermion_op = FermionOperator(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[loc_op], coefficient)

    def test_init_tuple_npcomplex128_coefficient(self):
        loc_op = ((0, 1), (5, 0), (6, 1))
        coefficient = numpy.complex128(-1.123j + 43.7)
        fermion_op = FermionOperator(loc_op, coefficient)
        self.assertEqual(len(fermion_op.terms), 1)
        self.assertEqual(fermion_op.terms[loc_op], coefficient)

    def test_identity_is_multiplicative_identity(self):
        u = FermionOperator.identity()
        f = FermionOperator(((0, 1), (5, 0), (6, 1)), 0.6j)
        g = FermionOperator(((0, 0), (5, 0), (6, 1)), 0.3j)
        h = f + g
        self.assertTrue(f.isclose(u * f))
        self.assertTrue(f.isclose(f * u))
        self.assertTrue(g.isclose(u * g))
        self.assertTrue(g.isclose(g * u))
        self.assertTrue(h.isclose(u * h))
        self.assertTrue(h.isclose(h * u))

        u *= h
        self.assertTrue(h.isclose(u))
        self.assertFalse(f.isclose(u))

        # Method always returns new instances.
        self.assertFalse(FermionOperator.identity().isclose(u))

    def test_zero_is_additive_identity(self):
        o = FermionOperator.zero()
        f = FermionOperator(((0, 1), (5, 0), (6, 1)), 0.6j)
        g = FermionOperator(((0, 0), (5, 0), (6, 1)), 0.3j)
        h = f + g
        self.assertTrue(f.isclose(o + f))
        self.assertTrue(f.isclose(f + o))
        self.assertTrue(g.isclose(o + g))
        self.assertTrue(g.isclose(g + o))
        self.assertTrue(h.isclose(o + h))
        self.assertTrue(h.isclose(h + o))

        o += h
        self.assertTrue(h.isclose(o))
        self.assertFalse(f.isclose(o))

        # Method always returns new instances.
        self.assertFalse(FermionOperator.zero().isclose(o))

    def test_zero_is_multiplicative_nil(self):
        o = FermionOperator.zero()
        u = FermionOperator.identity()
        f = FermionOperator(((0, 1), (5, 0), (6, 1)), 0.6j)
        g = FermionOperator(((0, 0), (5, 0), (6, 1)), 0.3j)
        self.assertTrue(o.isclose(o * u))
        self.assertTrue(o.isclose(o * f))
        self.assertTrue(o.isclose(o * g))
        self.assertTrue(o.isclose(o * (f + g)))

    def test_init_str(self):
        fermion_op = FermionOperator('0^ 5 12^', -1.)
        correct = ((0, 1), (5, 0), (12, 1))
        self.assertIn(correct, fermion_op.terms)
        self.assertEqual(fermion_op.terms[correct], -1.0)

    def test_merges_multiple_whitespace(self):
        fermion_op = FermionOperator('        \n ')
        self.assertEqual(fermion_op.terms, {(): 1})

    def test_init_str_identity(self):
        fermion_op = FermionOperator('')
        self.assertIn((), fermion_op.terms)

    def test_init_bad_term(self):
        with self.assertRaises(ValueError):
            _ = FermionOperator(list())

    def test_init_bad_coefficient(self):
        with self.assertRaises(ValueError):
            _ = FermionOperator('0^', "0.5")

    def test_init_bad_action_str(self):
        with self.assertRaises(FermionOperatorError):
            _ = FermionOperator('0-')

    def test_init_bad_action_tuple(self):
        with self.assertRaises(ValueError):
            FermionOperator(((0, 2),))

    def test_init_bad_tuple(self):
        with self.assertRaises(ValueError):
            _ = FermionOperator(((0, 1, 1),))

    def test_init_bad_str(self):
        with self.assertRaises(FermionOperatorError):
            _ = FermionOperator('^')

    def test_init_bad_mode_num(self):
        with self.assertRaises(FermionOperatorError):
            _ = FermionOperator('-1^')

    def test_init_invalid_tensor_factor(self):
        with self.assertRaises(FermionOperatorError):
            _ = FermionOperator(((-2, 1), (1, 0)))

    def test_FermionOperator(self):
        op = FermionOperator((), 3.)
        self.assertTrue(op.isclose(FermionOperator(()) * 3.))

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

    def test_isclose_abs_tol(self):
        a = FermionOperator('0^', -1.)
        b = FermionOperator('0^', -1.05)
        c = FermionOperator('0^', -1.11)
        self.assertTrue(a.isclose(b, rel_tol=1e-14, abs_tol=0.1))
        self.assertFalse(a.isclose(c, rel_tol=1e-14, abs_tol=0.1))
        a = FermionOperator('0^', -1.0j)
        b = FermionOperator('0^', -1.05j)
        c = FermionOperator('0^', -1.11j)
        self.assertTrue(a.isclose(b, rel_tol=1e-14, abs_tol=0.1))
        self.assertFalse(a.isclose(c, rel_tol=1e-14, abs_tol=0.1))

    def test_isclose_rel_tol(self):
        a = FermionOperator('0', 1)
        b = FermionOperator('0', 2)
        self.assertTrue(a.isclose(b, rel_tol=2.5, abs_tol=0.1))
        # Test symmetry
        self.assertTrue(a.isclose(b, rel_tol=1, abs_tol=0.1))
        self.assertTrue(b.isclose(a, rel_tol=1, abs_tol=0.1))

    def test_isclose_zero_terms(self):
        op = FermionOperator('1^ 0', -1j) * 0
        self.assertTrue(op.isclose(FermionOperator((), 0.0),
                                   rel_tol=1e-12, abs_tol=1e-12))
        self.assertTrue(FermionOperator().isclose(
            op, rel_tol=1e-12, abs_tol=1e-12))

    def test_isclose_different_terms(self):
        a = FermionOperator(((1, 0),), -0.1j)
        b = FermionOperator(((1, 1),), -0.1j)
        self.assertTrue(a.isclose(b, rel_tol=1e-12, abs_tol=0.2))
        self.assertFalse(a.isclose(b, rel_tol=1e-12, abs_tol=0.05))
        self.assertTrue(b.isclose(a, rel_tol=1e-12, abs_tol=0.2))
        self.assertFalse(b.isclose(a, rel_tol=1e-12, abs_tol=0.05))

    def test_isclose_different_num_terms(self):
        a = FermionOperator(((1, 0),), -0.1j)
        a += FermionOperator(((1, 1),), -0.1j)
        b = FermionOperator(((1, 0),), -0.1j)
        self.assertFalse(b.isclose(a, rel_tol=1e-12, abs_tol=0.05))
        self.assertFalse(a.isclose(b, rel_tol=1e-12, abs_tol=0.05))

    def test_imul_inplace(self):
        fermion_op = FermionOperator("1^")
        prev_id = id(fermion_op)
        fermion_op *= 3.
        self.assertEqual(id(fermion_op), prev_id)
        self.assertEqual(fermion_op.terms[((1, 1),)], 3.)

    def test_imul_scalar_real(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = 0.5
        fermion_op = FermionOperator(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_scalar_complex(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = 0.6j
        fermion_op = FermionOperator(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_scalar_npfloat64(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = numpy.float64(2.303)
        fermion_op = FermionOperator(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_scalar_npcomplex128(self):
        loc_op = ((1, 0), (2, 1))
        multiplier = numpy.complex128(-1.123j + 1.7911)
        fermion_op = FermionOperator(loc_op)
        fermion_op *= multiplier
        self.assertEqual(fermion_op.terms[loc_op], multiplier)

    def test_imul_fermion_op(self):
        op1 = FermionOperator(((0, 1), (3, 0), (8, 1), (8, 0), (11, 1)), 3.j)
        op2 = FermionOperator(((1, 1), (3, 1), (8, 0)), 0.5)
        op1 *= op2
        correct_coefficient = 1.j * 3.0j * 0.5
        correct_term = ((0, 1), (3, 0), (8, 1), (8, 0), (11, 1),
                        (1, 1), (3, 1), (8, 0))
        self.assertEqual(len(op1.terms), 1)
        self.assertIn(correct_term, op1.terms)

    def test_imul_fermion_op_2(self):
        op3 = FermionOperator(((1, 1), (0, 0)), -1j)
        op4 = FermionOperator(((1, 0), (0, 1), (2, 1)), -1.5)
        op3 *= op4
        op4 *= op3
        self.assertIn(((1, 1), (0, 0), (1, 0), (0, 1), (2, 1)), op3.terms)
        self.assertEqual(op3.terms[((1, 1), (0, 0), (1, 0), (0, 1), (2, 1))],
                         1.5j)

    def test_imul_bidir(self):
        op_a = FermionOperator(((1, 1), (0, 0)), -1j)
        op_b = FermionOperator(((1, 1), (0, 1), (2, 1)), -1.5)
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
        op = FermionOperator(((1, 1), (0, 1)), -1j)
        with self.assertRaises(TypeError):
            op *= "1"

    def test_mul_by_scalarzero(self):
        op = FermionOperator(((1, 1), (0, 1)), -1j) * 0
        self.assertNotIn(((0, 1), (1, 1)), op.terms)
        self.assertIn(((1, 1), (0, 1)), op.terms)
        self.assertEqual(op.terms[((1, 1), (0, 1))], 0.0)

    def test_mul_bad_multiplier(self):
        op = FermionOperator(((1, 1), (0, 1)), -1j)
        with self.assertRaises(TypeError):
            op = op * "0.5"

    def test_mul_out_of_place(self):
        op1 = FermionOperator(((0, 1), (3, 1), (3, 0), (11, 1)), 3.j)
        op2 = FermionOperator(((1, 1), (3, 1), (8, 0)), 0.5)
        op3 = op1 * op2
        correct_coefficient = 3.0j * 0.5
        correct_term = ((0, 1), (3, 1), (3, 0), (11, 1),
                        (1, 1), (3, 1), (8, 0))
        self.assertTrue(op1.isclose(FermionOperator(
            ((0, 1), (3, 1), (3, 0), (11, 1)), 3.j)))
        self.assertTrue(op2.isclose(FermionOperator(((1, 1), (3, 1), (8, 0)),
                                                    0.5)))
        self.assertTrue(op3.isclose(FermionOperator(correct_term,
                                                    correct_coefficient)))

    def test_mul_npfloat64(self):
        op = FermionOperator(((1, 0), (3, 1)), 0.5)
        res = op * numpy.float64(0.5)
        self.assertTrue(res.isclose(FermionOperator(((1, 0), (3, 1)),
                                                    0.5 * 0.5)))

    def test_mul_multiple_terms(self):
        op = FermionOperator(((1, 0), (8, 1)), 0.5)
        op += FermionOperator(((1, 1), (9, 1)), 1.4j)
        res = op * op
        correct = FermionOperator(((1, 0), (8, 1), (1, 0), (8, 1)), 0.5 ** 2)
        correct += (FermionOperator(((1, 0), (8, 1), (1, 1), (9, 1)), 0.7j) +
                    FermionOperator(((1, 1), (9, 1), (1, 0), (8, 1)), 0.7j))
        correct += FermionOperator(((1, 1), (9, 1), (1, 1), (9, 1)), 1.4j ** 2)
        self.assertTrue(res.isclose(correct))

    def test_rmul_scalar_real(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = 0.5
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1.isclose(res2))

    def test_rmul_scalar_complex(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = 0.6j
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1.isclose(res2))

    def test_rmul_scalar_npfloat64(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = numpy.float64(2.303)
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1.isclose(res2))

    def test_rmul_scalar_npcomplex128(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        multiplier = numpy.complex128(-1.5j + 7.7)
        res1 = op * multiplier
        res2 = multiplier * op
        self.assertTrue(res1.isclose(res2))

    def test_rmul_bad_multiplier(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        with self.assertRaises(TypeError):
            op = "0.5" * op

    def test_truediv_and_div_real(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.5
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res.isclose(correct))
        # Test if done out of place
        self.assertTrue(op.isclose(original))

    def test_truediv_and_div_complex(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.6j
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res.isclose(correct))
        # Test if done out of place
        self.assertTrue(op.isclose(original))

    def test_truediv_and_div_npfloat64(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.float64(2.303)
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res.isclose(correct))
        # Test if done out of place
        self.assertTrue(op.isclose(original))

    def test_truediv_and_div_npcomplex128(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.complex128(566.4j + 0.3)
        original = copy.deepcopy(op)
        res = op / divisor
        correct = op * (1. / divisor)
        self.assertTrue(res.isclose(correct))
        # Test if done out of place
        self.assertTrue(op.isclose(original))

    def test_truediv_bad_divisor(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        with self.assertRaises(TypeError):
            op = op / "0.5"

    def test_itruediv_and_idiv_real(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.5
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op.isclose(correct))
        # Test if done in-place
        self.assertFalse(op.isclose(original))

    def test_itruediv_and_idiv_complex(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = 0.6j
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op.isclose(correct))
        # Test if done in-place
        self.assertFalse(op.isclose(original))

    def test_itruediv_and_idiv_npfloat64(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.float64(2.3030)
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op.isclose(correct))
        # Test if done in-place
        self.assertFalse(op.isclose(original))

    def test_itruediv_and_idiv_npcomplex128(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        divisor = numpy.complex128(12.3 + 7.4j)
        original = copy.deepcopy(op)
        correct = op * (1. / divisor)
        op /= divisor
        self.assertTrue(op.isclose(correct))
        # Test if done in-place
        self.assertFalse(op.isclose(original))

    def test_itruediv_bad_divisor(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        with self.assertRaises(TypeError):
            op /= "0.5"

    def test_iadd_different_term(self):
        term_a = ((1, 1), (3, 0), (8, 1))
        term_b = ((1, 1), (3, 1), (8, 0))
        a = FermionOperator(term_a, 1.0)
        a += FermionOperator(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], 0.5)
        a += FermionOperator(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], 1.0)

    def test_iadd_bad_addend(self):
        op = FermionOperator((), 1.0)
        with self.assertRaises(TypeError):
            op += "0.5"

    def test_add(self):
        term_a = ((1, 1), (3, 0), (8, 1))
        term_b = ((1, 0), (3, 0), (8, 1))
        a = FermionOperator(term_a, 1.0)
        b = FermionOperator(term_b, 0.5)
        res = a + b + b
        self.assertEqual(len(res.terms), 2)
        self.assertEqual(res.terms[term_a], 1.0)
        self.assertEqual(res.terms[term_b], 1.0)
        # Test out of place
        self.assertTrue(a.isclose(FermionOperator(term_a, 1.0)))
        self.assertTrue(b.isclose(FermionOperator(term_b, 0.5)))

    def test_add_bad_addend(self):
        op = FermionOperator((), 1.0)
        with self.assertRaises(TypeError):
            _ = op + "0.5"

    def test_sub(self):
        term_a = ((1, 1), (3, 1), (8, 1))
        term_b = ((1, 0), (3, 1), (8, 1))
        a = FermionOperator(term_a, 1.0)
        b = FermionOperator(term_b, 0.5)
        res = a - b
        self.assertEqual(len(res.terms), 2)
        self.assertEqual(res.terms[term_a], 1.0)
        self.assertEqual(res.terms[term_b], -0.5)
        res2 = b - a
        self.assertEqual(len(res2.terms), 2)
        self.assertEqual(res2.terms[term_a], -1.0)
        self.assertEqual(res2.terms[term_b], 0.5)

    def test_sub_bad_subtrahend(self):
        op = FermionOperator((), 1.0)
        with self.assertRaises(TypeError):
            _ = op - "0.5"

    def test_isub_different_term(self):
        term_a = ((1, 1), (3, 1), (8, 0))
        term_b = ((1, 0), (3, 1), (8, 1))
        a = FermionOperator(term_a, 1.0)
        a -= FermionOperator(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], -0.5)
        a -= FermionOperator(term_b, 0.5)
        self.assertEqual(len(a.terms), 2)
        self.assertEqual(a.terms[term_a], 1.0)
        self.assertEqual(a.terms[term_b], -1.0)

    def test_isub_bad_addend(self):
        op = FermionOperator((), 1.0)
        with self.assertRaises(TypeError):
            op -= "0.5"

    def test_neg(self):
        op = FermionOperator(((1, 1), (3, 1), (8, 1)), 0.5)
        _ = -op
        # out of place
        self.assertTrue(op.isclose(FermionOperator(((1, 1), (3, 1), (8, 1)),
                                                   0.5)))
        correct = -1.0 * op
        self.assertTrue(correct.isclose(-op))

    def test_pow_square_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = FermionOperator(ops, coeff)
        squared = term ** 2
        expected = FermionOperator(ops + ops, coeff ** 2)
        self.assertTrue(squared.isclose(term * term))
        self.assertTrue(squared.isclose(expected))

    def test_pow_zero_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = FermionOperator(ops, coeff)
        zerod = term ** 0
        expected = FermionOperator(())
        self.assertTrue(expected.isclose(zerod))

    def test_pow_one_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = FermionOperator(ops, coeff)
        self.assertTrue(term.isclose(term ** 1))

    def test_pow_high_term(self):
        coeff = 6.7j
        ops = ((3, 1), (1, 0), (4, 1))
        term = FermionOperator(ops, coeff)
        high = term ** 10
        expected = FermionOperator(ops * 10, coeff ** 10)
        self.assertTrue(expected.isclose(high))

    def test_pow_neg_error(self):
        with self.assertRaises(ValueError):
            FermionOperator() ** -1

    def test_pow_nonint_error(self):
        with self.assertRaises(ValueError):
            FermionOperator('3 2^') ** 0.5

    def test_hermitian_conjugate_empty(self):
        op = FermionOperator()
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(FermionOperator()))

    def test_hermitian_conjugate_simple(self):
        op = FermionOperator('1^')
        op_hc = FermionOperator('1')
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugate_complex_const(self):
        op = FermionOperator('1^ 3', 3j)
        op_hc = -3j * FermionOperator('3^ 1')
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugate_notordered(self):
        op = FermionOperator('1 3^ 3 3^', 3j)
        op_hc = -3j * FermionOperator('3 3^ 3 1^')
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugate_semihermitian(self):
        op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
              FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
        op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
                 FermionOperator('3^ 1', -2j) +
                 FermionOperator('2^ 2', -0.1j))
        op = hermitian_conjugated(op)
        self.assertTrue(op.isclose(op_hc))

    def test_hermitian_conjugated_empty(self):
        op = FermionOperator()
        self.assertTrue(op.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_simple(self):
        op = FermionOperator('0')
        op_hc = FermionOperator('0^')
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_complex_const(self):
        op = FermionOperator('2^ 2', 3j)
        op_hc = FermionOperator('2^ 2', -3j)
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_multiterm(self):
        op = FermionOperator('1^ 2') + FermionOperator('2 3 4')
        op_hc = FermionOperator('2^ 1') + FermionOperator('4^ 3^ 2^')
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_hermitian_conjugated_semihermitian(self):
        op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
              FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
        op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
                 FermionOperator('3^ 1', -2j) +
                 FermionOperator('2^ 2', -0.1j))
        self.assertTrue(op_hc.isclose(hermitian_conjugated(op)))

    def test_compress_terms(self):
        op = (FermionOperator('3^ 1', 0.3 + 3e-11j) +
              FermionOperator('2^ 3', 5e-10) +
              FermionOperator('1^ 3', 1e-3))
        op_compressed = (FermionOperator('3^ 1', 0.3) +
                         FermionOperator('1^ 3', 1e-3))
        op.compress(1e-7)
        self.assertTrue(op_compressed.isclose(op))

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

    def test_str(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        self.assertEqual(str(op), "0.5 [1^ 3 8^]")
        op2 = FermionOperator((), 2)
        self.assertEqual(str(op2), "2 []")
        op3 = FermionOperator()
        self.assertEqual(str(op3), "0")

    def test_rep(self):
        op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
        # Not necessary, repr could do something in addition
        self.assertEqual(repr(op), str(op))
