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

"""Tests for interaction_tensors.py."""
from __future__ import absolute_import

import unittest

import copy
import numpy

from openfermion.ops import InteractionTensor


class InteractionTensorTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 2
        self.constant = 23.0

        one_body_a = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body_a = numpy.zeros((self.n_qubits, self.n_qubits,
                                  self.n_qubits, self.n_qubits))
        one_body_a[0, 1] = 2
        one_body_a[1, 0] = 3
        two_body_a[0, 1, 0, 1] = 4
        two_body_a[1, 1, 0, 0] = 5
        self.interaction_tensor_a = InteractionTensor(self.constant,
                                                      one_body_a, two_body_a)

        one_body_na = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body_na = numpy.zeros((self.n_qubits, self.n_qubits,
                                   self.n_qubits, self.n_qubits))
        one_body_na[0, 1] = -2
        one_body_na[1, 0] = -3
        two_body_na[0, 1, 0, 1] = -4
        two_body_na[1, 1, 0, 0] = -5
        self.interaction_tensor_na = InteractionTensor(-self.constant,
                                                       one_body_na,
                                                       two_body_na)

        one_body_b = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body_b = numpy.zeros((self.n_qubits, self.n_qubits,
                                  self.n_qubits, self.n_qubits))
        one_body_b[0, 1] = 1
        one_body_b[1, 0] = 2
        two_body_b[0, 1, 0, 1] = 3
        two_body_b[1, 0, 0, 1] = 4
        self.interaction_tensor_b = InteractionTensor(self.constant,
                                                      one_body_b,
                                                      two_body_b)

        one_body_ab = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body_ab = numpy.zeros((self.n_qubits, self.n_qubits,
                                   self.n_qubits, self.n_qubits))
        one_body_ab[0, 1] = 3
        one_body_ab[1, 0] = 5
        two_body_ab[0, 1, 0, 1] = 7
        two_body_ab[1, 0, 0, 1] = 4
        two_body_ab[1, 1, 0, 0] = 5
        self.interaction_tensor_ab = InteractionTensor(2.0 * self.constant,
                                                       one_body_ab,
                                                       two_body_ab)

        constant_axb = self.constant * self.constant
        one_body_axb = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body_axb = numpy.zeros((self.n_qubits, self.n_qubits,
                                    self.n_qubits, self.n_qubits))
        one_body_axb[0, 1] = 2
        one_body_axb[1, 0] = 6
        two_body_axb[0, 1, 0, 1] = 12
        self.interaction_tensor_axb = InteractionTensor(constant_axb,
                                                        one_body_axb,
                                                        two_body_axb)

        self.n_qubits_plus_one = self.n_qubits + 1
        one_body_c = numpy.zeros((self.n_qubits_plus_one,
                                  self.n_qubits_plus_one))
        two_body_c = numpy.zeros((self.n_qubits_plus_one,
                                  self.n_qubits_plus_one,
                                  self.n_qubits_plus_one,
                                  self.n_qubits_plus_one))
        one_body_c[0, 1] = 1
        one_body_c[1, 0] = 2
        two_body_c[0, 1, 0, 1] = 3
        two_body_c[1, 0, 0, 1] = 4
        self.interaction_tensor_c = InteractionTensor(self.constant,
                                                      one_body_c,
                                                      two_body_c)

    def test_setitem_1body(self):
        expected_one_body_tensor = numpy.array([[0, 3], [2, 0]])
        self.interaction_tensor_a[0, 1] = 3
        self.interaction_tensor_a[1, 0] = 2
        self.assertTrue(numpy.allclose(
            self.interaction_tensor_a.one_body_tensor,
            expected_one_body_tensor))

    def test_getitem_1body(self):
        self.assertEqual(self.interaction_tensor_c[0, 1], 1)
        self.assertEqual(self.interaction_tensor_c[1, 0], 2)

    def test_setitem_2body(self):
        self.interaction_tensor_a[0, 1, 1, 0] = 3
        self.interaction_tensor_a[1, 0, 0, 1] = 2
        self.assertEqual(self.interaction_tensor_a.two_body_tensor[0, 1, 1, 0],
                         3)
        self.assertEqual(self.interaction_tensor_a.two_body_tensor[1, 0, 0, 1],
                         2)

    def test_getitem_2body(self):
        self.assertEqual(self.interaction_tensor_c[0, 1, 0, 1], 3)
        self.assertEqual(self.interaction_tensor_c[1, 0, 0, 1], 4)

    def test_invalid_getitem_indexing(self):
        with self.assertRaises(ValueError):
            self.interaction_tensor_a[0, 0, 0]

    def test_invalid_setitem_indexing(self):
        test_tensor = copy.deepcopy(self.interaction_tensor_a)
        with self.assertRaises(ValueError):
            test_tensor[0, 0, 0] = 5

    def test_neq(self):
        self.assertNotEqual(self.interaction_tensor_a,
                            self.interaction_tensor_b)
        self.assertTrue(self.interaction_tensor_a != self.interaction_tensor_b)

    def test_add(self):
        new_tensor = self.interaction_tensor_a + self.interaction_tensor_b
        self.assertEqual(new_tensor, self.interaction_tensor_ab)

    def test_iadd(self):
        new_tensor = copy.deepcopy(self.interaction_tensor_a)
        new_tensor += self.interaction_tensor_b
        self.assertEqual(new_tensor, self.interaction_tensor_ab)

    def test_invalid_addend(self):
        with self.assertRaises(TypeError):
            self.interaction_tensor_a + 2

    def test_invalid_tensor_shape_add(self):
        with self.assertRaises(TypeError):
            self.interaction_tensor_a + self.interaction_tensor_c

    def test_neg(self):
        self.assertEqual(-self.interaction_tensor_a,
                         self.interaction_tensor_na)

    def test_sub(self):
        new_tensor = self.interaction_tensor_ab - self.interaction_tensor_b
        self.assertEqual(new_tensor, self.interaction_tensor_a)

    def test_isub(self):
        new_tensor = copy.deepcopy(self.interaction_tensor_ab)
        new_tensor -= self.interaction_tensor_b
        self.assertEqual(new_tensor, self.interaction_tensor_a)

    def test_invalid_subtrahend(self):
        with self.assertRaises(TypeError):
            self.interaction_tensor_a - 2

    def test_invalid_tensor_shape_sub(self):
        with self.assertRaises(TypeError):
            self.interaction_tensor_a - self.interaction_tensor_c

    def test_mul(self):
        new_tensor = self.interaction_tensor_a * self.interaction_tensor_b
        self.assertEqual(new_tensor, self.interaction_tensor_axb)

    def test_imul(self):
        new_tensor = copy.deepcopy(self.interaction_tensor_a)
        new_tensor *= self.interaction_tensor_b
        self.assertEqual(new_tensor, self.interaction_tensor_axb)

    def test_invalid_multiplier(self):
        with self.assertRaises(TypeError):
            self.interaction_tensor_a * 2

    def test_invalid_tensor_shape_mult(self):
        with self.assertRaises(TypeError):
            self.interaction_tensor_a * self.interaction_tensor_c

    def test_iter_and_str(self):
        one_body = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits))
        one_body[0, 1] = 11.0
        two_body[0, 1, 1, 0] = 22.0
        interaction_tensor = InteractionTensor(self.constant,
                                               one_body, two_body)
        want_str = '[] 23.0\n[0 1] 11.0\n[0 1 1 0] 22.0\n'
        self.assertEqual(interaction_tensor.__str__(), want_str)
        self.assertEqual(interaction_tensor.__repr__(), want_str)

    def test_init_none(self):
        n_qubits = 2
        constant = 23.0

        one_body = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits))
        one_body[0, 1] = 2
        one_body[1, 0] = 3
        two_body[0, 1, 0, 1] = 4
        two_body[1, 1, 0, 0] = 5
        interaction_tensor = InteractionTensor(None,
                                               one_body, two_body)
        self.assertAlmostEqual(interaction_tensor.constant, 0.)

        self.interaction_tensor_a[()] = 0.
        self.assertAlmostEqual(self.interaction_tensor_a, interaction_tensor)
        self.interaction_tensor_a[()] = self.constant


    def test_rotate_basis_identical(self):
        rotation_matrix_identical = numpy.zeros((self.n_qubits, self.n_qubits))
        rotation_matrix_identical[0, 0] = 1
        rotation_matrix_identical[1, 1] = 1

        one_body = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits))
        i = 0
        j = 0
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                one_body[p, q] = i
                i = i + 1
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        two_body[p, q, r, s] = j
                        j = j + 1
        interaction_tensor = InteractionTensor(self.constant,
                                               one_body, two_body)
        want_interaction_tensor = InteractionTensor(self.constant,
                                                    one_body, two_body)

        interaction_tensor.rotate_basis(rotation_matrix_identical)
        self.assertEqual(interaction_tensor, want_interaction_tensor)

    def test_rotate_basis_reverse(self):
        rotation_matrix_reverse = numpy.zeros((self.n_qubits, self.n_qubits))
        rotation_matrix_reverse[0, 1] = 1
        rotation_matrix_reverse[1, 0] = 1

        one_body = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits))
        one_body_reverse = numpy.zeros((self.n_qubits, self.n_qubits))
        two_body_reverse = numpy.zeros((self.n_qubits, self.n_qubits,
                                        self.n_qubits, self.n_qubits))
        i = 0
        j = 0
        i_reverse = pow(self.n_qubits, 2) - 1
        j_reverse = pow(self.n_qubits, 4) - 1
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                one_body[p, q] = i
                i = i + 1
                one_body_reverse[p, q] = i_reverse
                i_reverse = i_reverse - 1
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        two_body[p, q, r, s] = j
                        j = j + 1
                        two_body_reverse[p, q, r, s] = j_reverse
                        j_reverse = j_reverse - 1
        interaction_tensor = InteractionTensor(self.constant,
                                               one_body, two_body)
        want_interaction_tensor = InteractionTensor(self.constant,
                                                    one_body_reverse,
                                                    two_body_reverse)
        interaction_tensor.rotate_basis(rotation_matrix_reverse)
        self.assertEqual(interaction_tensor, want_interaction_tensor)
