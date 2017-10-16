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

"""Tests for _trotter_error.py."""
from future.utils import iteritems

from math import sqrt
import numpy
from scipy.linalg import expm
import unittest

from openfermion.config import *
from openfermion.hamiltonians import MolecularData
from openfermion.ops import normal_ordered, QubitOperator
from openfermion.transforms import get_sparse_operator
from openfermion.utils._trotter_error import *


class CommutatorTest(unittest.TestCase):

    def test_commutator_commutes(self):
        zero = QubitOperator()
        self.assertTrue(commutator(QubitOperator(()),
                        QubitOperator('X3')).isclose(zero))

    def test_commutator_single_pauli(self):
        com = commutator(QubitOperator('X3'),
                         QubitOperator('Y3'))
        expected = 2j * QubitOperator('Z3')
        self.assertTrue(expected.isclose(com))

    def test_commutator_multi_pauli(self):
        com = commutator(QubitOperator('Z1 X2 Y4'),
                         QubitOperator('X1 Z2 X4'))
        expected = -2j * QubitOperator('Y1 Y2 Z4')
        self.assertTrue(expected.isclose(com))


class TriviallyCommutesTest(unittest.TestCase):

    def test_trivially_commutes_id_id(self):
        self.assertTrue(trivially_commutes(
            QubitOperator(()), 3 * QubitOperator(())))

    def test_trivially_commutes_id_x(self):
        self.assertTrue(trivially_commutes(
            QubitOperator(()), QubitOperator('X1')))

    def test_trivially_commutes_id_xx(self):
        self.assertTrue(trivially_commutes(
            QubitOperator(()), QubitOperator('X1 X3')))

    def test_trivially_commutes_nonid_with_id(self):
        self.assertTrue(trivially_commutes(
            QubitOperator('X1 Z5 Y9 Z11'), QubitOperator(())))

    def test_trivially_commutes_no_intersect(self):
        self.assertTrue(trivially_commutes(
            QubitOperator('X1 Y3 Z6'), QubitOperator('Z0 Z2 X4 Y5')))

    def test_trivially_commutes_allsame_oddintersect(self):
        self.assertTrue(trivially_commutes(
            QubitOperator('X1 X3 X4 Z6 X8'), QubitOperator('X1 X3 X4 Z7 Y9')))

    def test_trivially_commutes_even_anti(self):
        self.assertTrue(trivially_commutes(
            QubitOperator('X1 Z2 Z3 X10'), QubitOperator('Y1 X2 Z3 Y9')))

    def test_no_trivial_commute_odd_anti(self):
        self.assertFalse(trivially_commutes(
            QubitOperator('X1'), QubitOperator('Y1')))

    def test_no_trivial_commute_triple_anti_intersect(self):
        self.assertFalse(trivially_commutes(
            QubitOperator('X0 Z2 Z4 Z9 Y17'),
            QubitOperator('Y0 X2 Y4 Z9 Z16')))

    def test_no_trivial_commute_mostly_commuting(self):
        self.assertFalse(trivially_commutes(
            QubitOperator('X0 Y1 Z2 X4 Y5 Y6'),
            QubitOperator('X0 Y1 Z2 X4 Z5 Y6')))


class TriviallyDoubleCommutesTest(unittest.TestCase):

    def test_trivial_double_commute_no_intersect(self):
        self.assertTrue(trivially_double_commutes(
            QubitOperator('X1 Z2 Y4'), QubitOperator('Y0 X3 Z6'),
            QubitOperator('Y5')))

    def test_trivial_double_commute_no_intersect_a_bc(self):
        self.assertTrue(trivially_double_commutes(
            QubitOperator('X1 Z2 Y4'), QubitOperator('Y0 X3 Z6'),
            QubitOperator('Z3 Y5')))

    def test_trivial_double_commute_bc_intersect_commute(self):
        self.assertTrue(trivially_double_commutes(
            QubitOperator('X1 Z2 Y4'), QubitOperator('X0 Z3'),
            QubitOperator('Y0 X3')))


class ErrorOperatorTest(unittest.TestCase):
    def test_error_operator_bad_order(self):
        with self.assertRaises(NotImplementedError):
            error_operator([QubitOperator], 1)

    def test_error_operator_all_diagonal(self):
        terms = [QubitOperator(()), QubitOperator('Z0 Z1 Z2'),
                 QubitOperator('Z0 Z3'), QubitOperator('Z0 Z1 Z2 Z3')]
        zero = QubitOperator()
        self.assertTrue(zero.isclose(error_operator(terms)))


class ErrorBoundTest(unittest.TestCase):
    def test_error_bound_xyz_tight(self):
        terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
        expected = sqrt(7. / 12)  # 2-norm of [[-2/3, 1/3+i/6], [1/3-i/6, 2/3]]
        self.assertLess(expected, error_bound(terms, tight=True))

    def test_error_bound_xyz_loose(self):
        terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
        self.assertTrue(numpy.isclose(
            error_bound(terms, tight=False), 4. * (2 ** 2 + 1 ** 2)))

    def test_error_operator_xyz(self):
        terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
        expected = numpy.array([[-2./3, 1./3 + 1.j/6, 0., 0.],
                                [1./3 - 1.j/6, 2./3, 0., 0.],
                                [0., 0., -2./3, 1./3 + 1.j/6],
                                [0., 0., 1./3 - 1.j/6, 2./3]])
        sparse_op = get_sparse_operator(error_operator(terms))
        matrix = sparse_op.todense()
        self.assertTrue(numpy.allclose(matrix, expected),
                        ("Got " + str(matrix)))

    def test_error_bound_qubit_tight_less_than_loose_integration(self):
        terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
        self.assertLess(error_bound(terms, tight=True),
                        error_bound(terms, tight=False))


class TrotterStepsRequiredTest(unittest.TestCase):
    def test_trotter_steps_required(self):
        self.assertEqual(trotter_steps_required(
            trotter_error_bound=0.3, time=2.5, energy_precision=0.04), 7)

    def test_trotter_steps_required_negative_time(self):
        self.assertEqual(trotter_steps_required(
            trotter_error_bound=0.1, time=3.3, energy_precision=0.11), 4)

    def test_return_type(self):
        self.assertIsInstance(trotter_steps_required(0.1, 0.1, 0.1), int)
