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

"""Tests  _reverse_jordan_wigner.py."""

import unittest

from openfermion.ops import (FermionOperator, QubitOperator)
from openfermion.transforms import jordan_wigner, reverse_jordan_wigner
from openfermion.utils import normal_ordered


class ReverseJWTest(unittest.TestCase):

    def setUp(self):
        self.coefficient = 0.5
        self.operators = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        self.term = QubitOperator(self.operators, self.coefficient)
        self.identity = QubitOperator(())
        self.coefficient_a = 6.7j
        self.coefficient_b = -88.
        self.operators_a = ((3, 'Z'), (1, 'Y'), (4, 'Y'))
        self.operators_b = ((2, 'X'), (3, 'Y'))
        self.operator_a = QubitOperator(self.operators_a, self.coefficient_a)
        self.operator_b = QubitOperator(self.operators_b, self.coefficient_b)
        self.operator_ab = self.operator_a + self.operator_b
        self.qubit_operator = QubitOperator(
            ((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        self.qubit_operator += QubitOperator(
            ((1, 'Z'), (3, 'X'), (8, 'Z')), 1.2)

    def test_identity_jwterm(self):
        self.assertTrue(FermionOperator(()) ==
            reverse_jordan_wigner(QubitOperator(())))

    def test_x(self):
        pauli_x = QubitOperator(((2, 'X'),))
        transmed_x = reverse_jordan_wigner(pauli_x)
        retransmed_x = jordan_wigner(transmed_x)
        self.assertTrue(pauli_x == retransmed_x)

    def test_y(self):
        pauli_y = QubitOperator(((2, 'Y'),))
        transmed_y = reverse_jordan_wigner(pauli_y)
        retransmed_y = jordan_wigner(transmed_y)
        self.assertTrue(pauli_y == retransmed_y)

    def test_z(self):
        pauli_z = QubitOperator(((2, 'Z'),))
        transmed_z = reverse_jordan_wigner(pauli_z)

        expected = (FermionOperator(()) +
                    FermionOperator(((2, 1), (2, 0)), -2.))
        self.assertTrue(transmed_z == expected)

        retransmed_z = jordan_wigner(transmed_z)
        self.assertTrue(pauli_z == retransmed_z)

    def test_reverse_jw_too_few_n_qubits(self):
        with self.assertRaises(ValueError):
            reverse_jordan_wigner(self.operator_a, 0)

    def test_identity(self):
        n_qubits = 5
        transmed_i = reverse_jordan_wigner(self.identity, n_qubits)
        expected_i = FermionOperator(())
        self.assertTrue(transmed_i == expected_i)
        retransmed_i = jordan_wigner(transmed_i)
        self.assertTrue(self.identity == retransmed_i)

    def test_zero(self):
        n_qubits = 5
        transmed_i = reverse_jordan_wigner(QubitOperator(), n_qubits)
        expected_i = FermionOperator()
        self.assertTrue(transmed_i == expected_i)

        retransmed_i = jordan_wigner(transmed_i)
        expected_i = QubitOperator()
        self.assertTrue(expected_i == retransmed_i)

    def test_yzxz(self):
        yzxz = QubitOperator(((0, 'Y'), (1, 'Z'), (2, 'X'), (3, 'Z')))
        transmed_yzxz = reverse_jordan_wigner(yzxz)
        retransmed_yzxz = jordan_wigner(transmed_yzxz)
        self.assertTrue(yzxz == retransmed_yzxz)

    def test_term(self):
        transmed_term = reverse_jordan_wigner(self.term)
        retransmed_term = jordan_wigner(transmed_term)
        self.assertTrue(self.term == retransmed_term)

    def test_xx(self):
        xx = QubitOperator(((3, 'X'), (4, 'X')), 2.)
        transmed_xx = reverse_jordan_wigner(xx)
        retransmed_xx = jordan_wigner(transmed_xx)

        expected1 = (FermionOperator(((3, 1),), 2.) -
                     FermionOperator(((3, 0),), 2.))
        expected2 = (FermionOperator(((4, 1),), 1.) +
                     FermionOperator(((4, 0),), 1.))
        expected = expected1 * expected2

        self.assertTrue(xx == retransmed_xx)
        self.assertTrue(normal_ordered(transmed_xx) ==
            normal_ordered(expected))

    def test_yy(self):
        yy = QubitOperator(((2, 'Y'), (3, 'Y')), 2.)
        transmed_yy = reverse_jordan_wigner(yy)
        retransmed_yy = jordan_wigner(transmed_yy)

        expected1 = -(FermionOperator(((2, 1),), 2.) +
                      FermionOperator(((2, 0),), 2.))
        expected2 = (FermionOperator(((3, 1),)) -
                     FermionOperator(((3, 0),)))
        expected = expected1 * expected2

        self.assertTrue(yy == retransmed_yy)
        self.assertTrue(normal_ordered(transmed_yy) ==
            normal_ordered(expected))

    def test_xy(self):
        xy = QubitOperator(((4, 'X'), (5, 'Y')), -2.j)
        transmed_xy = reverse_jordan_wigner(xy)
        retransmed_xy = jordan_wigner(transmed_xy)

        expected1 = -2j * (FermionOperator(((4, 1),), 1j) -
                           FermionOperator(((4, 0),), 1j))
        expected2 = (FermionOperator(((5, 1),)) -
                     FermionOperator(((5, 0),)))
        expected = expected1 * expected2

        self.assertTrue(xy == retransmed_xy)
        self.assertTrue(normal_ordered(transmed_xy) ==
            normal_ordered(expected))

    def test_yx(self):
        yx = QubitOperator(((0, 'Y'), (1, 'X')), -0.5)
        transmed_yx = reverse_jordan_wigner(yx)
        retransmed_yx = jordan_wigner(transmed_yx)

        expected1 = 1j * (FermionOperator(((0, 1),)) +
                          FermionOperator(((0, 0),)))
        expected2 = -0.5 * (FermionOperator(((1, 1),)) +
                            FermionOperator(((1, 0),)))
        expected = expected1 * expected2

        self.assertTrue(yx == retransmed_yx)
        self.assertTrue(normal_ordered(transmed_yx) ==
            normal_ordered(expected))

    def test_jw_term_bad_type(self):
        with self.assertRaises(TypeError):
            reverse_jordan_wigner(3)

    def test_reverse_jordan_wigner(self):
        transmed_operator = reverse_jordan_wigner(self.qubit_operator)
        retransmed_operator = jordan_wigner(transmed_operator)
        self.assertTrue(self.qubit_operator == retransmed_operator)

    def test_reverse_jw_linearity(self):
        term1 = QubitOperator(((0, 'X'), (1, 'Y')), -0.5)
        term2 = QubitOperator(((0, 'Y'), (1, 'X'), (2, 'Y'), (3, 'Y')), -1j)

        op12 = reverse_jordan_wigner(term1) - reverse_jordan_wigner(term2)
        self.assertTrue(op12 == reverse_jordan_wigner(term1 - term2))

    def test_bad_type(self):
        with self.assertRaises(TypeError):
            reverse_jordan_wigner(3)

    def test_jw_convention(self):
        """Test that the Jordan-Wigner convention places the Z-string on
        lower indices."""
        qubit_op = QubitOperator('Z0 X1')
        transformed_op = reverse_jordan_wigner(qubit_op)
        expected_op = FermionOperator('1^')
        expected_op += FermionOperator('1')
        self.assertTrue(transformed_op == expected_op)
