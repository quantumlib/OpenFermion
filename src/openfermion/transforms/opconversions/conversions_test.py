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
'''Tests for conversions.py'''
import unittest
import pytest
import numpy
import sympy

from openfermion.ops.operators import (
    QuadOperator,
    BosonOperator,
    FermionOperator,
    MajoranaOperator,
    QubitOperator,
)
from openfermion.transforms.repconversions.conversions import get_diagonal_coulomb_hamiltonian
from openfermion.transforms.opconversions.term_reordering import normal_ordered

from openfermion.transforms.opconversions.conversions import (
    get_quad_operator,
    get_boson_operator,
    get_majorana_operator,
    get_fermion_operator,
    check_no_sympy,
    _fermion_operator_to_majorana_operator,
    _fermion_term_to_majorana_operator,
)


class GetQuadOperatorTest(unittest.TestCase):
    def setUp(self):
        self.hbar = 0.5

    def test_invalid_op(self):
        op = QuadOperator()
        with self.assertRaises(TypeError):
            _ = get_quad_operator(op)

    def test_zero(self):
        b = BosonOperator()
        q = get_quad_operator(b)
        self.assertTrue(q == QuadOperator.zero())

    def test_identity(self):
        b = BosonOperator('')
        q = get_quad_operator(b)
        self.assertTrue(q == QuadOperator.identity())

    def test_creation(self):
        b = BosonOperator('0^')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') - 1j * QuadOperator('p0')
        expected /= numpy.sqrt(2 * self.hbar)
        self.assertTrue(q == expected)

    def test_annihilation(self):
        b = BosonOperator('0')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') + 1j * QuadOperator('p0')
        expected /= numpy.sqrt(2 * self.hbar)
        self.assertTrue(q == expected)

    def test_two_mode(self):
        b = BosonOperator('0^ 2')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = QuadOperator('q0') - 1j * QuadOperator('p0')
        expected *= QuadOperator('q2') + 1j * QuadOperator('p2')
        expected /= 2 * self.hbar
        self.assertTrue(q == expected)

    def test_two_term(self):
        b = BosonOperator('0^ 0') + BosonOperator('0 0^')
        q = get_quad_operator(b, hbar=self.hbar)
        expected = (QuadOperator('q0') - 1j * QuadOperator('p0')) * (
            QuadOperator('q0') + 1j * QuadOperator('p0')
        ) + (QuadOperator('q0') + 1j * QuadOperator('p0')) * (
            QuadOperator('q0') - 1j * QuadOperator('p0')
        )
        expected /= 2 * self.hbar
        self.assertTrue(q == expected)

    def test_q_squared(self):
        b = (
            self.hbar
            * (
                BosonOperator('0^ 0^')
                + BosonOperator('0 0')
                + BosonOperator('')
                + 2 * BosonOperator('0^ 0')
            )
            / 2
        )
        q = normal_ordered(get_quad_operator(b, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q0 q0')
        self.assertTrue(q == expected)

    def test_p_squared(self):
        b = (
            self.hbar
            * (
                -BosonOperator('1^ 1^')
                - BosonOperator('1 1')
                + BosonOperator('')
                + 2 * BosonOperator('1^ 1')
            )
            / 2
        )
        q = normal_ordered(get_quad_operator(b, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('p1 p1')
        self.assertTrue(q == expected)


class GetBosonOperatorTest(unittest.TestCase):
    def setUp(self):
        self.hbar = 0.5

    def test_invalid_op(self):
        op = BosonOperator()
        with self.assertRaises(TypeError):
            _ = get_boson_operator(op)

    def test_zero(self):
        q = QuadOperator()
        b = get_boson_operator(q)
        self.assertTrue(b == BosonOperator.zero())

    def test_identity(self):
        q = QuadOperator('')
        b = get_boson_operator(q)
        self.assertTrue(b == BosonOperator.identity())

    def test_x(self):
        q = QuadOperator('q0')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = BosonOperator('0') + BosonOperator('0^')
        expected *= numpy.sqrt(self.hbar / 2)
        self.assertTrue(b == expected)

    def test_p(self):
        q = QuadOperator('p2')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = BosonOperator('2') - BosonOperator('2^')
        expected *= -1j * numpy.sqrt(self.hbar / 2)
        self.assertTrue(b == expected)

    def test_two_mode(self):
        q = QuadOperator('p2 q0')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = (
            -1j
            * self.hbar
            / 2
            * (BosonOperator('0') + BosonOperator('0^'))
            * (BosonOperator('2') - BosonOperator('2^'))
        )
        self.assertTrue(b == expected)

    def test_two_term(self):
        q = QuadOperator('p0 q0') + QuadOperator('q0 p0')
        b = get_boson_operator(q, hbar=self.hbar)
        expected = (
            -1j
            * self.hbar
            / 2
            * (
                (BosonOperator('0') + BosonOperator('0^'))
                * (BosonOperator('0') - BosonOperator('0^'))
                + (BosonOperator('0') - BosonOperator('0^'))
                * (BosonOperator('0') + BosonOperator('0^'))
            )
        )
        self.assertTrue(b == expected)


def test_get_fermion_operator_majorana_operator():
    a = MajoranaOperator((0, 3), 2.0) + MajoranaOperator((1, 2, 3))
    op = get_fermion_operator(a)
    expected_op = (
        -2j
        * (
            FermionOperator(((0, 0), (1, 0)))
            - FermionOperator(((0, 0), (1, 1)))
            + FermionOperator(((0, 1), (1, 0)))
            - FermionOperator(((0, 1), (1, 1)))
        )
        - 2 * FermionOperator(((0, 0), (1, 1), (1, 0)))
        + 2 * FermionOperator(((0, 1), (1, 1), (1, 0)))
        + FermionOperator((0, 0))
        - FermionOperator((0, 1))
    )
    assert normal_ordered(op) == normal_ordered(expected_op)


def test_get_fermion_operator_wrong_type():
    with pytest.raises(TypeError):
        _ = get_fermion_operator(QubitOperator())


class GetMajoranaOperatorTest(unittest.TestCase):
    """Test class get Majorana Operator."""

    def test_raises(self):
        """Test raises errors."""
        with self.assertRaises(TypeError):
            get_majorana_operator(1.0)
        with self.assertRaises(TypeError):
            _fermion_operator_to_majorana_operator([1.0])
        with self.assertRaises(TypeError):
            _fermion_term_to_majorana_operator(1.0)

    def test_get_majorana_operator_fermion_operator(self):
        """Test conversion FermionOperator to MajoranaOperator."""
        fermion_op = (
            -2j
            * (
                FermionOperator(((0, 0), (1, 0)))
                - FermionOperator(((0, 0), (1, 1)))
                + FermionOperator(((0, 1), (1, 0)))
                - FermionOperator(((0, 1), (1, 1)))
            )
            - 2 * FermionOperator(((0, 0), (1, 1), (1, 0)))
            + 2 * FermionOperator(((0, 1), (1, 1), (1, 0)))
            + FermionOperator((0, 0))
            - FermionOperator((0, 1))
        )

        majorana_op = get_majorana_operator(fermion_op)
        expected_op = MajoranaOperator((0, 3), 2.0) + MajoranaOperator((1, 2, 3))
        self.assertTrue(majorana_op == expected_op)

    def test_get_majorana_operator_diagonalcoulomb(self):
        """Test get majorana from Diagonal Coulomb."""
        fermion_op = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)

        diagonal_ham = get_diagonal_coulomb_hamiltonian(fermion_op)

        self.assertTrue(get_majorana_operator(diagonal_ham) == get_majorana_operator(fermion_op))


class RaisesSympyExceptionTest(unittest.TestCase):
    def test_raises_sympy_expression(self):
        operator = FermionOperator('0^', sympy.Symbol('x'))
        with self.assertRaises(TypeError):
            check_no_sympy(operator)
