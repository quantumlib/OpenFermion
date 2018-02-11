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

"""Tests for _qubit_operator.py."""
import copy

import numpy
import pytest

from openfermion.ops._qubit_operator import (_PAULI_OPERATOR_PRODUCTS,
                                             QubitOperator,
                                             QubitOperatorError)


def test_pauli_operator_product_unchanged():
    correct = {('I', 'I'): (1., 'I'),
               ('I', 'X'): (1., 'X'),
               ('X', 'I'): (1., 'X'),
               ('I', 'Y'): (1., 'Y'),
               ('Y', 'I'): (1., 'Y'),
               ('I', 'Z'): (1., 'Z'),
               ('Z', 'I'): (1., 'Z'),
               ('X', 'X'): (1., 'I'),
               ('Y', 'Y'): (1., 'I'),
               ('Z', 'Z'): (1., 'I'),
               ('X', 'Y'): (1.j, 'Z'),
               ('X', 'Z'): (-1.j, 'Y'),
               ('Y', 'X'): (-1.j, 'Z'),
               ('Y', 'Z'): (1.j, 'X'),
               ('Z', 'X'): (1.j, 'Y'),
               ('Z', 'Y'): (-1.j, 'X')}
    assert _PAULI_OPERATOR_PRODUCTS == correct


def test_imul_inplace():
    qubit_op = QubitOperator("X1")
    prev_id = id(qubit_op)
    qubit_op *= 3.
    assert id(qubit_op) == prev_id


@pytest.mark.parametrize("multiplier", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_imul_scalar(multiplier):
    loc_op = ((1, 'X'), (2, 'Y'))
    qubit_op = QubitOperator(loc_op)
    qubit_op *= multiplier
    assert qubit_op.terms[loc_op] == pytest.approx(multiplier)


def test_imul_qubit_op():
    op1 = QubitOperator(((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j)
    op2 = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op1 *= op2
    correct_coefficient = 1.j * 3.0j * 0.5
    correct_term = ((0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X'))
    assert len(op1.terms) == 1
    assert correct_term in op1.terms


def test_imul_qubit_op_2():
    op3 = QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    op4 = QubitOperator(((1, 'Y'), (0, 'X'), (2, 'Z')), -1.5)
    op3 *= op4
    op4 *= op3
    assert ((2, 'Z'),) in op3.terms
    assert op3.terms[((2, 'Z'),)] == 1.5j


def test_imul_bidir():
    op_a = QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    op_b = QubitOperator(((1, 'Y'), (0, 'X'), (2, 'Z')), -1.5)
    op_a *= op_b
    op_b *= op_a
    assert ((2, 'Z'),) in op_a.terms
    assert op_a.terms[((2, 'Z'),)] == 1.5j
    assert ((0, 'X'), (1, 'Y')) in op_b.terms
    assert op_b.terms[((0, 'X'), (1, 'Y'))] == -2.25j


def test_imul_bad_multiplier():
    op = QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    with pytest.raises(TypeError):
        op *= "1"


def test_mul_by_scalarzero():
    op = QubitOperator(((1, 'Y'), (0, 'X')), -1j) * 0
    assert ((0, 'X'), (1, 'Y')) in op.terms
    assert op.terms[((0, 'X'), (1, 'Y'))] == pytest.approx(0.0)


def test_mul_bad_multiplier():
    op = QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    with pytest.raises(TypeError):
        op = op * "0.5"


def test_mul_out_of_place():
    op1 = QubitOperator(((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j)
    op2 = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op3 = op1 * op2
    correct_coefficient = 1.j * 3.0j * 0.5
    correct_term = ((0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X'))
    assert op1.isclose(QubitOperator(
        ((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j))
    assert op2.isclose(QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5))
    assert op3.isclose(QubitOperator(correct_term, correct_coefficient))


def test_mul_npfloat64():
    op = QubitOperator(((1, 'X'), (3, 'Y')), 0.5)
    res = op * numpy.float64(0.5)
    assert res.isclose(QubitOperator(((1, 'X'), (3, 'Y')), 0.5 * 0.5))


def test_mul_multiple_terms():
    op = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op += QubitOperator(((1, 'Z'), (3, 'X'), (8, 'Z')), 1.2)
    op += QubitOperator(((1, 'Z'), (3, 'Y'), (9, 'Z')), 1.4j)
    res = op * op
    correct = QubitOperator((), 0.5**2 + 1.2**2 + 1.4j**2)
    correct += QubitOperator(((1, 'Y'), (3, 'Z')),
                             2j * 1j * 0.5 * 1.2)
    assert res.isclose(correct)


def test_renormalize_error():
    op = QubitOperator()
    with pytest.raises(ZeroDivisionError):
        op.renormalize()


def test_renormalize():
    op = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 1)
    op += QubitOperator(((2, 'Z'), (3, 'Y')), 1)
    op.renormalize()
    for term in op.terms:
        assert op.terms[term] == pytest.approx(1/numpy.sqrt(2.))
    assert op.induced_norm(2) == pytest.approx(1.)
