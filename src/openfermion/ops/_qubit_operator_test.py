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

import numpy
import pytest

from openfermion.ops._qubit_operator import (_PAULI_OPERATOR_PRODUCTS,
                                             QubitOperator)


def test_pauli_operator_product():
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


def test_init_simplify():
    assert QubitOperator("X0 X0") == QubitOperator.identity()
    assert QubitOperator("X1 X1") == QubitOperator.identity()
    assert QubitOperator("Y0 Y0") == QubitOperator.identity()
    assert QubitOperator("Z0 Z0") == QubitOperator.identity()
    assert QubitOperator("X0 Y0") == QubitOperator("Z0", coefficient=1j)
    assert QubitOperator("Y0 X0") == QubitOperator("Z0", coefficient=-1j)
    assert QubitOperator("X0 Y0 Z0") == 1j * QubitOperator.identity()
    assert QubitOperator("Y0 Z0 X0") == 1j * QubitOperator.identity()
    assert QubitOperator("Z0 Y0 X0") == -1j * QubitOperator.identity()
    assert QubitOperator("X1 Y0 X1") == QubitOperator("Y0")
    assert QubitOperator("Y1 Y1 Y1") == QubitOperator("Y1")
    assert QubitOperator("Y2 Y2 Y2 Y2") == QubitOperator.identity()
    assert QubitOperator("Y3 Y3 Y3 Y3 Y3") == QubitOperator("Y3")
    assert QubitOperator("Y4 Y4 Y4 Y4 Y4 Y4") == QubitOperator.identity()
    assert QubitOperator("X0 Y1 Y0 X1") == QubitOperator("Z0 Z1")
    assert QubitOperator("X0 Y1 Z3 X2 Z3 Y0") == QubitOperator("Z0 Y1 X2",
            coefficient=1j)


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
    assert op4.terms[((0, 'X'), (1, 'Y'))] == -2.25j


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
    operator = QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    with pytest.raises(TypeError):
        operator *= "1"


def test_mul_by_scalarzero():
    operator = QubitOperator(((1, 'Y'), (0, 'X')), -1j) * 0
    assert ((0, 'X'), (1, 'Y')) in operator.terms
    assert operator.terms[((0, 'X'), (1, 'Y'))] == pytest.approx(0.0)


def test_mul_bad_multiplier():
    operator = QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    with pytest.raises(TypeError):
        operator = operator * "0.5"


def test_mul_out_of_place():
    op1 = QubitOperator(((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j)
    op2 = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op3 = op1 * op2
    correct_coefficient = 1.j * 3.0j * 0.5
    correct_term = ((0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X'))
    assert op1 == QubitOperator(((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j)
    assert op2 == QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    assert op3 == QubitOperator(correct_term, correct_coefficient)


def test_mul_npfloat64():
    operator = QubitOperator(((1, 'X'), (3, 'Y')), 0.5)
    res = operator * numpy.float64(0.5)
    assert res == QubitOperator(((1, 'X'), (3, 'Y')), 0.5 * 0.5)


def test_mul_multiple_terms():
    operator = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    operator += QubitOperator(((1, 'Z'), (3, 'X'), (8, 'Z')), 1.2)
    operator += QubitOperator(((1, 'Z'), (3, 'Y'), (9, 'Z')), 1.4j)
    res = operator * operator
    correct = QubitOperator((), 0.5**2 + 1.2**2 + 1.4j**2)
    correct += QubitOperator(((1, 'Y'), (3, 'Z')),
                             2j * 1j * 0.5 * 1.2)
    assert res == correct


def test_renormalize_error():
    operator = QubitOperator()
    with pytest.raises(ZeroDivisionError):
        operator.renormalize()


def test_renormalize():
    operator = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 1)
    operator += QubitOperator(((2, 'Z'), (3, 'Y')), 1)
    operator.renormalize()
    for term in operator.terms:
        assert operator.terms[term] == pytest.approx(1/numpy.sqrt(2.))
    assert operator.induced_norm(2) == pytest.approx(1.)


def test_get_operators_empty():
    """Tests get_operators() with zero operator."""
    operator = QubitOperator.zero()

    operators = list(operator.get_operators())
    assert operators == []


def test_get_operators_one():
    """Tests get_operators() with an operator with a single term."""
    operator = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 1)

    operators = list(operator.get_operators())
    assert operators == [operator]


def test_get_operators():
    """Tests get_operators() with an operator with two terms."""
    operator_00 = QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 1)
    operator_01 = QubitOperator(((2, 'Z'), (3, 'Y')), 1)
    sum_operator = operator_00 + operator_01

    operators = list(sum_operator.get_operators())
    assert operators in [[operator_00, operator_01],
                         [operator_01, operator_00]]


def test_get_operator_groups_empty():
    """Tests get_operator_groups() with zero operator."""
    operator = QubitOperator.zero()

    operators = list(operator.get_operator_groups(1))
    assert operators == []


# Utility functions.
def generate_operator(begin, end):
    """Returns a sum of Z operators at qubit [begin, end)."""
    operator = QubitOperator.zero()
    for i in range(begin, end):
        operator += QubitOperator(((i, 'Z'),), 1)
    return operator


def check_length(operators, lens):
    """Checks length operator is the same to lens."""
    if len(operators) != len(lens):
        return False

    for operator, length in zip(operators, lens):
        if len(operator.terms) != length:
            return False
    return True


def check_sum(operators, operator):
    """Checks sum of operators matches to operator."""
    return QubitOperator.accumulate(operators) == operator


# Tests for get_operator_groups().
def test_get_operator_groups_zero():
    """Tests get_operator_groups() with one group."""
    operator = generate_operator(0, 20)
    operator_groups = list(operator.get_operator_groups(0))

    # Using 1 group instead.
    assert check_length(operator_groups, [20])
    assert check_sum(operator_groups, operator)

    # Not using 0 groups.
    assert not check_length(operator_groups, [])
    assert not check_length(operator_groups, [0])
    assert not check_sum(operator_groups, operator * 2)


def test_get_operator_groups_one():
    """Tests get_operator_groups() with one group."""
    operator = generate_operator(0, 20)
    operator_groups = list(operator.get_operator_groups(1))

    assert check_length(operator_groups, [20])
    assert check_sum(operator_groups, operator)


def test_get_operator_groups_two():
    """Tests get_operator_groups() with two groups."""
    operator = generate_operator(0, 20)
    operator_groups = list(operator.get_operator_groups(2))

    assert check_length(operator_groups, [10, 10])
    assert check_sum(operator_groups, operator)


def test_get_operator_groups_three():
    """Tests get_operator_groups() with two groups."""
    operator = generate_operator(0, 20)
    operator_groups = list(operator.get_operator_groups(3))

    assert check_length(operator_groups, [7, 7, 6])
    assert check_sum(operator_groups, operator)


def test_get_operator_groups_six():
    """Tests get_operator_groups() with two groups."""
    operator = generate_operator(0, 20)
    operator_groups = list(operator.get_operator_groups(6))

    assert check_length(operator_groups, [4, 4, 3, 3, 3, 3])
    assert check_sum(operator_groups, operator)
