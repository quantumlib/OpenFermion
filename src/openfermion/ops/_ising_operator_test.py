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

import numpy

import pytest

from openfermion.ops._ising_operator import IsingOperator


def test_imul_inplace():
    qubit_op = IsingOperator("Z1")
    prev_id = id(qubit_op)
    qubit_op *= 3.
    assert id(qubit_op) == prev_id


@pytest.mark.parametrize("multiplier", [0.5, 0.6j, numpy.float64(2.303),
                                        numpy.complex128(-1j)])
def test_imul_scalar(multiplier):
    loc_op = ((1, 'Z'), (2, 'Z'))
    qubit_op = IsingOperator(loc_op)
    qubit_op *= multiplier
    assert qubit_op.terms[loc_op] == pytest.approx(multiplier)


def test_imul_qubit_op():
    op1 = IsingOperator(((0, 'Z'), (3, 'Z'), (8, 'Z'), (11, 'Z')), 3.j)
    op2 = IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 0.5)
    op1 *= op2
    correct_term = ((0, 'Z'), (1, 'Z'), (11, 'Z'))
    assert len(op1.terms) == 1
    assert correct_term in op1.terms


def test_imul_qubit_op_2():
    op3 = IsingOperator(((1, 'Z'), (0, 'Z')), -1j)
    op4 = IsingOperator(((1, 'Z'), (0, 'Z'), (2, 'Z')), -1.5)
    op3 *= op4
    op4 *= op3
    assert ((2, 'Z'),) in op3.terms
    assert op3.terms[((2, 'Z'),)] == 1.5j
    assert op4.terms[((0, 'Z'), (1, 'Z'))] == -2.25j


def test_imul_bidir():
    op_a = IsingOperator(((1, 'Z'), (0, 'Z')), -1j)
    op_b = IsingOperator(((1, 'Z'), (0, 'Z'), (2, 'Z')), -1.5)
    op_a *= op_b
    op_b *= op_a
    assert ((2, 'Z'),) in op_a.terms
    assert op_a.terms[((2, 'Z'),)] == 1.5j
    assert ((0, 'Z'), (1, 'Z')) in op_b.terms
    assert op_b.terms[((0, 'Z'), (1, 'Z'))] == -2.25j


def test_imul_bad_multiplier():
    operator = IsingOperator(((1, 'Z'), (0, 'Z')), -1j)
    with pytest.raises(TypeError):
        operator *= "1"


def test_mul_by_scalarzero():
    operator = IsingOperator(((1, 'Z'), (0, 'Z')), -1j) * 0
    assert ((0, 'Z'), (1, 'Z')) in operator.terms
    assert operator.terms[((0, 'Z'), (1, 'Z'))] == pytest.approx(0.0)


def test_mul_bad_multiplier():
    operator = IsingOperator(((1, 'Z'), (0, 'Z')), -1j)
    with pytest.raises(TypeError):
        operator = operator * "0.5"


def test_mul_out_of_place():
    op1 = IsingOperator(((0, 'Z'), (3, 'Z'), (8, 'Z'), (11, 'Z')), 3.j)
    op2 = IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 0.5)
    op3 = op1 * op2
    correct_coefficient = 1.5j
    correct_term = ((0, 'Z'), (1, 'Z'), (11, 'Z'))
    assert op1 == IsingOperator(((0, 'Z'), (3, 'Z'), (8, 'Z'), (11, 'Z')), 3.j)
    assert op2 == IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 0.5)
    assert op3 == IsingOperator(correct_term, correct_coefficient)


def test_mul_npfloat64():
    operator = IsingOperator(((1, 'Z'), (3, 'Z')), 0.5)
    res = operator * numpy.float64(0.5)
    assert res == IsingOperator(((1, 'Z'), (3, 'Z')), 0.5 * 0.5)


def test_mul_multiple_terms():
    operator = IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 0.5)
    operator += IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 1.2)
    operator += IsingOperator(((1, 'Z'), (3, 'Z'), (9, 'Z')), 1.4j)
    res = operator * operator
    correct = IsingOperator((), 1.7**2 - 1.4**2)
    correct += IsingOperator(((8, 'Z'), (9, 'Z')),
                             2j * 1.7 * 1.4)
    assert res == correct


def test_get_operators_empty():
    """Tests get_operators() with zero operator."""
    operator = IsingOperator.zero()

    operators = list(operator.get_operators())
    assert operators == []


def test_get_operators_one():
    """Tests get_operators() with an operator with a single term."""
    operator = IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 1)

    operators = list(operator.get_operators())
    assert operators == [operator]


def test_get_operators():
    """Tests get_operators() with an operator with two terms."""
    operator_00 = IsingOperator(((1, 'Z'), (3, 'Z'), (8, 'Z')), 1)
    operator_01 = IsingOperator(((2, 'Z'), (3, 'Z')), 1)
    sum_operator = operator_00 + operator_01

    operators = list(sum_operator.get_operators())
    assert operators in [[operator_00, operator_01],
                         [operator_01, operator_00]]


def test_get_operator_groups_empty():
    """Tests get_operator_groups() with zero operator."""
    operator = IsingOperator.zero()

    operators = list(operator.get_operator_groups(1))
    assert operators == []


# Utility functions.
def generate_operator(begin, end):
    """Returns a sum of Z operators at qubit [begin, end)."""
    operator = IsingOperator.zero()
    for i in range(begin, end):
        operator += IsingOperator(((i, 'Z'),), 1)
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
    return IsingOperator.accumulate(operators) == operator


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
