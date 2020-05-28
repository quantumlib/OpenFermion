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

import pytest
import numpy

from openfermion import MajoranaOperator


def test_majorana_operator_init():
    op = MajoranaOperator((0, 7, 4, 1, 10))
    assert op.terms == {(0, 1, 4, 7, 10): -1.0}

    op = MajoranaOperator((0, 1, 0, 1, 0))
    assert op.terms == {(0,): -1.0}

    op = MajoranaOperator((3, 2, 5, 1, 3, 4))
    assert op.terms == {(1, 2, 4, 5): 1.0}

    op = MajoranaOperator((5, 10, 4, 3, 6, 9, 6))
    assert op.terms == {(3, 4, 5, 9, 10): -1.0}


def test_majorana_operator_commutes_with():
    a = MajoranaOperator((0, 1, 5))
    b = MajoranaOperator((1, 2, 7))
    c = MajoranaOperator((2, 3, 4))
    d = MajoranaOperator((0, 3, 6))

    assert a.commutes_with(b)
    assert not a.commutes_with(c)
    assert a.commutes_with(d)
    assert b.commutes_with(c)
    assert not b.commutes_with(d)
    assert c.commutes_with(d)
    assert (a+c).commutes_with(b+d)

    e = MajoranaOperator((0, 1, 1, 1, 4, 5))
    f = MajoranaOperator((0, 1, 1, 4))

    assert e.commutes_with(f)

    with pytest.raises(TypeError):
        _ = e.commutes_with(0)


def test_majorana_operator_with_basis_rotated_by():
    H = numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2)

    a = MajoranaOperator((0, 1), 2.0)
    op = a.with_basis_rotated_by(H)
    assert op == MajoranaOperator.from_dict({(0, 1): -2.0})

    b = MajoranaOperator((0,), 2.0)
    op = b.with_basis_rotated_by(H)
    assert op == MajoranaOperator.from_dict({(0,): numpy.sqrt(2),
                                             (1,): numpy.sqrt(2)})

    c = MajoranaOperator((1,), 2.0)
    op = c.with_basis_rotated_by(H)
    assert op == MajoranaOperator.from_dict({(0,): numpy.sqrt(2),
                                             (1,): -numpy.sqrt(2)})

    P = numpy.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    d = MajoranaOperator((0, 1, 2)) + MajoranaOperator((1, 2))
    op = d.with_basis_rotated_by(P)
    assert op == MajoranaOperator.from_dict({(0, 1, 2): 1.0,
                                             (0, 1): 1.0})

    with pytest.raises(ValueError):
        _ = a.with_basis_rotated_by(2 * H)


def test_majorana_operator_add_subtract():
    a = MajoranaOperator((0, 2, 3), -1.25)
    b = MajoranaOperator((0, 2, 3), -0.5)
    c = MajoranaOperator((1, 5, 7), 4.75)
    d = MajoranaOperator((3, 5, 7), 2.25)

    a += c
    assert a.terms == {(0, 2, 3): -1.25,
                       (1, 5, 7): 4.75}

    a -= d
    assert a.terms == {(0, 2, 3): -1.25,
                       (1, 5, 7): 4.75,
                       (3, 5, 7): 2.25}

    a += MajoranaOperator((0, 2, 3), 0.5)
    assert a.terms == {(0, 2, 3): -.75,
                       (1, 5, 7): 4.75,
                       (3, 5, 7): 2.25}

    a -= MajoranaOperator((0, 2, 3), 0.25)
    assert a.terms == {(0, 2, 3): -1.0,
                       (1, 5, 7): 4.75,
                       (3, 5, 7): 2.25}

    assert (a + b).terms == {(0, 2, 3): -1.5,
                             (1, 5, 7): 4.75,
                             (3, 5, 7): 2.25}

    assert (a - b).terms == {(0, 2, 3): -0.5,
                             (1, 5, 7): 4.75,
                             (3, 5, 7): 2.25}

    with pytest.raises(TypeError):
        _ = a + 0

    with pytest.raises(TypeError):
        a += 0

    with pytest.raises(TypeError):
        _ = a - 0

    with pytest.raises(TypeError):
        a -= 0


def test_majorana_operator_multiply():
    a = MajoranaOperator((0, 1, 5), 1.5) + MajoranaOperator((1, 2, 7), -0.5)
    b = MajoranaOperator((2, 3, 4), 1.75) - MajoranaOperator((0, 3, 6), 0.25)
    assert (a * a).terms == {(): -2.5,
                             (0, 2, 5, 7): -1.5}
    assert (a * b).terms == {(1, 3, 4, 7): 0.875,
                             (0, 1, 2, 3, 6, 7): -0.125,
                             (1, 3, 5, 6): 0.375,
                             (0, 1, 2, 3, 4, 5): -2.625}
    assert (2 * a).terms == (a * 2).terms == {(0, 1, 5): 3.0,
                                              (1, 2, 7): -1.0}

    a *= 2
    a *= MajoranaOperator(())
    assert a.terms == {(0, 1, 5): 3.0,
                       (1, 2, 7): -1.0}

    with pytest.raises(TypeError):
        _ = a * 'a'

    with pytest.raises(TypeError):
        _ = 'a' * a

    with pytest.raises(TypeError):
        a *= 'a'


def test_majorana_operator_pow():
    a = MajoranaOperator((0, 1, 5), 1.5) + MajoranaOperator((1, 2, 7), -0.5)
    assert (a**2).terms == {(): -2.5,
                            (0, 2, 5, 7): -1.5}

    with pytest.raises(TypeError):
        _ = a**-1

    with pytest.raises(TypeError):
        _ = a**'a'


def test_majorana_operator_divide():
    a = MajoranaOperator((0, 1, 5), 1.5) + MajoranaOperator((1, 2, 7), -0.5)
    assert (a / 2).terms == {(0, 1, 5): 0.75,
                             (1, 2, 7): -0.25}

    a /= 2
    assert a.terms == {(0, 1, 5): 0.75,
                       (1, 2, 7): -0.25}

    with pytest.raises(TypeError):
        _ = a / 'a'

    with pytest.raises(TypeError):
        a /= 'a'


def test_majorana_operator_neg():
    a = MajoranaOperator((0, 1, 5), 1.5) + MajoranaOperator((1, 2, 7), -0.5)
    assert (-a).terms == {(0, 1, 5): -1.5,
                          (1, 2, 7): 0.5}


def test_majorana_operator_eq():
    a = MajoranaOperator((0, 1, 5), 1.5) + MajoranaOperator((1, 2, 7), -0.5)
    b = (MajoranaOperator((0, 1, 5), 1.5) +
         MajoranaOperator((1, 2, 7), -0.5) +
         MajoranaOperator((3, 4, 5), 0.0))
    c = (MajoranaOperator((0, 1, 5), 1.5) +
         MajoranaOperator((1, 2, 7), -0.5) +
         MajoranaOperator((3, 4, 5), 0.1))
    d = MajoranaOperator((0, 1, 5), 1.75) + MajoranaOperator((1, 2, 7), -0.75)
    e = MajoranaOperator((0, 1, 5), 1.5) - MajoranaOperator((0, 3, 6), 0.25)

    assert a == b
    assert a != c
    assert a != d
    assert a != e

    assert a != 0


def test_majorana_operator_str():
    zero = MajoranaOperator()
    one = MajoranaOperator(())
    still_zero = MajoranaOperator((0, 1, 5), 0.0)
    a = MajoranaOperator((0, 1, 5), 1.5)
    b = MajoranaOperator((1, 2, 7), -0.5)

    assert str(zero) == '0'
    assert str(one) == '1.0 ()'
    assert str(still_zero) == '0'
    assert str(a) == '1.5 (0, 1, 5)'
    assert str(b) == '-0.5 (1, 2, 7)'
    assert str(a+b) == """1.5 (0, 1, 5) +
-0.5 (1, 2, 7)"""


def test_majorana_operator_repr():
    a = MajoranaOperator((0, 1, 5), 1.5)
    assert repr(a) == 'MajoranaOperator.from_dict(terms={(0, 1, 5): 1.5})'
