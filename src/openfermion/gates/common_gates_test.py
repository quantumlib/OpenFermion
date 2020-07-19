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
import warnings

import numpy as np
from scipy.linalg import expm, kron
import sympy
import cirq
import pytest

import openfermioncirq as ofc
from openfermioncirq._compat_test import deprecated_test


def test_fswap_interchangeable():
    a, b = cirq.LineQubit.range(2)
    assert ofc.FSWAP(a, b) == ofc.FSWAP(b, a)


def test_fswap_inverse():
    assert ofc.FSWAP**-1 == ofc.FSWAP


def test_fswap_str():
    assert str(ofc.FSWAP) == 'FSWAP'
    assert str(ofc.FSWAP**0.5) == 'FSWAP**0.5'
    assert str(ofc.FSWAP**-0.25) == 'FSWAP**-0.25'


def test_fswap_repr():
    assert repr(ofc.FSWAP) == 'ofc.FSWAP'
    assert repr(ofc.FSWAP**0.5) == '(ofc.FSWAP**0.5)'
    assert repr(ofc.FSWAP**-0.25) == '(ofc.FSWAP**-0.25)'

    ofc.testing.assert_equivalent_repr(ofc.FSWAP)
    ofc.testing.assert_equivalent_repr(ofc.FSWAP**0.5)
    ofc.testing.assert_equivalent_repr(ofc.FSWAP**-0.25)


def test_fswap_matrix():
    np.testing.assert_allclose(
        cirq.unitary(ofc.FSWAP),
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(ofc.FSWAP**0.5),
        np.array([[1, 0, 0, 0], [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                  [0, 0.5 - 0.5j, 0.5 + 0.5j, 0], [0, 0, 0, 1j]]))

    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        val=ofc.FSWAP, exponents=[1])


@deprecated_test
def test_xxyy_init():
    assert ofc.XXYYPowGate(exponent=0.5).exponent == 0.5
    assert ofc.XXYYPowGate(exponent=1.5).exponent == 1.5
    assert ofc.XXYYPowGate(exponent=5).exponent == 5


@deprecated_test
def test_xxyy_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(ofc.XXYYPowGate(exponent=3.5),
                          ofc.XXYYPowGate(exponent=-0.5))

    eq.add_equality_group(ofc.XXYYPowGate(exponent=1.5),
                          ofc.XXYYPowGate(exponent=-2.5))

    eq.make_equality_group(lambda: ofc.XXYYPowGate(exponent=0))
    eq.make_equality_group(lambda: ofc.XXYYPowGate(exponent=0.5))


@deprecated_test
def test_xxyy_interchangeable():
    a, b = cirq.LineQubit(0), cirq.LineQubit(1)
    assert ofc.XXYY(a, b) == ofc.XXYY(b, a)


@deprecated_test
def test_xxyy_repr():
    assert repr(ofc.XXYYPowGate(exponent=1)) == 'XXYY'
    assert repr(ofc.XXYYPowGate(exponent=0.5)) == 'XXYY**0.5'


@deprecated_test
@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_xxyy_decompose(exponent):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        ofc.XXYY**exponent)


@deprecated_test
def test_xxyy_matrix():
    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        ofc.XXYY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1,
                   sympy.Symbol('s')])

    np.testing.assert_allclose(cirq.unitary(ofc.XXYYPowGate(exponent=2)),
                               np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                         [0, 0, -1, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ofc.XXYYPowGate(exponent=1)),
                               np.array([[1, 0, 0, 0], [0, 0, -1j, 0],
                                         [0, -1j, 0, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ofc.XXYYPowGate(exponent=0)),
                               np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                         [0, 0, 1, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ofc.XXYYPowGate(exponent=-1)),
                               np.array([[1, 0, 0, 0], [0, 0, 1j, 0],
                                         [0, 1j, 0, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    np.testing.assert_allclose(cirq.unitary(ofc.XXYYPowGate(exponent=0.25)),
                               expm(-1j * np.pi * 0.25 * (XX + YY) / 4))


@deprecated_test
@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_compare_xxyy_to_cirq_equivalent(exponent):
    old_gate = ofc.XXYYPowGate(exponent=exponent)
    new_gate = cirq.ISwapPowGate(exponent=-exponent)
    np.testing.assert_allclose(cirq.unitary(old_gate), cirq.unitary(new_gate))


@deprecated_test
def test_yxxy_init():
    assert ofc.YXXYPowGate(exponent=0.5).exponent == 0.5
    assert ofc.YXXYPowGate(exponent=1.5).exponent == 1.5
    assert ofc.YXXYPowGate(exponent=5).exponent == 5


@deprecated_test
def test_yxxy_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(ofc.YXXYPowGate(exponent=3.5),
                          ofc.YXXYPowGate(exponent=-0.5))

    eq.add_equality_group(ofc.YXXYPowGate(exponent=1.5),
                          ofc.YXXYPowGate(exponent=-2.5))

    eq.make_equality_group(lambda: ofc.YXXYPowGate(exponent=0))
    eq.make_equality_group(lambda: ofc.YXXYPowGate(exponent=0.5))


@deprecated_test
def test_yxxy_repr():
    assert repr(ofc.YXXYPowGate(exponent=1)) == 'YXXY'
    assert repr(ofc.YXXYPowGate(exponent=0.5)) == 'YXXY**0.5'


@deprecated_test
@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_yxxy_decompose(exponent):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        ofc.YXXY**exponent)


@deprecated_test
def test_yxxy_matrix():
    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        ofc.YXXY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1,
                   sympy.Symbol('s')])

    np.testing.assert_allclose(cirq.unitary(ofc.YXXYPowGate(exponent=2)),
                               np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                         [0, 0, -1, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ofc.YXXYPowGate(exponent=1)),
                               np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                         [0, 1, 0, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ofc.YXXYPowGate(exponent=0)),
                               np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                         [0, 0, 1, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ofc.YXXYPowGate(exponent=-1)),
                               np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                         [0, -1, 0, 0], [0, 0, 0, 1]]),
                               atol=1e-8)

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    YX = kron(Y, X)
    XY = kron(X, Y)
    np.testing.assert_allclose(cirq.unitary(ofc.YXXYPowGate(exponent=0.25)),
                               expm(-1j * np.pi * 0.25 * (YX - XY) / 4))


@deprecated_test
@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_compare_yxxy_to_cirq_equivalent(exponent):
    old_gate = ofc.YXXYPowGate(exponent=exponent)
    new_gate = cirq.PhasedISwapPowGate(exponent=exponent)
    np.testing.assert_allclose(cirq.unitary(old_gate), cirq.unitary(new_gate))


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_compare_ryxxy_to_cirq_equivalent(rads):
    old_gate = ofc.Ryxxy(rads=rads)
    new_gate = cirq.givens(angle_rads=rads)
    np.testing.assert_allclose(cirq.unitary(old_gate), cirq.unitary(new_gate))


# Deprecated test
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    args = [
        (ofc.XXYY, 1.0, np.array([0, 1, 1, 0]) / np.sqrt(2),
         np.array([0, -1j, -1j, 0]) / np.sqrt(2), 1e-7),
        (ofc.XXYY, 0.5, np.array([1, 1, 0, 0]) / np.sqrt(2),
         np.array([1 / np.sqrt(2), 0.5, -0.5j, 0]), 1e-7),
        (ofc.XXYY, -0.5, np.array([1, 1, 0, 0]) / np.sqrt(2),
         np.array([1 / np.sqrt(2), 0.5, 0.5j, 0]), 1e-7),
        (ofc.YXXY, 1.0, np.array([0, 1, 1, 0]) / np.sqrt(2),
         np.array([0, 1, -1, 0]) / np.sqrt(2), 1e-7),
        (ofc.YXXY, 0.5, np.array([0, 1, 1, 0]) / np.sqrt(2),
         np.array([0, 0, 1, 0]), 1e-7),
        (ofc.YXXY, -0.5, np.array([0, 1, 1, 0]) / np.sqrt(2),
         np.array([0, 1, 0, 0]), 1e-7),
    ]


@deprecated_test
@pytest.mark.parametrize('gate, exponent, initial_state, correct_state, atol',
                         args)
def test_two_qubit_rotation_gates_on_simulator(gate, exponent, initial_state,
                                               correct_state, atol):
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(gate(a, b)**exponent)
    result = circuit.final_wavefunction(initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(result,
                                                    correct_state,
                                                    atol=atol)


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_rxxyy_unitary(rads):
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    np.testing.assert_allclose(cirq.unitary(ofc.Rxxyy(rads)),
                               expm(-1j * rads * (XX + YY) / 2),
                               atol=1e-8)


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_ryxxy_unitary(rads):
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    YX = kron(Y, X)
    XY = kron(X, Y)
    np.testing.assert_allclose(cirq.unitary(ofc.Ryxxy(rads)),
                               expm(-1j * rads * (YX - XY) / 2),
                               atol=1e-8)


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_rzz_unitary(rads):
    ZZ = np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(cirq.unitary(ofc.Rzz(rads)),
                               expm(-1j * ZZ * rads))


@deprecated_test
def test_common_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    circuit = cirq.Circuit(ofc.FSWAP(a, b),
                           ofc.FSWAP(a, b)**0.5, ofc.XXYY(a, b),
                           ofc.YXXY(a, b))
    cirq.testing.assert_has_diagram(
        circuit, """
a: ───×ᶠ───×ᶠ───────XXYY───YXXY───
      │    │        │      │
b: ───×ᶠ───×ᶠ^0.5───XXYY───#2─────
""")

    cirq.testing.assert_has_diagram(circuit,
                                    """
a: ---fswap---fswap-------XXYY---YXXY---
      |       |           |      |
b: ---fswap---fswap^0.5---XXYY---#2-----
""",
                                    use_unicode_characters=False)

    circuit = cirq.Circuit(ofc.XXYY(a, b)**0.5, ofc.YXXY(a, b)**0.5)
    cirq.testing.assert_has_diagram(
        circuit, """
a: ───XXYY───────YXXY─────
      │          │
b: ───XXYY^0.5───#2^0.5───
""")
