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


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_compare_ryxxy_to_cirq_equivalent(rads):
    old_gate = ofc.Ryxxy(rads=rads)
    new_gate = cirq.givens(angle_rads=rads)
    np.testing.assert_allclose(cirq.unitary(old_gate), cirq.unitary(new_gate))


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


def test_common_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    circuit = cirq.Circuit(ofc.FSWAP(a, b),
                           ofc.FSWAP(a, b)**0.5)
    cirq.testing.assert_has_diagram(
        circuit, """
a: ───×ᶠ───×ᶠ───────
      │    │        
b: ───×ᶠ───×ᶠ^0.5───
""")

    cirq.testing.assert_has_diagram(circuit,
                                    """
a: ---fswap---fswap-------
      |       |           
b: ---fswap---fswap^0.5---
""",
                                    use_unicode_characters=False)
