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
import scipy
import pytest
import cirq

import openfermion


def test_double_excitation_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = openfermion.DoubleExcitationGate(exponent=1.0, duration=numpy.pi / 2)


def test_double_excitation_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(
        openfermion.DoubleExcitationGate(exponent=1.5),
        openfermion.DoubleExcitationGate(exponent=-0.5),
        openfermion.DoubleExcitationGate(rads=-0.5 * numpy.pi),
        openfermion.DoubleExcitationGate(degs=-90),
        openfermion.DoubleExcitationGate(duration=-0.5 * numpy.pi / 2),
    )

    eq.add_equality_group(
        openfermion.DoubleExcitationGate(exponent=0.5),
        openfermion.DoubleExcitationGate(exponent=-1.5),
        openfermion.DoubleExcitationGate(rads=0.5 * numpy.pi),
        openfermion.DoubleExcitationGate(degs=90),
        openfermion.DoubleExcitationGate(duration=-1.5 * numpy.pi / 2),
    )

    eq.make_equality_group(lambda: openfermion.DoubleExcitationGate(exponent=0.0))
    eq.make_equality_group(lambda: openfermion.DoubleExcitationGate(exponent=0.75))


def test_double_excitation_consistency():
    openfermion.testing.assert_implements_consistent_protocols(openfermion.DoubleExcitation)


double_excitation_simulator_test_cases = [
    (
        openfermion.DoubleExcitation,
        1.0,
        numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.0,
        numpy.array([1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1]) / 4.0,
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        -1.0,
        numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.0,
        numpy.array([1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1]) / 4.0,
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        0.5,
        numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(8),
        numpy.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1j, 0, 0, 0]) / numpy.sqrt(8),
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        -0.5,
        numpy.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.0,
        numpy.array([1, -1, -1, -1j, -1, -1, 1, 1, 1, 1, 1, 1, 1j, 1, 1, 1]) / 4.0,
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        -1.0 / 7,
        numpy.array([1, 1j, -1j, -1, 1, 1j, -1j, -1, 1, 1j, -1j, -1, 1, 1j, -1j, -1]) / 4.0,
        numpy.array(
            [
                1,
                1j,
                -1j,
                -numpy.cos(numpy.pi / 7) - 1j * numpy.sin(numpy.pi / 7),
                1,
                1j,
                -1j,
                -1,
                1,
                1j,
                -1j,
                -1,
                numpy.cos(numpy.pi / 7) + 1j * numpy.sin(numpy.pi / 7),
                1j,
                -1j,
                -1,
            ]
        )
        / 4.0,
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        7.0 / 3,
        numpy.array(
            [
                0,
                0,
                0,
                2,
                (1 + 1j) / numpy.sqrt(2),
                (1 - 1j) / numpy.sqrt(2),
                -(1 + 1j) / numpy.sqrt(2),
                -1,
                1,
                1j,
                -1j,
                -1,
                1,
                1j,
                -1j,
                -1,
            ]
        )
        / 4.0,
        numpy.array(
            [
                0,
                0,
                0,
                1 + 1j * numpy.sqrt(3) / 2,
                (1 + 1j) / numpy.sqrt(2),
                (1 - 1j) / numpy.sqrt(2),
                -(1 + 1j) / numpy.sqrt(2),
                -1,
                1,
                1j,
                -1j,
                -1,
                0.5 + 1j * numpy.sqrt(3),
                1j,
                -1j,
                -1,
            ]
        )
        / 4.0,
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        0,
        numpy.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.0,
        numpy.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.0,
        5e-6,
    ),
    (
        openfermion.DoubleExcitation,
        0.25,
        numpy.array([1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1]) / numpy.sqrt(15),
        numpy.array(
            [
                1,
                0,
                0,
                +3j / numpy.sqrt(2) - numpy.sqrt(2),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3 / numpy.sqrt(2) - 1j * numpy.sqrt(2),
                0,
                0,
                1,
            ]
        )
        / numpy.sqrt(15),
        5e-6,
    ),
]


@pytest.mark.parametrize(
    'gate, exponent, initial_state, correct_state, atol', double_excitation_simulator_test_cases
)
def test_four_qubit_rotation_gates_on_simulator(gate, exponent, initial_state, correct_state, atol):
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(gate(a, b, c, d) ** exponent)
    result = circuit.final_state_vector(initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(result, correct_state, atol=atol)


def test_double_excitation_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    circuit = cirq.Circuit(openfermion.DoubleExcitation(a, b, c, d))
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───⇅───
      │
b: ───⇅───
      │
c: ───⇵───
      │
d: ───⇵───
""",
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(a, b, c, d) ** -0.5)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───⇅────────
      │
b: ───⇅────────
      │
c: ───⇵────────
      │
d: ───⇵^-0.5───
""",
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(a, c, b, d) ** 0.2)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───⇅───────
      │
b: ───⇵───────
      │
c: ───⇅───────
      │
d: ───⇵^0.2───
""",
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(d, b, a, c) ** 0.7)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───⇵───────
      │
b: ───⇅───────
      │
c: ───⇵───────
      │
d: ───⇅^0.7───
""",
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(d, b, a, c) ** 2.3)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───⇵───────
      │
b: ───⇅───────
      │
c: ───⇵───────
      │
d: ───⇅^0.3───
""",
    )


def test_double_excitation_gate_text_diagrams_no_unicode():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    circuit = cirq.Circuit(openfermion.DoubleExcitation(a, b, c, d))
    # pylint: disable=anomalous-backslash-in-string
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
a: ---/\ \/---
      |
b: ---/\ \/---
      |
c: ---\/ /\---
      |
d: ---\/ /\---
""",
        use_unicode_characters=False,
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(a, b, c, d) ** -0.5)
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
a: ---/\ \/--------
      |
b: ---/\ \/--------
      |
c: ---\/ /\--------
      |
d: ---\/ /\^-0.5---
""",
        use_unicode_characters=False,
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(a, c, b, d) ** 0.2)
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
a: ---/\ \/-------
      |
b: ---\/ /\-------
      |
c: ---/\ \/-------
      |
d: ---\/ /\^0.2---
""",
        use_unicode_characters=False,
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(d, b, a, c) ** 0.7)
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
a: ---\/ /\-------
      |
b: ---/\ \/-------
      |
c: ---\/ /\-------
      |
d: ---/\ \/^0.7---
""",
        use_unicode_characters=False,
    )

    circuit = cirq.Circuit(openfermion.DoubleExcitation(d, b, a, c) ** 2.3)
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
a: ---\/ /\-------
      |
b: ---/\ \/-------
      |
c: ---\/ /\-------
      |
d: ---/\ \/^0.3---
""",
        use_unicode_characters=False,
    )


@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_double_excitation_matches_fermionic_evolution(exponent):
    gate = openfermion.DoubleExcitation**exponent

    op = openfermion.FermionOperator('3^ 2^ 1 0')
    op += openfermion.hermitian_conjugated(op)
    matrix_op = openfermion.get_sparse_operator(op)

    time_evol_op = scipy.sparse.linalg.expm(-1j * matrix_op * exponent * numpy.pi)
    time_evol_op = time_evol_op.todense()

    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(gate), time_evol_op, atol=1e-7)
