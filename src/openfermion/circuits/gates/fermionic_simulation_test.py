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

import itertools
from typing import cast, Tuple

import numpy as np
import pytest
import scipy.linalg as la
import sympy
import cirq
import cirq.contrib.acquaintance as cca

import openfermion
from openfermion.circuits.gates.fermionic_simulation import (
    sum_of_interaction_operator_gate_generators,
    state_swap_eigen_component,
)


def test_state_swap_eigen_component_args():
    with pytest.raises(TypeError):
        state_swap_eigen_component(0, '12', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '01', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '10', 0)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '100', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', 'ab', 1)


@pytest.mark.parametrize('index_pair,n_qubits', [
    ((0, 1), 2),
    ((0, 3), 2),
])
def test_state_swap_eigen_component(index_pair, n_qubits):
    state_pair = tuple(format(i, '0' + str(n_qubits) + 'b') for i in index_pair)
    i, j = index_pair
    dim = 2**n_qubits
    for sign in (-1, 1):
        actual_component = state_swap_eigen_component(state_pair[0],
                                                      state_pair[1], sign)
        expected_component = np.zeros((dim, dim))
        expected_component[i, i] = expected_component[j, j] = 0.5
        expected_component[i, j] = expected_component[j, i] = sign * 0.5
        assert np.allclose(actual_component, expected_component)


@pytest.mark.parametrize('n_modes, seed',
                         [(7, np.random.randint(1 << 30)) for _ in range(2)])
def test_interaction_operator_interconversion(n_modes, seed):
    operator = openfermion.random_interaction_operator(n_modes,
                                                       real=False,
                                                       seed=seed)
    gates = openfermion.fermionic_simulation_gates_from_interaction_operator(
        operator)
    other_operator = sum_of_interaction_operator_gate_generators(n_modes, gates)
    operator = openfermion.normal_ordered(operator)
    other_operator = openfermion.normal_ordered(other_operator)
    assert operator == other_operator


def test_interaction_operator_from_bad_gates():
    for gates in [{(): 'bad'}, {(0,): cirq.X}]:
        with pytest.raises(TypeError):
            sum_of_interaction_operator_gate_generators(5, gates)


def random_real(size=None, mag=20):
    return np.random.uniform(-mag, mag, size)


def random_complex(size=None, mag=20):
    return random_real(size, mag) + 1j * random_real(size, mag)


def random_fermionic_simulation_gate(order):
    exponent = random_real()
    if order == 2:
        weights = (random_complex(), random_real())
        return openfermion.QuadraticFermionicSimulationGate(weights,
                                                            exponent=exponent)
    weights = random_complex(3)
    if order == 3:
        return openfermion.CubicFermionicSimulationGate(weights,
                                                        exponent=exponent)
    if order == 4:
        return openfermion.QuarticFermionicSimulationGate(weights,
                                                          exponent=exponent)


def assert_symbolic_decomposition_consistent(gate):
    expected_unitary = cirq.unitary(gate)

    weights = tuple(sympy.Symbol(f'w{i}') for i in range(gate.num_weights()))
    exponent = sympy.Symbol('t')
    symbolic_gate = type(gate)(weights, exponent=exponent)
    qubits = cirq.LineQubit.range(gate.num_qubits())
    circuit = cirq.Circuit(symbolic_gate._decompose_(qubits))
    resolver = {'t': gate.exponent}
    for i, w in enumerate(gate.weights):
        resolver[f'w{i}'] = w
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    decomp_unitary = resolved_circuit.unitary(qubit_order=qubits)

    assert np.allclose(expected_unitary, decomp_unitary)


def assert_generators_consistent(gate):
    qubit_generator = gate.qubit_generator_matrix
    qubit_generator_from_fermion_generator = (super(
        type(gate), gate).qubit_generator_matrix)
    assert np.allclose(qubit_generator, qubit_generator_from_fermion_generator)


def assert_resolution_consistent(gate):
    weight_names = [f'w{i}' for i in range(gate.num_weights())]
    weight_params = [sympy.Symbol(w) for w in weight_names]
    resolver = dict(zip(weight_names, gate.weights))
    resolver['s'] = gate._global_shift
    resolver['e'] = gate._exponent
    symbolic_gate = type(gate)(weight_params,
                               exponent=sympy.Symbol('e'),
                               global_shift=sympy.Symbol('s'))
    resolved_gate = cirq.resolve_parameters(symbolic_gate, resolver)
    assert resolved_gate == gate


def assert_fswap_consistent(gate):
    gate = gate.__copy__()
    n_qubits = gate.num_qubits()
    for i in range(n_qubits - 1):
        fswap = cirq.kron(np.eye(1 << i), cirq.unitary(openfermion.FSWAP),
                          np.eye(1 << (n_qubits - i - 2)))
        assert fswap.shape == (1 << n_qubits,) * 2
        generator = gate.qubit_generator_matrix
        fswapped_generator = np.linalg.multi_dot([fswap, generator, fswap])
        gate.fswap(i)
        assert np.allclose(gate.qubit_generator_matrix, fswapped_generator)
    for i in (-1, n_qubits):
        with pytest.raises(ValueError):
            gate.fswap(i)


def assert_permute_consistent(gate):
    gate = gate.__copy__()
    n_qubits = gate.num_qubits()
    qubits = cirq.LineQubit.range(n_qubits)
    for pos in itertools.permutations(range(n_qubits)):
        permuted_gate = gate.__copy__()
        gate.permute(pos)
        assert permuted_gate.permuted(pos) == gate
        actual_unitary = cirq.unitary(permuted_gate)

        ops = [
            cca.LinearPermutationGate(n_qubits, dict(zip(range(n_qubits), pos)),
                                      openfermion.FSWAP)(*qubits),
            gate(*qubits),
            cca.LinearPermutationGate(n_qubits, dict(zip(pos, range(n_qubits))),
                                      openfermion.FSWAP)(*qubits)
        ]
        circuit = cirq.Circuit(ops)
        expected_unitary = cirq.unitary(circuit)
        assert np.allclose(actual_unitary, expected_unitary)

    with pytest.raises(ValueError):
        gate.permute(range(1, n_qubits))
    with pytest.raises(ValueError):
        gate.permute([1] * n_qubits)


def assert_interaction_operator_consistent(gate):
    interaction_op = gate.interaction_operator_generator()
    other_gate = gate.from_interaction_operator(operator=interaction_op)
    if other_gate is None:
        assert np.allclose(gate.weights, 0)
    else:
        assert cirq.approx_eq(gate, other_gate)
    interaction_op = openfermion.normal_ordered(interaction_op)
    other_interaction_op = openfermion.InteractionOperator.zero(
        interaction_op.n_qubits)
    super(type(gate),
          gate).interaction_operator_generator(operator=other_interaction_op)
    other_interaction_op = openfermion.normal_ordered(interaction_op)
    assert interaction_op == other_interaction_op

    other_interaction_op = super(type(gate),
                                 gate).interaction_operator_generator()
    other_interaction_op = openfermion.normal_ordered(interaction_op)
    assert interaction_op == other_interaction_op


random_quadratic_gates = [random_fermionic_simulation_gate(2) for _ in range(5)]
manual_quadratic_gates = [
    openfermion.QuadraticFermionicSimulationGate(weights)
    for weights in [cast(Tuple[float, float], (1, 1)), (1, 0), (0, 1), (0, 0)]
]
quadratic_gates = random_quadratic_gates + manual_quadratic_gates
cubic_gates = ([openfermion.CubicFermionicSimulationGate()] +
               [random_fermionic_simulation_gate(3) for _ in range(5)])
quartic_gates = ([openfermion.QuarticFermionicSimulationGate()] +
                 [random_fermionic_simulation_gate(4) for _ in range(5)])
gates = quadratic_gates + cubic_gates + quartic_gates


@pytest.mark.parametrize('gate', gates)
def test_fermionic_simulation_gate(gate):
    openfermion.testing.assert_implements_consistent_protocols(gate)

    generator = gate.qubit_generator_matrix
    expected_unitary = la.expm(-1j * gate.exponent * generator)
    actual_unitary = cirq.unitary(gate)
    assert np.allclose(expected_unitary, actual_unitary)

    assert_fswap_consistent(gate)
    assert_permute_consistent(gate)
    assert_generators_consistent(gate)
    assert_resolution_consistent(gate)
    assert_interaction_operator_consistent(gate)

    assert gate.num_weights() == super(type(gate), gate).num_weights()


@pytest.mark.parametrize('weights', list(np.random.rand(10, 3)) + [(1, 0, 1)])
def test_weights_and_exponent(weights):
    exponents = np.linspace(-1, 1, 8)
    gates = tuple(
        openfermion.QuarticFermionicSimulationGate(
            weights / exponent, exponent=exponent, absorb_exponent=True)
        for exponent in exponents)

    for g1, g2 in itertools.combinations(gates, 2):
        assert cirq.approx_eq(g1, g2, atol=1e-100)

    for i, (gate, exponent) in enumerate(zip(gates, exponents)):
        assert gate.exponent == 1
        new_exponent = exponents[-i]
        new_gate = gate._with_exponent(new_exponent)
        assert new_gate.exponent == new_exponent


def test_zero_weights():
    for gate_type in [
            openfermion.QuadraticFermionicSimulationGate,
            openfermion.CubicFermionicSimulationGate,
            openfermion.QuarticFermionicSimulationGate
    ]:
        weights = (0,) * gate_type.num_weights()
        gate = gate_type(weights)
        n_qubits = gate.num_qubits()

        assert np.allclose(cirq.unitary(gate), np.eye(2**n_qubits))
        cirq.testing.assert_decompose_is_consistent_with_unitary(gate)

        operator = openfermion.InteractionOperator.zero(n_qubits)
        assert gate_type.from_interaction_operator(operator=operator) is None


@pytest.mark.parametrize(
    'weights,exponent',
    [((np.random.uniform(-5, 5) + 1j * np.random.uniform(-5, 5),
       np.random.uniform(-5, 5)), np.random.uniform(-5, 5)) for _ in range(5)])
def test_quadratic_fermionic_simulation_gate_unitary(weights, exponent):
    generator = np.zeros((4, 4), dtype=np.complex128)
    # w0 |10><01| + h.c.
    generator[2, 1] = weights[0]
    generator[1, 2] = weights[0].conjugate()
    # w1 |11><11|
    generator[3, 3] = weights[1]
    expected_unitary = la.expm(-1j * exponent * generator)

    gate = openfermion.QuadraticFermionicSimulationGate(weights,
                                                        exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)

    symbolic_gate = (openfermion.QuadraticFermionicSimulationGate(
        (sympy.Symbol('w0'), sympy.Symbol('w1')), exponent=sympy.Symbol('t')))
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(symbolic_gate._decompose_(qubits))
    resolver = {'w0': weights[0], 'w1': weights[1], 't': exponent}
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    decomp_unitary = resolved_circuit.unitary(qubit_order=qubits)

    assert np.allclose(expected_unitary, decomp_unitary)


@pytest.mark.parametrize('gate', random_quadratic_gates)
def test_quadratic_fermionic_simulation_gate_symbolic_decompose(gate):
    assert_symbolic_decomposition_consistent(gate)


def test_cubic_fermionic_simulation_gate_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        openfermion.CubicFermionicSimulationGate()**0.5,
        openfermion.CubicFermionicSimulationGate((1,) * 3, exponent=0.5),
        openfermion.CubicFermionicSimulationGate((0.5,) * 3),
    )
    eq.add_equality_group(openfermion.CubicFermionicSimulationGate((1j, 0, 0)),)
    eq.add_equality_group(
        openfermion.CubicFermionicSimulationGate((sympy.Symbol('s'), 0, 0),
                                                 exponent=2),
        openfermion.CubicFermionicSimulationGate((2 * sympy.Symbol('s'), 0, 0),
                                                 exponent=1),
    )
    eq.add_equality_group(
        openfermion.CubicFermionicSimulationGate((0, 0.7, 0), global_shift=2),
        openfermion.CubicFermionicSimulationGate((0, 0.35, 0),
                                                 global_shift=1,
                                                 exponent=2),
    )
    eq.add_equality_group(
        openfermion.CubicFermionicSimulationGate((1, 1, 1)),
        openfermion.CubicFermionicSimulationGate(((1 + 2 * np.pi), 1, 1)),
    )


@pytest.mark.parametrize('exponent,control',
                         itertools.product([0, 1, -1, 0.25, -0.5, 0.1],
                                           [0, 1, 2]))
def test_cubic_fermionic_simulation_gate_consistency_special(exponent, control):
    weights = tuple(np.eye(1, 3, control)[0] * 0.5 * np.pi)
    general_gate = openfermion.CubicFermionicSimulationGate(weights,
                                                            exponent=exponent)
    general_unitary = cirq.unitary(general_gate)

    indices = np.dot(list(itertools.product((0, 1), repeat=3)),
                     (2**np.roll(np.arange(3), -control))[::-1])
    special_gate = cirq.ControlledGate(cirq.ISWAP**-exponent)
    special_unitary = (
        cirq.unitary(special_gate)[indices[:, np.newaxis], indices])

    assert np.allclose(general_unitary, special_unitary)


@pytest.mark.parametrize(
    'weights,exponent',
    [(np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
      np.random.uniform(-5, 5)) for _ in range(5)])
def test_cubic_fermionic_simulation_gate_consistency_docstring(
        weights, exponent):
    generator = np.zeros((8, 8), dtype=np.complex128)
    # w0 |110><101| + h.c.
    generator[6, 5] = weights[0]
    generator[5, 6] = weights[0].conjugate()
    # w1 |110><011| + h.c.
    generator[6, 3] = weights[1]
    generator[3, 6] = weights[1].conjugate()
    # w2 |101><011| + h.c.
    generator[5, 3] = weights[2]
    generator[3, 5] = weights[2].conjugate()
    expected_unitary = la.expm(-1j * exponent * generator)

    gate = openfermion.CubicFermionicSimulationGate(weights, exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)


def test_quartic_fermionic_simulation_consistency():
    openfermion.testing.assert_implements_consistent_protocols(
        openfermion.QuarticFermionicSimulationGate())


quartic_fermionic_simulation_simulator_test_cases = [
    (openfermion.QuarticFermionicSimulationGate(
        (0, 0, 0)), 1., np.ones(16) / 4., np.ones(16) / 4., 5e-6),
    (openfermion.QuarticFermionicSimulationGate((0.2, -0.1, 0.7)), 0.,
     np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
     np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
     5e-6),
    (openfermion.QuarticFermionicSimulationGate((0.2, -0.1, 0.7)), 0.3,
     np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
     np.array([
         1, -1, -1, -np.exp(0.21j), -1, -np.exp(-0.03j),
         np.exp(-0.06j), 1, 1,
         np.exp(-0.06j),
         np.exp(-0.03j), 1,
         np.exp(0.21j), 1, 1, 1
     ]) / 4., 5e-6),
    (openfermion.QuarticFermionicSimulationGate((1. / 3, 0, 0)), 1.,
     np.array([0, 0, 0, 0, 0, 0, 1., 0, 0, 1., 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
     np.array([0, 0, 0, 0, 0, 0, 1., 0, 0, 1., 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
     5e-6),
    (openfermion.QuarticFermionicSimulationGate((0, np.pi / 3, 0)), 1.,
     np.array([1., 1., 0, 0, 0, 1., 0, 0, 0, 0., -1., 0, 0, 0, 0, 0]) / 2.,
     np.array([
         1., 1., 0, 0, 0, -np.exp(4j * np.pi / 3), 0, 0, 0, 0.,
         -np.exp(1j * np.pi / 3), 0, 0, 0, 0, 0
     ]) / 2., 5e-6),
    (openfermion.QuarticFermionicSimulationGate((0, 0, -np.pi / 2)), 1.,
     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0,
               0]), np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0]), 5e-6),
    (openfermion.QuarticFermionicSimulationGate((0, 0, -0.25 * np.pi)), 1.,
     np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1j, 0, 0, 0]) / np.sqrt(2),
     5e-6),
    (openfermion.QuarticFermionicSimulationGate(
        (-np.pi / 4, np.pi / 6, -np.pi / 2)), 1.,
     np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]) / np.sqrt(3),
     np.array([
         0, 0, 0, 1j, 0, -1j / 2., 1 / np.sqrt(2), 0, 0, 1j / np.sqrt(2),
         np.sqrt(3) / 2, 0, 0, 0, 0, 0
     ]) / np.sqrt(3), 5e-6),
]


@pytest.mark.parametrize('gate, exponent, initial_state, correct_state, atol',
                         quartic_fermionic_simulation_simulator_test_cases)
def test_quartic_fermionic_simulation_on_simulator(gate, exponent,
                                                   initial_state, correct_state,
                                                   atol):

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(gate(a, b, c, d)**exponent)
    result = circuit.final_state_vector(initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(result,
                                                    correct_state,
                                                    atol=atol)


def test_quartic_fermionic_simulation_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(
        openfermion.QuarticFermionicSimulationGate((1.2, 0.4, -0.4),
                                                   exponent=0.5),
        openfermion.QuarticFermionicSimulationGate((0.3, 0.1, -0.1),
                                                   exponent=2),
        openfermion.QuarticFermionicSimulationGate((-0.6, -0.2, 0.2),
                                                   exponent=-1),
        openfermion.QuarticFermionicSimulationGate((0.6, 0.2, 2 * np.pi - 0.2)),
    )

    eq.add_equality_group(
        openfermion.QuarticFermionicSimulationGate((-0.6, 0.0, 0.3),
                                                   exponent=0.5))

    eq.make_equality_group(lambda: openfermion.QuarticFermionicSimulationGate(
        (0.1, -0.3, 0.0), exponent=0.0))
    eq.make_equality_group(lambda: openfermion.QuarticFermionicSimulationGate(
        (1., -1., 0.5), exponent=0.75))


def test_quadratic_fermionic_simulation_gate_text_diagram():
    gate = openfermion.QuadraticFermionicSimulationGate((1, 1))
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit([gate(a, b), gate(b, c)])

    assert super(type(gate), gate).wire_symbol(False) == type(gate).__name__
    assert (super(type(gate), gate)._diagram_exponent(
        cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT) == gate._exponent)

    expected_text_diagram = """
0: ───↓↑(1, 1)──────────────
      │
1: ───↓↑─────────↓↑(1, 1)───
                 │
2: ──────────────↓↑─────────
""".strip()
    cirq.testing.assert_has_diagram(circuit, expected_text_diagram)

    expected_text_diagram = """
0: ---a*a(1, 1)---------------
      |
1: ---a*a---------a*a(1, 1)---
                  |
2: ---------------a*a---------
""".strip()
    cirq.testing.assert_has_diagram(circuit,
                                    expected_text_diagram,
                                    use_unicode_characters=False)


def test_cubic_fermionic_simulation_gate_text_diagram():
    gate = openfermion.CubicFermionicSimulationGate((1, 1, 1))
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit([gate(*qubits[:3]), gate(*qubits[2:5])])

    assert super(type(gate), gate).wire_symbol(False) == type(gate).__name__
    assert (super(type(gate), gate)._diagram_exponent(
        cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT) == gate._exponent)

    expected_text_diagram = """
0: ───↕↓↑(1, 1, 1)──────────────────
      │
1: ───↕↓↑───────────────────────────
      │
2: ───↕↓↑────────────↕↓↑(1, 1, 1)───
                     │
3: ──────────────────↕↓↑────────────
                     │
4: ──────────────────↕↓↑────────────
""".strip()
    cirq.testing.assert_has_diagram(circuit, expected_text_diagram)

    expected_text_diagram = """
0: ---na*a(1, 1, 1)-------------------
      |
1: ---na*a----------------------------
      |
2: ---na*a------------na*a(1, 1, 1)---
                      |
3: -------------------na*a------------
                      |
4: -------------------na*a------------
""".strip()
    cirq.testing.assert_has_diagram(circuit,
                                    expected_text_diagram,
                                    use_unicode_characters=False)


test_weights = [1.0, 0.5, 0.25, 0.1, 0.0, -0.5]


@pytest.mark.parametrize('weights',
                         itertools.chain(
                             itertools.product(test_weights, repeat=3),
                             np.random.rand(10, 3)))
def test_quartic_fermionic_simulation_decompose(weights):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        openfermion.QuarticFermionicSimulationGate(weights))


@pytest.mark.parametrize(
    'weights,exponent',
    [(np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
      np.random.uniform(-5, 5)) for _ in range(5)])
def test_quartic_fermionic_simulation_unitary(weights, exponent):
    generator = np.zeros((1 << 4,) * 2, dtype=np.complex128)

    # w0 |1001><0110| + h.c.
    generator[9, 6] = weights[0]
    generator[6, 9] = weights[0].conjugate()
    # w1 |1010><0101| + h.c.
    generator[10, 5] = weights[1]
    generator[5, 10] = weights[1].conjugate()
    # w2 |1100><0011| + h.c.
    generator[12, 3] = weights[2]
    generator[3, 12] = weights[2].conjugate()
    expected_unitary = la.expm(-1j * exponent * generator)

    gate = openfermion.QuarticFermionicSimulationGate(weights,
                                                      exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)


def test_quartic_fermionic_simulation_gate_text_diagram():
    gate = openfermion.QuarticFermionicSimulationGate((1, 1, 1))
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([gate(*qubits[:4]), gate(*qubits[-4:])])

    assert super(type(gate), gate).wire_symbol(False) == type(gate).__name__
    for G in (gate, gate._with_exponent('e')):
        assert (super(type(G), G)._diagram_exponent(
            cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT) == G._exponent)

    expected_text_diagram = """
0: ───⇊⇈(1, 1, 1)─────────────────
      │
1: ───⇊⇈──────────────────────────
      │
2: ───⇊⇈────────────⇊⇈(1, 1, 1)───
      │             │
3: ───⇊⇈────────────⇊⇈────────────
                    │
4: ─────────────────⇊⇈────────────
                    │
5: ─────────────────⇊⇈────────────
""".strip()
    cirq.testing.assert_has_diagram(circuit, expected_text_diagram)

    expected_text_diagram = """
0: ---a*a*aa(1, 1, 1)---------------------
      |
1: ---a*a*aa------------------------------
      |
2: ---a*a*aa------------a*a*aa(1, 1, 1)---
      |                 |
3: ---a*a*aa------------a*a*aa------------
                        |
4: ---------------------a*a*aa------------
                        |
5: ---------------------a*a*aa------------
""".strip()
    cirq.testing.assert_has_diagram(circuit,
                                    expected_text_diagram,
                                    use_unicode_characters=False)


@pytest.mark.parametrize(
    'weights,exponent',
    [(np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
      np.random.uniform(-5, 5)) for _ in range(5)])
def test_quartic_fermionic_simulation_apply_unitary(weights, exponent):
    gate = openfermion.QuarticFermionicSimulationGate(weights,
                                                      exponent=exponent)
    cirq.testing.assert_has_consistent_apply_unitary(gate, atol=5e-6)
