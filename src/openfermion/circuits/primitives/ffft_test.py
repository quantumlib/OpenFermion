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

from typing import List, TYPE_CHECKING

import numpy as np
import pytest
import cirq
from cirq import LineQubit

from openfermion import bogoliubov_transform, ffft
from openfermion.circuits.primitives.ffft import _F0Gate, _TwiddleGate

if TYPE_CHECKING:
    from typing import Dict


def _fourier_transform_single_fermionic_modes(amplitudes: List[complex]
                                             ) -> List[complex]:
    """Fermionic Fourier transform of a list of single Fermionic modes.

    Args:
        amplitudes: List of amplitudes for each Fermionic mode.

    Return:
        List representing a new, Fourier transformed amplitudes of the input
        amplitudes.
    """

    def fft(k, n):
        unit = np.exp(-2j * np.pi * k / n)
        return sum(unit**j * amplitudes[j] for j in range(n)) / np.sqrt(n)

    n = len(amplitudes)
    return [fft(k, n) for k in range(n)]


def _fourier_transform_multi_fermionic_mode(n: int, amplitude: complex,
                                            modes: List[int]) -> np.ndarray:
    """Fermionic Fourier transform of a multi Fermionic mode base state.

    Args:
        n: State length, number of qubits used.
        amplitude: State amplitude. Absolute value must be equal to 1.
        modes: List of mode numbers which should appear in the resulting state.
            List order defines the sequence of applied creation operators which
            does not need to be normally ordered.

    Return:
        List representing a new, Fourier transformed amplitudes of the input
        modes.
    """

    def fourier_transform_mode(k):
        unit = np.exp(-2j * np.pi * k / n)
        return [unit**j / np.sqrt(n) for j in range(n)]

    def append_in_normal_order(index, mode):
        phase = 1
        mode = n - 1 - mode
        for i in range(n):
            bit = 1 << i
            if i == mode:
                if index & bit != 0:
                    return None, None
                return index | bit, phase
            elif index & bit:
                phase *= -1

    state = {0: amplitude}
    for m in modes:
        transform = fourier_transform_mode(m)
        new_state = {}  # type: Dict[int, complex]
        for index in state:
            for mode in range(len(transform)):
                new_index, new_phase = append_in_normal_order(index, mode)
                if new_index:
                    if not new_index in new_state:
                        new_state[new_index] = 0
                    new_amplitude = state[index] * transform[mode] * new_phase
                    new_state[new_index] += new_amplitude
        state = new_state

    result = np.zeros(1 << n, dtype=complex)
    for i in range(len(result)):
        if i in state:
            result[i] = state[i]

    return result / np.linalg.norm(result)


def _single_fermionic_modes_state(amplitudes: List[complex]) -> np.ndarray:
    """Prepares state which is a superposition of single Fermionic modes.

    Args:
        amplitudes: List of amplitudes to be assigned for a state representing
            a Fermionic mode.

    Return:
        State vector which is a superposition of single fermionic modes under
        JWT representation, each with appropriate amplitude assigned.
    """
    n = len(amplitudes)
    state = np.zeros(1 << n, dtype=complex)
    for m in range(len(amplitudes)):
        state[1 << (n - 1 - m)] = amplitudes[m]
    return state / np.linalg.norm(state)


def _multi_fermionic_mode_base_state(n: int, amplitude: complex,
                                     modes: List[int]) -> np.ndarray:
    """Prepares state which is a base vector with list of desired modes.

    Prepares a state to be one of the basis vectors of an n-dimensional qubits
    Hilbert space. The basis state has qubits from the list modes set to 1, and
    all other qubits set to 0.

    Args:
        n: State length, number of qubits used.
        amplitude: State amplitude. Absolute value must be equal to 1.
        modes: List of mode numbers which should appear in the resulting state.
            This method assumes the list is sorted resulting in normally ordered
            operator.

    Return:
        State vector that represents n-dimensional Hilbert space base state,
        with listed modes prepared in state 1. State vector is big-endian
        encoded to match Cirq conventions.
    """
    state = np.zeros(1 << n, dtype=complex)
    state[sum(1 << (n - 1 - m) for m in modes)] = amplitude
    return state


@pytest.mark.parametrize('amplitudes',
                         [[1, 0], [1j, 0], [0, 1], [0, -1j], [1, 1]])
def test_F0Gate_transform(amplitudes):
    qubits = LineQubit.range(2)
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = _single_fermionic_modes_state(amplitudes)
    expected_state = _single_fermionic_modes_state(
        _fourier_transform_single_fermionic_modes(amplitudes))

    circuit = cirq.Circuit(_F0Gate().on(*qubits))
    state = sim.simulate(circuit,
                         initial_state=initial_state).final_state_vector

    assert np.allclose(state, expected_state, rtol=0.0)


def test_F0Gate_text_unicode_diagram():
    qubits = LineQubit.range(2)
    circuit = cirq.Circuit(_F0Gate().on(*qubits))

    assert circuit.to_text_diagram().strip() == """
0: ───F₀───
      │
1: ───F₀───
    """.strip()


def test_F0Gate_text_diagram():
    qubits = LineQubit.range(2)
    circuit = cirq.Circuit(_F0Gate().on(*qubits))

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
0: ---F0---
      |
1: ---F0---
    """.strip()


@pytest.mark.parametrize('k, n, qubit, initial, expected', [
    (0, 2, 0, [1, 0], [1, 0]),
    (2, 8, 0, [1, 0], [np.exp(-2 * np.pi * 1j * 2 / 8), 0]),
    (4, 8, 1, [0, 1], [0, np.exp(-2 * np.pi * 1j * 4 / 8)]),
    (3, 5, 0, [1, 1], [np.exp(-2 * np.pi * 1j * 3 / 5), 1]),
])
def test_TwiddleGate_transform(k, n, qubit, initial, expected):
    qubits = LineQubit.range(2)
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = _single_fermionic_modes_state(initial)
    expected_state = _single_fermionic_modes_state(expected)

    circuit = cirq.Circuit(_TwiddleGate(k, n).on(qubits[qubit]))
    state = sim.simulate(circuit,
                         initial_state=initial_state,
                         qubit_order=qubits).final_state_vector

    assert np.allclose(state, expected_state, rtol=0.0)


def test_TwiddleGate_text_unicode_diagram():
    qubit = LineQubit.range(1)
    circuit = cirq.Circuit(_TwiddleGate(2, 8).on(*qubit))

    assert circuit.to_text_diagram().strip() == """
0: ───ω^2_8───
    """.strip()


def test_TwiddleGate_text_diagram():
    qubit = LineQubit.range(1)
    circuit = cirq.Circuit(_TwiddleGate(2, 8).on(*qubit))

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
0: ---w^2_8---
    """.strip()


@pytest.mark.parametrize('amplitudes',
                         [[1], [1, 0], [0, 1], [1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, -1j / np.sqrt(2), 0, 0, 1 / np.sqrt(2), 0, 0]])
def test_ffft_single_fermionic_modes(amplitudes):
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = _single_fermionic_modes_state(amplitudes)
    expected_state = _single_fermionic_modes_state(
        _fourier_transform_single_fermionic_modes(amplitudes))
    qubits = LineQubit.range(len(amplitudes))

    circuit = cirq.Circuit(ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)
    state = sim.simulate(circuit,
                         initial_state=initial_state,
                         qubit_order=qubits).final_state_vector

    assert np.allclose(state, expected_state, rtol=0.0)


@pytest.mark.parametrize('amplitudes', [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])
def test_ffft_single_fermionic_modes_non_power_of_2(amplitudes):
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = _single_fermionic_modes_state(amplitudes)
    expected_state = _single_fermionic_modes_state(
        _fourier_transform_single_fermionic_modes(amplitudes))
    qubits = LineQubit.range(len(amplitudes))

    circuit = cirq.Circuit(ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)
    state = sim.simulate(circuit,
                         initial_state=initial_state,
                         qubit_order=qubits).final_state_vector

    cirq.testing.assert_allclose_up_to_global_phase(state,
                                                    expected_state,
                                                    atol=1e-8)


@pytest.mark.parametrize('n, initial', [
    (2, (1, [0, 1])),
    (4, (1, [0, 1])),
    (4, (1, [1, 2])),
    (4, (1, [2, 3])),
    (8, (1, [0, 7])),
    (8, (1, [3, 5])),
    (8, (1, [0, 3, 5, 7])),
    (8, (1, [0, 1, 2, 3])),
    (8, (1, [0, 1, 6, 7])),
    (8, (1j, [0, 3, 5, 7])),
    (8, (-1j, [0, 1, 2, 3])),
    (8, (np.sqrt(0.5) + np.sqrt(0.5) * 1j, [0, 1, 6, 7])),
    (8, (1, [0, 1, 2, 3, 5, 6, 7])),
    (8, (1, [0, 1, 2, 3, 4, 5, 6, 7])),
])
def test_ffft_multi_fermionic_mode(n, initial):
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = _multi_fermionic_mode_base_state(n, *initial)
    expected_state = _fourier_transform_multi_fermionic_mode(n, *initial)
    qubits = LineQubit.range(n)

    circuit = cirq.Circuit(ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)
    state = sim.simulate(circuit,
                         initial_state=initial_state,
                         qubit_order=qubits).final_state_vector

    assert np.allclose(state, expected_state, rtol=0.0)


@pytest.mark.parametrize('n, initial', [
    (3, (1, [0, 1])),
    (3, (1, [0, 1])),
    (5, (1, [0, 3, 4])),
    (6, (1, [0, 1, 2, 3])),
    (7, (1, [0, 1, 5, 6])),
    (9, (1, [2, 4, 6])),
])
def test_ffft_multi_fermionic_mode_non_power_of_2(n, initial):
    initial_state = _multi_fermionic_mode_base_state(n, *initial)
    expected_state = _fourier_transform_multi_fermionic_mode(n, *initial)
    qubits = LineQubit.range(n)
    sim = cirq.Simulator(dtype=np.complex128)

    circuit = cirq.Circuit(ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)
    state = sim.simulate(circuit,
                         initial_state=initial_state,
                         qubit_order=qubits).final_state_vector

    cirq.testing.assert_allclose_up_to_global_phase(state,
                                                    expected_state,
                                                    atol=1e-8)


def test_ffft_text_diagram():
    qubits = LineQubit.range(8)

    circuit = cirq.Circuit(ffft(qubits), strategy=cirq.InsertStrategy.EARLIEST)

    assert circuit.to_text_diagram(transpose=True) == """
0   1     2   3     4   5     6   7
│   │     │   │     │   │     │   │
0↦0─1↦4───2↦1─3↦5───4↦2─5↦6───6↦3─7↦7
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦1─3↦3   0↦0─1↦2───2↦1─3↦3
│   │     │   │     │   │     │   │
F₀──F₀    F₀──F₀    F₀──F₀    F₀──F₀
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦1─3↦3   0↦0─1↦2───2↦1─3↦3
│   │     │   │     │   │     │   │
│   ω^0_4 │   ω^1_4 │   ω^0_4 │   ω^1_4
│   │     │   │     │   │     │   │
F₀──F₀    F₀──F₀    F₀──F₀    F₀──F₀
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦1─3↦3   0↦0─1↦2───2↦1─3↦3
│   │     │   │     │   │     │   │
0↦0─1↦2───2↦4─3↦6───4↦1─5↦3───6↦5─7↦7
│   │     │   │     │   │     │   │
│   ω^0_8 │   ω^1_8 │   ω^2_8 │   ω^3_8
│   │     │   │     │   │     │   │
F₀──F₀    F₀──F₀    F₀──F₀    F₀──F₀
│   │     │   │     │   │     │   │
0↦0─1↦4───2↦1─3↦5───4↦2─5↦6───6↦3─7↦7
│   │     │   │     │   │     │   │
    """.strip()


def test_ffft_fails_without_qubits():
    with pytest.raises(ValueError):
        ffft([])


@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 6, 7, 8])
def test_ffft_equal_to_bogoliubov(size):

    def fourier_transform_matrix():
        root_of_unity = np.exp(-2j * np.pi / size)
        return np.array([[root_of_unity**(j * k)
                          for k in range(size)]
                         for j in range(size)]) / np.sqrt(size)

    qubits = LineQubit.range(size)

    ffft_circuit = cirq.Circuit(ffft(qubits),
                                strategy=cirq.InsertStrategy.EARLIEST)
    ffft_matrix = ffft_circuit.unitary(qubits_that_should_be_present=qubits)

    bogoliubov_circuit = cirq.Circuit(bogoliubov_transform(
        qubits, fourier_transform_matrix()),
                                      strategy=cirq.InsertStrategy.EARLIEST)
    bogoliubov_matrix = bogoliubov_circuit.unitary(
        qubits_that_should_be_present=qubits)

    cirq.testing.assert_allclose_up_to_global_phase(ffft_matrix,
                                                    bogoliubov_matrix,
                                                    atol=1e-8)


@pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 6, 7, 8])
def test_ffft_inverse(size):

    qubits = LineQubit.range(size)

    ffft_circuit = cirq.Circuit(ffft(qubits),
                                strategy=cirq.InsertStrategy.EARLIEST)
    ffft_circuit.append(cirq.inverse(ffft(qubits)))
    ffft_matrix = ffft_circuit.unitary(qubits_that_should_be_present=qubits)

    cirq.testing.assert_allclose_up_to_global_phase(ffft_matrix,
                                                    np.identity(1 << size),
                                                    atol=1e-8)
