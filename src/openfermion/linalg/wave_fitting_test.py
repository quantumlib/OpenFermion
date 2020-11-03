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
""" Tests for wave_fitting.py"""

import numpy
from .wave_fitting import prony, fit_known_frequencies


def test_prony_zeros():
    signal = numpy.zeros(10)
    amplitudes, phases = prony(signal)
    assert (len(amplitudes) == 5)
    assert (len(phases) == 5)
    for j in range(5):
        numpy.testing.assert_allclose(amplitudes[j], 0)
        numpy.testing.assert_allclose(phases[j], 0)


def test_prony_signal():
    x_vec = numpy.linspace(0, 1, 11)
    y_vec = (0.5 * numpy.exp(1j * x_vec * 3) + 0.3 * numpy.exp(1j * x_vec * 5) +
             0.15 * numpy.exp(1j * x_vec * 1.5) +
             0.1 * numpy.exp(1j * x_vec * 4) +
             0.05 * numpy.exp(1j * x_vec * 1.2))
    print(y_vec)
    amplitudes, phases = prony(y_vec)
    assert (len(amplitudes) == 5)
    assert (len(phases) == 5)
    for a, p in zip(amplitudes, phases):
        print(a, numpy.angle(p))
    numpy.testing.assert_allclose(numpy.abs(amplitudes[0]), 0.5, atol=1e-4)
    numpy.testing.assert_allclose(numpy.abs(amplitudes[1]), 0.3, atol=1e-4)
    numpy.testing.assert_allclose(numpy.abs(amplitudes[2]), 0.15, atol=1e-4)
    numpy.testing.assert_allclose(numpy.abs(amplitudes[3]), 0.1, atol=1e-4)
    numpy.testing.assert_allclose(numpy.abs(amplitudes[4]), 0.05, atol=1e-4)
    numpy.testing.assert_allclose(numpy.angle(phases[0]), 0.3, atol=1e-4)
    numpy.testing.assert_allclose(numpy.angle(phases[1]), 0.5, atol=1e-4)
    numpy.testing.assert_allclose(numpy.angle(phases[2]), 0.15, atol=1e-4)
    numpy.testing.assert_allclose(numpy.angle(phases[3]), 0.4, atol=1e-4)
    numpy.testing.assert_allclose(numpy.angle(phases[4]), 0.12, atol=1e-4)


def test_fitting_signal():
    frequencies = numpy.array([0.4, 0.5, 0.8])
    amplitudes = numpy.array([0.2, 0.4, 0.4])
    times = numpy.linspace(0, 10, 21)
    signal = numpy.array([
        numpy.sum([
            amp * numpy.exp(1j * time * freq)
            for freq, amp in zip(frequencies, amplitudes)
        ])
        for time in times
    ])
    amplitudes_guess = fit_known_frequencies(signal, times, frequencies)
    assert len(amplitudes_guess == 3)
    for index in range(3):
        assert numpy.isclose(amplitudes_guess[index], amplitudes[index])
