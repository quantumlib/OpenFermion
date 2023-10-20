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
"""Tests for vpe_estimators.py"""

import pytest
import numpy
import pandas
import cirq

from .vpe_estimators import PhaseFitEstimator, get_phase_function

rng = numpy.random.RandomState(seed=42)


def test_requests_simulation_at_pi_for_pauli():
    estimator = PhaseFitEstimator(evals=[-1, +1])
    sim_points = estimator.get_simulation_points(safe=False)
    assert len(sim_points) == 2
    assert numpy.isclose(sim_points[0], 0)
    assert numpy.isclose(sim_points[1], numpy.pi / 2)


def test_estimates_expectation_value_pauli_nonoise():
    evals = numpy.array([-1, +1])
    true_amps = numpy.array([0.2, 0.8])
    true_expectation_value = numpy.dot(evals, true_amps)

    estimator = PhaseFitEstimator(evals)
    sim_points = estimator.get_simulation_points()
    phase_function = numpy.array(
        [
            numpy.sum([amp * numpy.exp(1j * ev * time) for ev, amp in zip(evals, true_amps)])
            for time in sim_points
        ]
    )
    print(phase_function)
    test_expectation_value = estimator.get_expectation_value(phase_function)
    assert numpy.isclose(true_expectation_value, test_expectation_value)


def test_estimates_expectation_value_scattered_nonoise():
    evals = rng.uniform(-1, 1, 10)
    true_amps = rng.uniform(0, 1, 10)
    true_amps = true_amps / numpy.sum(true_amps)
    true_expectation_value = numpy.dot(evals, true_amps)

    estimator = PhaseFitEstimator(evals)
    sim_points = estimator.get_simulation_points()
    phase_function = numpy.array(
        [
            numpy.sum([amp * numpy.exp(1j * ev * time) for ev, amp in zip(evals, true_amps)])
            for time in sim_points
        ]
    )
    test_expectation_value = estimator.get_expectation_value(phase_function)

    assert numpy.isclose(true_expectation_value, test_expectation_value)


def test_phase_function_gen_raises_error():
    results = [0, 0]
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    target_qid = 0
    with pytest.raises(ValueError):
        get_phase_function(results, qubits, target_qid)


def test_phase_function_gen():
    class FakeResult:
        def __init__(self, data):
            self.data = {'msmt': pandas.Series(data)}

    datasets = [[] for j in range(8)]
    for xindex in [2, 3, 6, 7]:
        datasets[xindex] = [0] * 50 + [2] * 50
    for z0index in [0, 4]:
        datasets[z0index] = [0] * 100
    for z1index in [1, 5]:
        datasets[z1index] = [2] * 100

    results = [FakeResult(dataset) for dataset in datasets]
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    target_qid = 0
    phase_function_est = get_phase_function(results, qubits, target_qid)
    assert phase_function_est == 1
