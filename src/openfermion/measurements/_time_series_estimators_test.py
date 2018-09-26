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

"""Tests for _time_series_estimators.py"""

import numpy
import unittest
import warnings

from ._time_series_estimators import (
    TimeSeriesEstimator,
    TimeSeriesMultiRoundEstimator)


class TimeSeriesEstimatorTest(unittest.TestCase):

    def test_running(self):
        estimator = TimeSeriesEstimator(
            max_experiment_length=10,
            num_freqs_max=2,
            singular_values_cutoff=0,
            automatic_average=False,
            store_history=True)
        for k in range(1, 11):
            experiment_data = [{
                'num_rotations': k,
                'measurement': 0,
                'final_rotation': 0}]
            estimator.update(experiment_data)
            experiment_data = [{
                'num_rotations': k,
                'measurement': 0,
                'final_rotation': numpy.pi/2}]
            estimator.update(experiment_data)
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore')
            estimator.estimate(return_amplitudes=False)
            angles, amplitudes = estimator.estimate(return_amplitudes=True)
        self.assertEqual(len(angles), 2)
        self.assertEqual(len(amplitudes), 2)

    def test_accuracy(self):
        estimator = TimeSeriesEstimator(
            max_experiment_length=10,
            num_freqs_max=5,
            singular_values_cutoff=1e-3,
            automatic_average=True,
            store_history=True)
        estimator._function_estimate = numpy.array([1]*21)
        estimator._update_flag = True
        angles, amplitudes = estimator.estimate(return_amplitudes=True)
        self.assertEqual(len(angles), 1)
        self.assertEqual(len(amplitudes), 1)
        self.assertAlmostEqual(angles[0], 0)
        self.assertAlmostEqual(amplitudes[0], 1)

    def test_depolarizing_noise_mass_update(self):
        estimator = TimeSeriesEstimator(
            max_experiment_length=10,
            num_freqs_max=5,
            singular_values_cutoff=1e-3,
            depolarizing_noise=True,
            automatic_average=True,
            store_history=True)
        f_data = numpy.array([[[1, 0], [1, 1]] for j in range(11)])
        estimator.update_mass(f_data)
        self.assertEqual(len(estimator._function_estimate), 11)

        angles, amplitudes = estimator.estimate(return_amplitudes=True)
        self.assertEqual(len(angles), 1)
        self.assertEqual(len(amplitudes), 1)
        self.assertAlmostEqual(angles[0], 0)
        self.assertAlmostEqual(amplitudes[0], 1)

    def test_errors(self):
        with self.assertRaises(ValueError):
            estimator = TimeSeriesEstimator(
                max_experiment_length=2,
                num_freqs_max=3)
        estimator = TimeSeriesEstimator(
                max_experiment_length=10,
                num_freqs_max=2)
        with self.assertRaises(NotImplementedError):
            estimator.update([0, 0])


class TimeSeriesMultiRoundEstimatorTest(unittest.TestCase):

    def test_running(self):
        estimator = TimeSeriesMultiRoundEstimator(
            max_experiment_length=20,
            num_freqs_max=2,
            singular_values_cutoff=0,
            automatic_average=True,
            store_history=True)
        experiment_data = [{
            'num_rotations': 1,
            'measurement': 0,
            'final_rotation': 0} for _ in range(10)] + [{
                'num_rotations': 1,
                'measurement': 0,
                'final_rotation': numpy.pi/2} for _ in range(5)] + [{
                    'num_rotations': 1,
                    'measurement': 1,
                    'final_rotation': numpy.pi/2} for _ in range(5)]

        estimator.update(experiment_data)
        estimator.estimate(return_amplitudes=False)
        angles, amplitudes = estimator.estimate(return_amplitudes=True)

        self.assertEqual(len(angles), 2)
        self.assertEqual(len(amplitudes), 2)

    def test_update_mass(self):
        estimator = TimeSeriesMultiRoundEstimator(
            max_experiment_length=4,
            num_freqs_max=1,
            singular_values_cutoff=0,
            automatic_average=True,
            store_history=True)

        hamming_mat = numpy.zeros([3, 3])
        hamming_mat[0, 0] = 2500
        hamming_mat[0, 1] = 5000
        hamming_mat[0, 2] = 2500

        estimator.update_mass(hamming_mat)

    def test_errors(self):
        estimator = TimeSeriesMultiRoundEstimator(
            max_experiment_length=2,
            num_freqs_max=1)
        with self.assertRaises(NotImplementedError):
            estimator.update([0])
        test_experiments = [{'final_rotation': 1}, {'final_rotation': 1}]
        with self.assertRaises(NotImplementedError):
            estimator.update(test_experiments)
