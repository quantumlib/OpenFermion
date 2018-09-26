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

"""Tests for _qpe_estimators.py"""
import numpy
import unittest
from numpy import pi

from ._qpe_estimators import (
    ProbabilityDist,
    BayesEstimator,
    BayesDepolarizingEstimator,
    TimeSeriesEstimator,
    TimeSeriesMultiRoundEstimator)


class ProbabilityDistTest(unittest.TestCase):

    def setup(self):
        pass

    def test_basic_initialization(self):
        pd = ProbabilityDist(num_vectors=1,
                             amplitude_guess=[1],
                             amplitude_vars=[[1]],
                             num_freqs=10,
                             max_n=1)
        self.assertEqual(pd._num_freqs, 10)
        self.assertEqual(pd._num_vectors, 1)
        self.assertEqual(len(pd._matrices), 1)
        self.assertEqual(len(pd._matrices[0]), 2)
        self.assertEqual(len(pd._amplitude_estimates), 1)
        self.assertEqual(pd._fourier_vectors.shape, (21, 1))

    def test_init_dist(self):

        vector_guess = numpy.zeros([21, 1])
        vector_guess[0, 0] = 1
        vector_guess[2, 0] = 1  # cos wave
        pd = ProbabilityDist(num_vectors=1,
                             amplitude_guess=[1],
                             amplitude_vars=[[1]],
                             num_freqs=10,
                             max_n=1,
                             vector_guess=vector_guess)
        x_vec = numpy.linspace(-pi, pi, 11)
        dist = pd.get_real_dist(x_vec)
        dist_comp = (1 + numpy.cos(x_vec)) / (2*pi)
        self.assertAlmostEqual(numpy.sum(numpy.abs(dist-dist_comp)), 0)

    def test_Holevo(self):
        vector_guess = numpy.zeros([21, 1])
        vector_guess[0, 0] = 1
        vector_guess[2, 0] = 1  # sine wave
        pd = ProbabilityDist(num_vectors=1,
                             amplitude_guess=[1],
                             amplitude_vars=[[1]],
                             num_freqs=10,
                             max_n=1,
                             vector_guess=vector_guess)
        self.assertEqual(pd._Holevo_centers(), 0)
        self.assertEqual(pd._Holevo_variances()[0], 3)

    def test_vector_product(self):
        pd = ProbabilityDist(num_vectors=1,
                             amplitude_guess=[1],
                             amplitude_vars=[[1]],
                             num_freqs=10,
                             max_n=1)

        round_data = {
            'final_rotation': 0,
            'measurement': 0,
            'num_rotations': 1
        }

        res = pd._vector_product(pd._fourier_vectors, round_data)
        res_comp = numpy.zeros([21, 1])
        res_comp[0] = 0.5
        res_comp[2] = 0.5
        self.assertAlmostEqual(numpy.sum(numpy.abs(res-res_comp)), 0)

    def test_diffs(self):
        pd = ProbabilityDist(num_vectors=2,
                             amplitude_guess=[0.9, 0.1],
                             amplitude_vars=[[0.3, -0.29], [-0.29, 0.3]],
                             num_freqs=10,
                             max_n=1)

        test_vec = numpy.array([0.9, 0.1])
        test_vec2 = numpy.array([1, 0])
        pd.p_vecs = [test_vec]

        ml = pd._mlikelihood(test_vec2)
        sd = pd._single_diff(test_vec2, test_vec)
        jt = pd._jacobian_term(test_vec2, test_vec)

        self.assertAlmostEqual(ml, -numpy.log(0.9))
        self.assertAlmostEqual(
            numpy.sum(numpy.abs(sd + 1/0.9*test_vec2)), 0)
        self.assertAlmostEqual(
            numpy.sum(numpy.abs(
                numpy.dot(test_vec2[:, numpy.newaxis],
                          test_vec2[numpy.newaxis, :]) / 0.81 - jt)), 0)


class BayesEstimatorTest(unittest.TestCase):

    def setup(self):
        pass

    def test_init(self):
        be = BayesEstimator(num_vectors=1,
                            amplitude_guess=[1],
                            amplitude_vars=[[1]],
                            num_freqs=10,
                            max_n=1,
                            store_history=True)
        self.assertEqual(be.averages, [])
        self.assertEqual(be.variances, [])
        self.assertEqual(be.log_Bayes_factor_history, [])
        self.assertEqual(be.amplitudes_history, [])
        self.assertEqual(be.log_Bayes_factor, 0)

    def test_update(self):
        be = BayesEstimator(num_vectors=1,
                            amplitude_guess=[1],
                            amplitude_vars=[[1]],
                            num_freqs=10,
                            max_n=1,
                            store_history=True)
        test_experiment = [{
            'final_rotation': -numpy.pi/4,
            'measurement': 0,
            'num_rotations': 1
        }]
        self.assertEqual(be.update(test_experiment), True)

        self.assertEqual(be.averages, [numpy.pi/4])
        self.assertEqual(be.log_Bayes_factor, numpy.log(0.5))

    def test_estimation(self):
        be = BayesEstimator(num_vectors=1,
                            amplitude_guess=[1],
                            amplitude_vars=numpy.array([[1]]),
                            num_freqs=1000,
                            max_n=1)

        test_experiment1 = [{
            'final_rotation': numpy.pi/2,
            'measurement': 0,
            'num_rotations': 1
        }]
        test_experiment2 = [{
            'final_rotation': 0,
            'measurement': 0,
            'num_rotations': 1
        }]
        test_experiment3 = [{
            'final_rotation': 0,
            'measurement': 1,
            'num_rotations': 1
        }]
        num2 = 0
        for j in range(100):
            be.update(test_experiment1)
            if j % 2 == 0:
                be.update(test_experiment2)
                num2 += 1
            else:
                be.update(test_experiment3)

        self.assertEqual(be.estimate(), -numpy.pi/2)

    def test_amplitude_approx_update(self):
        be = BayesEstimator(num_vectors=2,
                            amplitude_guess=[0.9, 0.1],
                            amplitude_vars=[[0.1, -0.09], [-0.09, 0.1]],
                            num_freqs=10,
                            max_n=1,
                            amplitude_approx_cutoff=15)
        be2 = BayesEstimator(num_vectors=2,
                             amplitude_guess=[0.9, 0.1],
                             amplitude_vars=[[0.1, -0.09], [-0.09, 0.1]],
                             num_freqs=10,
                             max_n=1,
                             amplitude_approx_cutoff=10)
        test_experiment = [{
            'final_rotation': -numpy.pi/4,
            'measurement': 0,
            'num_rotations': 1
        }]
        test_experiment2 = [{
            'final_rotation': numpy.pi/4,
            'measurement': 0,
            'num_rotations': 2
        }]
        test_experiment3 = [{
            'final_rotation': 0,
            'measurement': 0,
            'num_rotations': 2
        }]

        for j in range(5):
            be.update(test_experiment)
            be2.update(test_experiment)
            be.update(test_experiment2)
            be2.update(test_experiment2)
            be.update(test_experiment3)
            be2.update(test_experiment3)

        self.assertTrue(
            numpy.sum(numpy.abs(
                be._amplitude_estimates-be2._amplitude_estimates)) < 1e-3)


class BayesDepolarizingEstimatorTest(unittest.TestCase):
    def setup(self):
        pass

    def test_epsilons(self):
        estimator = BayesDepolarizingEstimator(
            num_vectors=1,
            amplitude_guess=[1],
            amplitude_vars=numpy.array([[1]]),
            num_freqs=1000,
            max_n=1,
            K1=1,
            Kerr=1)
        self.assertAlmostEqual(estimator._epsilon_D_function(2),
                               1-numpy.exp(-2))
        self.assertAlmostEqual(estimator._epsilon_B_function(),
                               1-numpy.exp(-1))

    def test_vector_product(self):
        estimator = BayesDepolarizingEstimator(
            num_vectors=1,
            amplitude_guess=[1],
            amplitude_vars=numpy.array([[1]]),
            num_freqs=1000,
            max_n=1,
            K1=1,
            Kerr=1)

        vectors = numpy.zeros(2001)
        vectors[0] = 1
        round_data = {
            'measurement': 0,
            'num_rotations': 2,
            'final_rotation': 0
        }
        updated_vectors = estimator._vector_product(vectors, round_data)
        epsilon_D = 1-numpy.exp(-2)
        epsilon_B = 1-numpy.exp(-1)
        self.assertAlmostEqual(updated_vectors[0], 0.5+0.5*epsilon_B)
        self.assertAlmostEqual(updated_vectors[1], 0)
        self.assertAlmostEqual(updated_vectors[2], 0)
        self.assertAlmostEqual(updated_vectors[3], 0)
        self.assertAlmostEqual(updated_vectors[4], 0.5*(1-epsilon_D))


class TimeSeriesEstimatorTest(unittest.TestCase):
    def setup(self):
        pass

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
        estimator._update_flag=True
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
    def setup(self):
        pass

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
