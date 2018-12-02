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
import warnings

from ._bayesian_estimators import (
    BayesDepolarizingEstimator,
    BayesEstimator)
from ._data_containers import(
    QPEExperimentData,
    QPERoundData)

class IteratingSampler:

    def __init__(
            self, max_experiments_per_round, angles, random_state):
        self.angles = angles
        self.num_angles = len(angles)
        self.max_experiments_per_round = max_experiments_per_round
        self.n_next_round = 1
        self.parity = 0
        self.random_state = random_state

    def sample(self, **kwargs):
        experiment = QPEExperimentData([QPERoundData(
            num_rotations=self.n_next_round,
            final_rotation=self.angles[self.parity],
            measurement=None
        )])
        self.n_next_round = self.n_next_round %\
            self.max_experiments_per_round
        if self.n_next_round == 0:
            self.parity = 1 - self.parity
        self.n_next_round += 1
        return experiment


class BayesEstimatorTest(unittest.TestCase):

    def test_init(self):
        be = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=[[1]],
                            num_freqs=10,
                            max_rotations=1,
                            store_history=True)
        self.assertEqual(be.averages, [])
        self.assertEqual(be.variances, [])
        self.assertEqual(be.log_bayes_factor_history, [])
        self.assertEqual(be.amplitudes_history, [])
        self.assertEqual(be.log_bayes_factor, 0)

    def test_vector_product(self):
        pd = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=[[1]],
                            num_freqs=10,
                            max_rotations=1)

        round_data = QPERoundData(
            final_rotation=0,
            measurement=0,
            num_rotations=1)

        res = pd._vector_product(
            pd._probability_dist._fourier_vectors,
            round_data)
        res_comp = numpy.zeros([21, 1])
        res_comp[0] = 0.5
        res_comp[2] = 0.5
        self.assertAlmostEqual(numpy.sum(numpy.abs(res-res_comp)), 0)

    def test_diffs(self):
        pd = BayesEstimator(num_vectors=2,
                            prior_amplitude_mean=[0.9, 0.1],
                            prior_amplitude_var=[[0.3, -0.29], [-0.29, 0.3]],
                            num_freqs=10,
                            max_rotations=1)

        test_vec = numpy.array([0.9, 0.1])
        test_vec2 = numpy.array([1, 0])
        pd.p_vecs = [test_vec]

        ml = pd._probability_dist._mlikelihood(test_vec2)
        init_l = pd._probability_dist._init_mlikelihood(test_vec2)
        sd = pd._probability_dist._single_diff(test_vec2, test_vec)
        jt = pd._probability_dist._jacobian_term(test_vec2, test_vec)

        self.assertAlmostEqual(ml-init_l, 0)
        self.assertAlmostEqual(
            numpy.sum(numpy.abs(sd + 1/0.9*test_vec2)), 0)
        self.assertAlmostEqual(
            numpy.sum(numpy.abs(
                numpy.dot(test_vec2[:, numpy.newaxis],
                          test_vec2[numpy.newaxis, :]) / 0.81 - jt)), 0)

    def test_warning(self):
        be = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=[[1]],
                            num_freqs=10,
                            max_rotations=1,
                            store_history=True)

        with warnings.catch_warnings(record=True) as w:
            be.update(QPEExperimentData())

    def test_update(self):
        be = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=[[1]],
                            num_freqs=10,
                            max_rotations=1,
                            store_history=True)
        test_experiment = QPEExperimentData([QPERoundData(
            final_rotation=-numpy.pi/4,
            measurement=0,
            num_rotations=1
        )])
        self.assertEqual(be.update(test_experiment), True)
        self.assertEqual(len(be.estimate_amplitudes()), 1)
        evs, amplitudes = be.estimate(return_amplitudes=True)
        self.assertEqual(len(evs), 1)
        self.assertEqual(len(amplitudes), 1)
        self.assertEqual(be.averages, [numpy.pi/4])
        self.assertEqual(be.log_bayes_factor, numpy.log(0.5))

    def test_estimation(self):
        be = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=numpy.array([[1]]),
                            num_freqs=1000,
                            max_rotations=1)

        test_experiment1 = QPEExperimentData([QPERoundData(
            final_rotation=numpy.pi/2,
            measurement=0,
            num_rotations=1
        )])
        test_experiment2 = QPEExperimentData([QPERoundData(
            final_rotation=0,
            measurement=0,
            num_rotations=1
        )])
        test_experiment3 = QPEExperimentData([QPERoundData(
            final_rotation=0,
            measurement=1,
            num_rotations=1
        )])
        num2 = 0
        for j in range(100):
            be.update(test_experiment1)
            if j % 2 == 0:
                be.update(test_experiment2)
                num2 += 1
            else:
                be.update(test_experiment3)

        self.assertEqual(be.estimate(), -numpy.pi/2)

    def test_update_warnings(self):

        be = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=numpy.array([[1]]),
                            num_freqs=10,
                            max_rotations=1)

        mock_result = numpy.array([[-1, 0], [0, 0]])

        def mock_function(**kwargs):
            return mock_result
        be._calc_vectors = mock_function
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(be.update(QPEExperimentData()))
            self.assertEqual(len(w), 2)

        be = BayesEstimator(num_vectors=1,
                            prior_amplitude_mean=[1],
                            prior_amplitude_var=numpy.array([[1]]),
                            num_freqs=10,
                            max_rotations=1)

        test_experiment1 = QPEExperimentData([QPERoundData(
            final_rotation=numpy.pi/2,
            measurement=0,
            num_rotations=1
        )])

        def mock_function(**kwargs):
            pass
        be._update_amplitudes = mock_function
        be._probability_dist._amplitude_estimates = numpy.array([-1])
        with warnings.catch_warnings(record=True) as w:
            be.update(test_experiment1)
            self.assertFalse(be.update(test_experiment1))
            self.assertEqual(len(w), 2)

    def test_amplitude_approx_update(self):
        be = BayesEstimator(num_vectors=2,
                            prior_amplitude_mean=[0.9, 0.1],
                            prior_amplitude_var=[[0.1, -0.09], [-0.09, 0.1]],
                            num_freqs=10,
                            max_rotations=1,
                            amplitude_approx_cutoff=15,
                            full_update_with_failure=True)
        be2 = BayesEstimator(num_vectors=2,
                             prior_amplitude_mean=[0.9, 0.1],
                             prior_amplitude_var=[[0.1, -0.09], [-0.09, 0.1]],
                             num_freqs=10,
                             max_rotations=1,
                             amplitude_approx_cutoff=10)

        test_experiment = QPEExperimentData([QPERoundData(
            final_rotation=-numpy.pi/4,
            measurement= 0,
            num_rotations= 1
        )])
        test_experiment2 = QPEExperimentData([QPERoundData(
            final_rotation=numpy.pi/4,
            measurement=0,
            num_rotations=2
        )])
        test_experiment3 = QPEExperimentData([QPERoundData(
            final_rotation=0,
            measurement=0,
            num_rotations=2
        )])

        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            for j in range(6):
                be.update(test_experiment)
                be2.update(test_experiment)
                be.update(test_experiment2)
                be2.update(test_experiment2)
                be.update(test_experiment3)
                be2.update(test_experiment3)

        self.assertTrue(
            numpy.sum(numpy.abs(
                be._probability_dist._amplitude_estimates-
                be2._probability_dist._amplitude_estimates)) < 1e-3)

    def test_full(self):

        def do_experiment(ev, experiment, random_state):
            for round_data in experiment.rounds:
                n = round_data.num_rotations
                beta = round_data.final_rotation
                p0 = numpy.cos(n*ev/2 + beta/2)**2
                if p0 > random_state.uniform(0, 1):
                    round_data.measurement = 0
                else:
                    round_data.measurement = 1

        angles = [0, numpy.pi/2]
        random_state = numpy.random.RandomState(seed=42)
        ev = random_state.uniform(-numpy.pi, numpy.pi)
        sampler = IteratingSampler(50, angles, random_state)
        estimator = BayesEstimator(num_vectors=1, max_rotations=50,
                                   prior_amplitude_mean=[1],
                                   prior_amplitude_var=[[1]],
                                   num_freqs=1000*2,
                                   amplitude_approx_cutoff=100)
        for j in range(1000):
            experiment = sampler.sample()
            do_experiment(ev, experiment, random_state)
            estimator.update(experiment)

        self.assertTrue(numpy.abs(ev-estimator.estimate()[0]) < 1e-1)

    def test_depol_failure(self):

        def do_experiment_depol(ev, experiment, random_state, T2):
            for round_data in experiment.rounds:
                n = round_data.num_rotations
                beta = round_data.final_rotation
                p_noerr = numpy.exp(-n/T2)
                if p_noerr > random_state.uniform(0, 1):
                    p0 = numpy.cos(n*ev/2 + beta/2)**2
                else:
                    p0 = 0.5
                if p0 > random_state.uniform(0, 1):
                    round_data.measurement = 0
                else:
                    round_data.measurement = 1
                round_data.true_measurement = round_data.measurement

        angles = [0, numpy.pi/2]
        random_state = numpy.random.RandomState(seed=42)
        ev = random_state.uniform(-numpy.pi, numpy.pi)
        sampler = IteratingSampler(50, angles, random_state)
        estimator = BayesEstimator(num_vectors=1, max_rotations=10,
                                   prior_amplitude_mean=[1],
                                   prior_amplitude_var=[[1]],
                                   num_freqs=1000*5,
                                   amplitude_approx_cutoff=100)

        with warnings.catch_warnings(record=True) as w:
            for j in range(1000):
                experiment = sampler.sample()
                do_experiment_depol(ev, experiment, random_state, T2=20)
                estimator.update(experiment)
            self.assertGreater(len(w), 1)

        self.assertFalse(numpy.isfinite(estimator.estimate()[0]))


class BayesDepolarizingEstimatorTest(unittest.TestCase):

    def test_epsilons(self):
        estimator = BayesDepolarizingEstimator(
            num_vectors=1,
            prior_amplitude_mean=[1],
            prior_amplitude_var=numpy.array([[1]]),
            num_freqs=1000,
            max_rotations=1,
            k_1=1,
            k_err=1)
        self.assertAlmostEqual(estimator._epsilon_d_function(2),
                               1-numpy.exp(-2))
        self.assertAlmostEqual(estimator._epsilon_b_function(),
                               1-numpy.exp(-1))

    def test_vector_product(self):
        estimator = BayesDepolarizingEstimator(
            num_vectors=1,
            prior_amplitude_mean=[1],
            prior_amplitude_var=numpy.array([[1]]),
            num_freqs=1000,
            max_rotations=1,
            k_1=1,
            k_err=1)

        vectors = numpy.zeros(2001)
        vectors[0] = 1
        round_data = QPERoundData(
            measurement= 0,
            num_rotations= 2,
            final_rotation= 0)
        updated_vectors = estimator._vector_product(vectors, round_data)
        epsilon_D = 1-numpy.exp(-2)
        epsilon_B = 1-numpy.exp(-1)
        self.assertAlmostEqual(updated_vectors[0], 0.5+0.5*epsilon_B)
        self.assertAlmostEqual(updated_vectors[1], 0)
        self.assertAlmostEqual(updated_vectors[2], 0)
        self.assertAlmostEqual(updated_vectors[3], 0)
        self.assertAlmostEqual(updated_vectors[4], 0.5*(1-epsilon_D))

    def test_full(self):

        def do_experiment_depol(ev, experiment, random_state, T2):
            for round_data in experiment.rounds:
                n = round_data.num_rotations
                beta = round_data.final_rotation
                p_noerr = numpy.exp(-n/T2)
                if p_noerr > random_state.uniform(0, 1):
                    p0 = numpy.cos(n*ev/2 + beta/2)**2
                else:
                    p0 = 0.5
                if p0 > random_state.uniform(0, 1):
                    round_data.measurement = 0
                else:
                    round_data.measurement = 1
                round_data.true_measurement = round_data.measurement

        angles = [0, numpy.pi/2]
        random_state = numpy.random.RandomState(seed=42)
        ev = random_state.uniform(-numpy.pi, numpy.pi)
        sampler = IteratingSampler(10, angles, random_state)
        estimator = BayesDepolarizingEstimator(num_vectors=1, max_rotations=10,
                                               prior_amplitude_mean=[1],
                                               prior_amplitude_var=[[1]],
                                               num_freqs=1000*5,
                                               amplitude_approx_cutoff=100,
                                               k_err=20)
        for j in range(1000):
            experiment = sampler.sample()
            do_experiment_depol(ev, experiment, random_state, T2=20)
            estimator.update(experiment)

        self.assertLess(numpy.abs(ev-estimator.estimate()[0]), 2e-1)
