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

"""Tests for _probability_distributions.py"""
import unittest
import numpy
from numpy import pi
from ._probability_distributions import FourierProbabilityDist


class FourierProbabilityDistTest(unittest.TestCase):

    def test_basic_initialization(self):
        pd = FourierProbabilityDist(num_vectors=1,
                                    num_freqs=10)
        self.assertEqual(pd._num_freqs, 10)
        self.assertEqual(pd._num_vectors, 1)
        self.assertEqual(len(pd._amplitude_estimates), 1)
        self.assertEqual(pd._fourier_vectors.shape, (21, 1))
        self.assertAlmostEqual(pd._holevo_variances(maxvar=1)[0], 1)

    def test_raises_errors(self):

        with self.assertRaises(ValueError):
            FourierProbabilityDist(num_vectors=1,
                                   prior_vector=numpy.array([[1, 0], [1, 0]]),
                                   prior_amplitude_mean=[1],
                                   prior_amplitude_var=[[1]],
                                   num_freqs=10)
        with self.assertRaises(ValueError):
            FourierProbabilityDist(num_vectors=1,
                                   prior_amplitude_mean=[1, 0],
                                   prior_amplitude_var=[[1]],
                                   num_freqs=10)
        with self.assertRaises(ValueError):
            FourierProbabilityDist(num_vectors=2,
                                   prior_amplitude_mean=[0.5, 0.5],
                                   prior_amplitude_var=[[1]],
                                   num_freqs=10)

    def test_init_dist(self):

        prior_vector = numpy.zeros([21, 1])
        prior_vector[0, 0] = 1
        prior_vector[2, 0] = 1  # cos wave
        pd = FourierProbabilityDist(num_vectors=1,
                                    prior_amplitude_mean=[1],
                                    prior_amplitude_var=[[1]],
                                    num_freqs=10,
                                    prior_vector=prior_vector)
        x_vec = numpy.linspace(-pi, pi, 11)
        dist = pd.get_real_dist()
        dist = pd.get_real_dist(x_vec)
        dist_comp = (1 + numpy.cos(x_vec)) / (2*pi)
        self.assertAlmostEqual(numpy.sum(numpy.abs(dist-dist_comp)), 0)

    def test_holevo(self):
        prior_vector = numpy.zeros([21, 1])
        prior_vector[0, 0] = 1
        prior_vector[2, 0] = 1  # sine wave
        pd = FourierProbabilityDist(num_vectors=1,
                                    prior_amplitude_mean=[1],
                                    prior_amplitude_var=[[1]],
                                    num_freqs=10,
                                    prior_vector=prior_vector)
        self.assertEqual(pd._holevo_centers(), 0)
        self.assertEqual(pd._holevo_variances()[0], 3)
