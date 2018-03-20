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

"""Tests for _lcu_util.py."""

from __future__ import absolute_import, division

import random
import unittest

from openfermion.utils._lcu_util import (
    _preprocess_for_efficient_roulette_selection,
    _discretize_probability_distribution,
    preprocess_lcu_coefficients_for_reversible_sampling,
)


class DiscretizeDistributionTest(unittest.TestCase):

    def assertGetDiscretizedDistribution(self, probabilities, epsilon):
        total_probability = sum(probabilities)
        numers, denom = _discretize_probability_distribution(probabilities,
                                                             epsilon)
        self.assertEqual(sum(numers), denom)
        self.assertEqual(len(numers), len(probabilities))
        for i in range(len(numers)):
            self.assertAlmostEqual(numers[i] / denom,
                                   probabilities[i] / total_probability,
                                   delta=epsilon)
        return numers, denom

    def test_fuzz(self):
        for _ in range(100):
            n = random.randint(1, 50)
            weights = [random.random() for _ in range(n)]
            self.assertGetDiscretizedDistribution(
                weights, 2**-random.randint(1, 20))

    def test_known_discretizations(self):
        self.assertEqual(
            self.assertGetDiscretizedDistribution([1], 0.25),
            ([4], 4))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([1], 0.125),
            ([8], 8))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([0.1, 0.1, 0.1], 0.25),
            ([2, 2, 2], 6))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.25),
            ([2, 2, 2], 6))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.1),
            ([4, 4, 4], 12))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.05),
            ([6, 8, 7], 21))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.01),
            ([31, 37, 34], 102))

        self.assertEqual(
            self.assertGetDiscretizedDistribution([0.09, 0.11, 0.1], 0.00335),
            ([90, 110, 100], 300))


class PreprocessForEfficientRouletteSelectionTest(unittest.TestCase):

    def assertPreprocess(self, weights):
        alternates, keep_chances = (
            _preprocess_for_efficient_roulette_selection(weights))

        self.assertEqual(len(alternates), len(keep_chances))

        target_weight = sum(weights) // len(alternates)
        distribution = list(keep_chances)
        for i in range(len(alternates)):
            distribution[alternates[i]] += target_weight - keep_chances[i]
        self.assertEqual(weights, distribution)

        return alternates, keep_chances

    def test_fuzz(self):
        for _ in range(100):
            n = random.randint(1, 50)
            weights = [random.randint(0, 100) for _ in range(n)]
            weights[-1] += n - sum(weights) % n  # Ensure multiple of length.
            self.assertPreprocess(weights)

    def test_validation(self):
        with self.assertRaises(ValueError):
            _ = self.assertPreprocess(weights=[])
        with self.assertRaises(ValueError):
            _ = self.assertPreprocess(weights=[1, 2])
        with self.assertRaises(ValueError):
            _ = self.assertPreprocess(weights=[3, 3, 2])

    def test_already_uniform(self):
        self.assertEqual(
            self.assertPreprocess(weights=[1]),
            ([0], [0]))
        self.assertEqual(
            self.assertPreprocess(weights=[1, 1]),
            ([0, 1], [0, 0]))
        self.assertEqual(
            self.assertPreprocess(weights=[1, 1, 1]),
            ([0, 1, 2], [0, 0, 0]))
        self.assertEqual(
            self.assertPreprocess(weights=[2, 2, 2]),
            ([0, 1, 2], [0, 0, 0]))

    def test_donation(self):
        # v2 donates 1 to v0.
        self.assertEqual(
            self.assertPreprocess(weights=[1, 2, 3]),
            ([2, 1, 2], [1, 0, 0]))
        # v0 donates 1 to v1.
        self.assertEqual(
            self.assertPreprocess(weights=[3, 1, 2]),
            ([0, 0, 2], [0, 1, 0]))
        # v0 donates 1 to v1, then 2 to v2.
        self.assertEqual(
            self.assertPreprocess(weights=[5, 1, 0]),
            ([0, 0, 0], [0, 1, 0]))

    def test_over_donation(self):
        # v0 donates 2 to v1, leaving v0 needy, then v2 donates 1 to v0.
        self.assertEqual(
            self.assertPreprocess(weights=[3, 0, 3]),
            ([2, 0, 2], [1, 0, 0]))


class PreprocessLCUCoefficientsForReversibleSamplingTest(unittest.TestCase):
    def assertPreprocess(self, lcu_coefs, epsilon):
        alternates, keep_numers, keep_denom = (
            preprocess_lcu_coefficients_for_reversible_sampling(
                lcu_coefs, epsilon))

        n = len(lcu_coefs)
        self.assertEqual(len(alternates), n)
        self.assertEqual(len(keep_numers), n)

        out_distribution = [numer/keep_denom for numer in keep_numers]
        for i in range(n):
            out_distribution[alternates[i]] += 1/n - keep_numers[i]/keep_denom

        total = sum(lcu_coefs)
        for i in range(n):
            self.assertAlmostEqual(out_distribution[i],
                                   lcu_coefs[i] / total,
                                   delta=epsilon)

        return alternates, keep_numers, keep_denom

    def test_fuzz(self):
        for _ in range(100):
            n = random.randint(1, 50)
            weights = [random.randint(0, 100) for _ in range(n)]
            weights[-1] += n - sum(weights) % n  # Ensure multiple of length.
            self.assertPreprocess(weights, 2**-random.randint(1, 20))

    def test_known(self):
        self.assertEqual(
            self.assertPreprocess([1, 2], epsilon=0.01),
            ([1, 1], [33, 0], 100))
