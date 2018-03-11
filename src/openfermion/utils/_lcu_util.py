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

"""Utility methods for LCU circuits."""

from __future__ import absolute_import
from __future__ import division

import math


def _partial_sums(vals):
    total = 0
    for v in vals:
        yield total
        total += v
    yield total


def _differences(weights):
    p = None
    have_p = False
    for w in weights:
        if have_p:
            yield w - p
        p = w
        have_p = True


def _discretize_probability_distribution(unnormalized_probabilities, epsilon):
    """Approximates probabilities with integers over a common denominator.

    Args:
        unnormalized_probabilities: A list of non-negative numbers proportional
            to probabilities from a probability distribution. The numbers may
            not be normalized (they do not have to add up to 1).
        epsilon: The absolute error tolerance.

    Returns:
        A tuple containing:
            [0]: A list of numerators.
            [1]: A common denominator.

        It is guaranteed that numerators[i] / denominator is within epsilon of
        the i'th input probability (after normalization).

        It is guaranteed that the common denominator is a multiple of the
        number of probabilities in the distribution.
    """
    n = len(unnormalized_probabilities)
    bin_count = math.ceil(1.0 / (epsilon * n)) * n

    cumulative = list(_partial_sums(unnormalized_probabilities))
    total = cumulative[-1]
    discretized_cumulative = [int(math.floor(c/total*bin_count + 0.5))
                              for c in cumulative]
    discretized = list(_differences(discretized_cumulative))
    return discretized, bin_count


def _preprocess_for_efficient_roulette_selection(discretized_probabilities):
    """Prepares data for performing efficient roulette selection.

    The output is a tuple (alternates, keep_weights). The output is guaranteed
    to satisfy a sampling-equivalence property. Specifically, the following
    sampling process is guaranteed to be equivalent to simply picking index i
    with probability weights[i] / sum(weights):

        1. Pick a number i from 0 (inclusive) to len(weights) (exclusive)
        2. With probability keep_weights[i] / sum(weights), return 1
        3. Otherwise return alternates[i]

    In other words, the output makes it possible to perform roulette selection
    while generating only two random numbers, doing a single lookup of the
    relevant (keep_chance, alternate) pair, and doing one comparison. This is
    not so useful classically, but in the context of a quantum computation
    where all those things are expensive the second sampling process is far
    superior.

    Args:
        discretized_probabilities: A list of probabilities approximated by
            integer numerators (with an implied common denominator). In order
            to operate without floating point error, it is required that the
            sum of this list is a multiple of the number of items in the list.

    Returns:
        A tuple containing:
            [0]: An alternate item for each index from 0 to len(weights) - 1
            [1]: A list of "donated weight", indicating how often one should
                stay at index i instead of switching to alternate[i].
    """
    weights = list(discretized_probabilities)  # Need a copy we can mutate.
    if not weights:
        raise ValueError('Empty input.')

    n = len(weights)
    target_weight = sum(weights) // n
    if sum(weights) != n * target_weight:
        raise ValueError('sum(weights) must be a multiple of len(weights).')

    # Initially, every item's alternative is itself.
    alternates = list(range(n))
    keep_weights = [0] * n

    # Scan for needy items and donors. First pass will handle all
    # initially-needy items. Second pass will handle any remaining items that
    # started as donors but become needy due to over-donation (though some may
    # also be handled during the first pass).
    donor_position = 0
    for _ in range(2):
        for i in range(n):
            # Is this a needy item?
            if weights[i] >= target_weight:
                continue  # Nope.

            # Find a donor.
            while weights[donor_position] <= target_weight:
                donor_position += 1

            # Donate.
            donated = target_weight - weights[i]
            weights[donor_position] -= donated
            alternates[i] = donor_position
            keep_weights[i] = weights[i]

            # Needy item has been paired. Remove it from consideration.
            weights[i] = target_weight

    return alternates, keep_weights


def preprocess_lcu_coefficients_for_reversible_sampling(
        lcu_coefficients,
        epsilon):
    """Prepares data used to perform efficient reversible roulette selection.

    Treats the coefficients of unitaries in the linear combination of
    unitaries decomposition of the Hamiltonian as probabilities in order to
    decompose them into a list of alternate and keep_numers allowing for
    an efficient preparation method of a state where the computational basis
    state |k> has an amplitude proportional to the coefficient.

    Args:
        lcu_coefficients: A list of non-negative floats, with the i'th float
            corresponding to the i'th coefficient of an LCU decomposition
            of the Hamiltonian (in an ordering determined by the caller).
        epsilon: Absolute error tolerance.

    Returns:
        An (alternates, keep_numers, keep_denom) tuple containing:
            [0]: A python list of ints indicating alternative indices that may
                be switched to after generating a uniform index. The int at
                offset k is the alternate to use when the initial index is k.
            [1]: A python list of ints indicating the numerators of the
                probability that the alternative index should be used instead
                of the initial index.
            [2]: A python int indicating the denominator to divide the
                numerators in [1] by in order to get a probability.

            It is guaranteed keep_numers[i] / keep_denom will be within epsilon
            of lcu_coefficients[i] / sum(lcu_coefficients).
    """
    numers, denom = _discretize_probability_distribution(
        lcu_coefficients, epsilon)
    alternates, keep_numers = _preprocess_for_efficient_roulette_selection(
        numers)
    return alternates, keep_numers, denom
