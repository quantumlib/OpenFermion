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

import unittest

import numpy
from numpy.random import rand, seed
from numpy.linalg import norm

from openfermion.utils.bch_expansion import bch_expand


def bch_expand_baseline(x, y, order):
    """Compute log[e^x e^y] using the Baker-Campbell-Hausdorff formula
    Args:
        x: An operator for which multiplication and addition are supported.
            For instance, a QubitOperator, FermionOperator or numpy array.
        y: The same type as x.
        order(int): The order to truncate the BCH expansions. Currently
            function goes up to only third order.

    Returns:
        z: The truncated BCH operator.

    Raises:
        ValueError: operator x is not same type as operator y.
        ValueError: invalid order parameter.
        ValueError: order exceeds maximum order supported.
    """
    from openfermion.utils import commutator

    # First order.
    z = x + y

    # Second order.
    if order > 1:
        z += commutator(x, y) / 2.0

    # Third order.
    if order > 2:
        z += commutator(x, commutator(x, y)) / 12.0
        z += commutator(y, commutator(y, x)) / 12.0

    # Fourth order.
    if order > 3:
        z -= commutator(y, commutator(x, commutator(x, y))) / 24.0

    # Fifth order.
    if order > 4:
        z -= commutator(y, commutator(y, commutator(y, commutator(y, x)))) / 720.0
        z -= commutator(x, commutator(x, commutator(x, commutator(x, y)))) / 720.0
        z += commutator(x, commutator(y, commutator(y, commutator(y, x)))) / 360.0
        z += commutator(y, commutator(x, commutator(x, commutator(x, y)))) / 360.0
        z += commutator(y, commutator(x, commutator(y, commutator(x, y)))) / 120.0
        z += commutator(x, commutator(y, commutator(x, commutator(y, x)))) / 120.0

    return z


class BCHTest(unittest.TestCase):
    def setUp(self):
        """Initialize a few density matrices"""
        self.seed = [13579, 34628, 2888, 11111, 67917]
        self.dim = 6
        self.test_order = 5

    def test_bch(self):
        """Test efficient bch expansion against hard-coded baseline
        coefficients"""
        for s in self.seed:
            seed(s)
            x = rand(self.dim, self.dim)
            y = rand(self.dim, self.dim)
            z = rand(self.dim, self.dim)

            test = bch_expand(x, y, order=self.test_order)
            baseline = bch_expand_baseline(x, y, order=self.test_order)
            self.assertAlmostEqual(norm(test - baseline), 0.0)

            test = bch_expand(x, y, z, order=self.test_order)
            baseline = bch_expand_baseline(
                x, bch_expand_baseline(y, z, order=self.test_order), order=self.test_order
            )
            self.assertAlmostEqual(norm(test - baseline), 0.0)

    def test_verification(self):
        """Verify basic sanity checking on inputs"""
        with self.assertRaises(TypeError):
            order = 2
            _ = bch_expand(1, numpy.ones((2, 2)), order=order)

        with self.assertRaises(ValueError):
            order = '38'
            _ = bch_expand(numpy.ones((2, 2)), numpy.ones((2, 2)), order=order)

        with self.assertRaises(ValueError):
            _ = bch_expand(1)
