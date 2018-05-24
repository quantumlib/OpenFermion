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

"""Tests for _davidson.py."""

from __future__ import absolute_import, division

import unittest
import numpy
import numpy.linalg
import scipy.sparse.linalg

from openfermion.utils._davidson import *

def generate_matrix(dimension):
    """Generates matrix with shape (dimension, dimension)."""
    numpy.random.seed(dimension)
    rand = numpy.array(numpy.random.rand(dimension, dimension))

    numpy.random.seed(dimension)
    diag = numpy.array(xrange(dimension)) + numpy.random.rand(dimension)

    # Makes sure matrix is hermitian, which is symmetric when real.
    matrix = rand + rand.T + diag
    return matrix


class DavidsonTest(unittest.TestCase):
    """"Tests for Davidson class."""

    def setUp(self):
        """Sets up all variables needed for Davidson class."""
        dimension = 10
        matrix = generate_matrix(dimension)

        def mat_vec(vec):
            """Trivial matvec with a numpy matrix."""
            return numpy.dot(matrix, vec)

        self.linear_op = scipy.sparse.linalg.LinearOperator((dimension,
                                                             dimension),
                                                            matvec=mat_vec)
        self.diagonal = numpy.diag(matrix)
        self.eps = 1e-8

        self.davidson = Davidson(self.linear_op, self.diagonal, self.eps)
        self.matrix = matrix
        self.dimension = dimension
        self.initial_guess = numpy.eye(self.dimension, 10)

        self.eigen_values = numpy.array([
            -2.39763401, -1.21764127, -0.51686198, -0.36457375, 0.58244882,
            0.82719468, 1.13003449, 1.69418437, 1.90207692, 58.74151042])

    def test_init(self):
        """Test for __init__()."""
        davidson = self.davidson

        self.assertTrue(davidson.linear_operator)
        self.assertTrue(numpy.allclose(davidson.linear_operator_diagonal,
                                       self.diagonal))
        self.assertAlmostEqual(davidson.eps, self.eps, places=8)

    def test_with_built_in(self):
        """Compare with eigen values from built-in functions."""
        eigen_values, _ = numpy.linalg.eig(self.matrix)
        eigen_values = sorted(eigen_values)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values))

    def test_get_lowest_n_zero_n(self):
        """Test for get_lowest_n() with invalid n_lowest."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(0)

    def test_get_lowest_one(self):
        """Test for get_lowest_n() with n_lowest = 1."""
        n_lowest = 1
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values,
                                       self.eigen_values[:n_lowest]))

    def test_get_lowest_two(self):
        """Test for get_lowest_n() with n_lowest = 2."""
        n_lowest = 2
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, eigen_vectors = self.davidson.get_lowest_n(
            n_lowest, initial_guess)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values,
                                       self.eigen_values[:n_lowest]))
        self.assertTrue(numpy.allclose(
            self.davidson.linear_operator * eigen_vectors,
            eigen_vectors * eigen_values))

    def test_get_lowest_six(self):
        """Test for get_lowest_n() with n_lowest = 6."""
        n_lowest = 6
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess)
        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values[:n_lowest]))

    def test_get_lowest_all(self):
        """Test for get_lowest_n() with n_lowest = 10."""
        n_lowest = 10
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess)
        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values[:n_lowest]))
