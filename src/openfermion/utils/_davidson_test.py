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
import scipy.linalg
import scipy.sparse.linalg

from openfermion.ops import QubitOperator
from openfermion.utils._davidson import Davidson, QubitDavidson

def generate_matrix(dimension):
    """Generates matrix with shape (dimension, dimension)."""
    numpy.random.seed(dimension)
    rand = numpy.array(numpy.random.rand(dimension, dimension))

    numpy.random.seed(dimension * 2)
    diag = numpy.array(range(dimension)) + numpy.random.rand(dimension)

    # Makes sure matrix is hermitian, which is symmetric when real.
    matrix = rand + rand.T + numpy.diag(diag)
    return matrix

def get_difference(linear_operator, eigen_values, eigen_vectors):
    """Get difference of M * v - lambda v."""
    return numpy.max(numpy.abs(linear_operator * eigen_vectors -
                               eigen_vectors * eigen_values))


class DavidsonTest(unittest.TestCase):
    """"Tests for Davidson class with a real matrix."""

    def setUp(self):
        """Sets up all variables needed for Davidson class."""
        dimension = 10
        matrix = generate_matrix(dimension)

        def mat_vec(vec):
            """Trivial matvec with a numpy matrix."""
            return numpy.dot(matrix, vec)

        self.linear_operator = scipy.sparse.linalg.LinearOperator(
            (dimension, dimension), matvec=mat_vec)
        self.diagonal = numpy.diag(matrix)
        self.eps = 1e-6

        self.davidson = Davidson(linear_operator=self.linear_operator,
                                 linear_operator_diagonal=self.diagonal,
                                 eps=self.eps)
        self.matrix = matrix
        self.dimension = dimension
        self.initial_guess = numpy.eye(self.dimension, 10)

        self.eigen_values = numpy.array([
            1.15675714, 1.59132505, 2.62268014, 4.44533793, 5.3722743,
            5.54393114, 7.73652405, 8.50089897, 9.4229309, 15.54405993,
        ])

    def test_init(self):
        """Test for __init__()."""
        davidson = self.davidson

        self.assertAlmostEqual(numpy.max(numpy.abs(
            self.matrix - self.matrix.T)), 0)

        self.assertTrue(davidson.linear_operator)
        self.assertTrue(numpy.allclose(davidson.linear_operator_diagonal,
                                       self.diagonal))
        self.assertAlmostEqual(davidson.eps, self.eps, places=8)

    def test_orthonormalize(self):
        """Test for orthonormalization."""
        sqrt_half = numpy.sqrt(0.5)
        expected_array = numpy.array([
            [sqrt_half, sqrt_half, 0],
            [sqrt_half, -sqrt_half, 0],
            [0, 0, 1],
        ])

        array = numpy.array([[1, 1, 10], [1, -1, 10], [0, 0, 2]], dtype=float)
        array[:, 0] *= sqrt_half
        array = self.davidson.orthonormalize(array, 1)
        self.assertTrue(numpy.allclose(array, expected_array))

    def test_orthonormalize_complex(self):
        """Test for orthonormalization with complex matrix."""
        sqrt_half = numpy.sqrt(0.5)
        expected_array = numpy.array([
            [sqrt_half * 1.0j, sqrt_half * 1.0j, 0],
            [sqrt_half * 1.0j, -sqrt_half * 1.0j, 0],
            [0, 0, 1],
        ], dtype=complex)

        array = numpy.array([[1.j, 1.j, 10], [1.j, -1.j, 10], [0, 0, 2]], dtype=complex)
        array[:, 0] *= sqrt_half
        array = self.davidson.orthonormalize(array, 1)
        self.assertTrue(numpy.allclose(array, expected_array))

    def test_with_built_in(self):
        """Compare with eigenvalues from built-in functions."""
        eigen_values, _ = numpy.linalg.eig(self.matrix)
        eigen_values = sorted(eigen_values)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values))

        # Checks for eigh() function.
        eigen_values, eigen_vectors = numpy.linalg.eigh(self.matrix)
        self.assertAlmostEqual(get_difference(self.davidson.linear_operator,
                                              eigen_values, eigen_vectors), 0)

    def test_lowest_invalid_subspace(self):
        """Test for get_lowest_n() with invalid max_subspace."""
        with self.assertRaises(ValueError):
            Davidson(None, numpy.zeros(8), 1)

    def test_lowest_invalid_eps(self):
        """Test for get_lowest_n() with invalid eps."""
        with self.assertRaises(ValueError):
            Davidson(None, numpy.zeros(8), eps=-1e-6)

    def test_lowest_zero_n(self):
        """Test for get_lowest_n() with invalid n_lowest."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(0)

    def test_lowest_invalid_shape(self):
        """Test for get_lowest_n() with invalid dimension for initial guess."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(1, numpy.ones((self.dimension * 2, 1),
                                                     dtype=complex))

    def test_get_lowest_n_trivial_guess(self):
        """Test for get_lowest_n() with trivial initial guess."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(1, numpy.zeros((self.dimension, 1),
                                                      dtype=complex))

    def test_get_lowest_fail(self):
        """Test for get_lowest_n() with n_lowest = 1."""
        n_lowest = 1
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=2)

        self.assertTrue(not success)
        self.assertTrue(numpy.allclose(eigen_values,
                                       numpy.array([1.41556103])))

    def test_get_lowest_one(self):
        """Test for get_lowest_n() with n_lowest = 1."""
        n_lowest = 1
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=10)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values,
                                       self.eigen_values[:n_lowest]))

    def test_get_lowest_two(self):
        """Test for get_lowest_n() with n_lowest = 2.

        See the iteration results (eigenvalues and max error) below:
            [1.87267714 4.06259537] 3.8646520980719212
            [1.28812931 2.50316266] 1.548676934730246
            [1.16659255 1.82600658] 0.584638880856119
            [1.15840263 1.65254981] 0.4016803134102507
            [1.15675714 1.59132505] 0
        """
        n_lowest = 2
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, eigen_vectors = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=5)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values,
                                       self.eigen_values[:n_lowest]))
        self.assertTrue(numpy.allclose(
            self.davidson.linear_operator * eigen_vectors,
            eigen_vectors * eigen_values))

    def test_get_lowest_two_subspace(self):
        """Test for get_lowest_n() with n_lowest = 2.

        See the iteration results (eigenvalues and max error) below:
            [1.87267714 4.06259537] 3.8646520980719212
            [1.28812931 2.50316266] 1.548676934730246
            [1.16659255 1.82600658] 0.584638880856119
            [1.15947254 1.69773006] 0.5077687725257688
            [1.1572995  1.61393264] 0.3318982487563453

        """
        self.davidson.max_subspace = 8
        expected_eigen_values = numpy.array([1.1572995, 1.61393264])

        n_lowest = 2
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, eigen_vectors = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=5)

        self.assertTrue(not success)
        self.assertTrue(numpy.allclose(eigen_values, expected_eigen_values))
        self.assertFalse(numpy.allclose(
            self.davidson.linear_operator * eigen_vectors,
            eigen_vectors * eigen_values))

    def test_get_lowest_six(self):
        """Test for get_lowest_n() with n_lowest = 6."""
        n_lowest = 6
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=2)
        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values[:n_lowest]))

    def test_get_lowest_all(self):
        """Test for get_lowest_n() with n_lowest = 10."""
        n_lowest = 10
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=1)
        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values[:n_lowest]))


class QubitDavidsonTest(unittest.TestCase):
    """"Tests for QubitDavidson class with a QubitOperator."""

    def setUp(self):
        """Sets up all variables needed for QubitDavidson class."""
        self.coefficient = 2
        self.n_qubits = 12

    def test_get_lowest_n(self):
        """Test for get_lowest_n()."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator.zero()
        for i in range(min(self.n_qubits, 4)):
            numpy.random.seed(dimension + i)
            qubit_operator += QubitOperator(((i, 'Z'),),
                                            numpy.random.rand(1)[0])
        qubit_operator *= self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)

        n_lowest = 6
        numpy.random.seed(dimension)
        initial_guess = numpy.random.rand(dimension, n_lowest)
        success, eigen_values, eigen_vectors = davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=20)

        expected_eigen_values = -3.80376934 * numpy.ones(n_lowest)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, expected_eigen_values))
        self.assertAlmostEqual(get_difference(davidson.linear_operator,
                                              eigen_values, eigen_vectors), 0)

    def test_get_lowest_zzx(self):
        """Test for get_lowest_n() for one term only within 10 iterations.
        Also the number of starting vectors is smaller than n_lowest."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator('Z0 Z1 X2') * self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)

        n_lowest = 6
        numpy.random.seed(dimension)
        initial_guess = numpy.random.rand(dimension, n_lowest // 2)
        success, eigen_values, eigen_vectors = davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=10)

        # one half of the eigenvalues is -1 and the other half is +1, together
        # with the coefficient.
        expected_eigen_values = -self.coefficient * numpy.ones(n_lowest)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, expected_eigen_values))
        self.assertAlmostEqual(get_difference(davidson.linear_operator,
                                              eigen_values, eigen_vectors), 0)

    def test_get_lowest_xyz(self):
        """Test for get_lowest_n() for one term only within 10 iterations."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator('X0 Y1 Z3') * self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)

        n_lowest = 6
        # Guess vectors have both real and imaginary parts.
        numpy.random.seed(dimension)
        initial_guess = 1.0j * numpy.random.rand(dimension, n_lowest)
        numpy.random.seed(dimension * 2)
        initial_guess += numpy.random.rand(dimension, n_lowest)
        success, eigen_values, eigen_vectors = davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=10)

        # one half of the eigenvalues is -1 and the other half is +1, together
        # with the coefficient.
        expected_eigen_values = -self.coefficient * numpy.ones(n_lowest)

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, expected_eigen_values))
        self.assertAlmostEqual(get_difference(davidson.linear_operator,
                                              eigen_values, eigen_vectors), 0)
