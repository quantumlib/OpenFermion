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


import logging
import unittest

import numpy
import numpy.linalg
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from openfermion.ops import QubitOperator
from openfermion.utils._davidson import (
    Davidson,
    DavidsonOptions,
    QubitDavidson,
    SparseDavidson,
    append_random_vectors,
    orthonormalize)

def generate_matrix(dimension):
    """Generates matrix with shape (dimension, dimension)."""
    numpy.random.seed(dimension)
    rand = numpy.array(numpy.random.rand(dimension, dimension))

    numpy.random.seed(dimension * 2)
    diag = numpy.array(range(dimension)) + numpy.random.rand(dimension)

    # Makes sure matrix is hermitian, which is symmetric when real.
    matrix = rand + rand.conj().T + numpy.diag(diag)
    return matrix

def generate_sparse_matrix(dimension, diagonal_factor=30):
    """Generates a hermitian sparse matrix with specified dimension."""
    numpy.random.seed(dimension)
    diagonal = sorted(numpy.array(numpy.random.rand(dimension)))

    numpy.random.seed(dimension - 1)
    off_diagonal = numpy.array(numpy.random.rand(dimension - 1))

    # Makes sure matrix is hermitian, which is symmetric when real.
    matrix = numpy.diag(diagonal) * diagonal_factor
    for row in range(dimension - 2):
        col = row + 1
        matrix[row, col] = off_diagonal[row]
        matrix[col, row] = off_diagonal[row]
    return matrix

def get_difference(linear_operator, eigen_values, eigen_vectors):
    """Get difference of M * v - lambda v."""
    return numpy.max(numpy.abs(linear_operator.dot(eigen_vectors) -
                               eigen_vectors * eigen_values))


class DavidsonOptionsTest(unittest.TestCase):
    """"Tests for DavidsonOptions class."""

    def setUp(self):
        """Sets up all variables needed for DavidsonOptions class."""
        self.max_subspace = 10
        self.max_iterations = 100
        self.eps = 1e-7
        self.davidson_options = DavidsonOptions(self.max_subspace,
                                                self.max_iterations, self.eps)

    def test_init(self):
        """Tests vars in __init__()."""
        self.assertEqual(self.davidson_options.max_subspace, self.max_subspace)
        self.assertEqual(self.davidson_options.max_iterations,
                         self.max_iterations)
        self.assertAlmostEqual(self.davidson_options.eps, self.eps, places=8)
        self.assertFalse(self.davidson_options.real_only)

    def test_set_dimension_small(self):
        """Tests set_dimension() with a small dimension."""
        dimension = 6
        self.davidson_options.set_dimension(dimension)
        self.assertEqual(self.davidson_options.max_subspace, dimension + 1)

    def test_set_dimension_large(self):
        """Tests set_dimension() with a large dimension not affecting
            max_subspace."""
        self.davidson_options.set_dimension(60)
        self.assertEqual(self.davidson_options.max_subspace, self.max_subspace)

    def test_invalid_max_subspace(self):
        """Test for invalid max_subspace."""
        with self.assertRaises(ValueError):
            DavidsonOptions(max_subspace=1)

    def test_invalid_max_iterations(self):
        """Test for invalid max_iterations."""
        with self.assertRaises(ValueError):
            DavidsonOptions(max_iterations=0)

    def test_invalid_eps(self):
        """Test for invalid eps."""
        with self.assertRaises(ValueError):
            DavidsonOptions(eps=-1e-6)

    def test_invalid_dimension(self):
        """Test for invalid dimension."""
        with self.assertRaises(ValueError):
            self.davidson_options.set_dimension(0)


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

        self.davidson = Davidson(linear_operator=self.linear_operator,
                                 linear_operator_diagonal=self.diagonal)

        self.matrix = matrix
        self.initial_guess = numpy.eye(self.matrix.shape[0], 10)

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

        # Options default values except max_subspace.
        self.assertEqual(davidson.options.max_subspace, 11)
        self.assertAlmostEqual(davidson.options.eps, 1e-6, places=8)
        self.assertFalse(davidson.options.real_only)

    def test_with_built_in(self):
        """Compare with eigenvalues from built-in functions."""
        eigen_values, _ = numpy.linalg.eig(self.matrix)
        eigen_values = sorted(eigen_values)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values))

        # Checks for eigh() function.
        eigen_values, eigen_vectors = numpy.linalg.eigh(self.matrix)
        self.assertAlmostEqual(get_difference(self.davidson.linear_operator,
                                              eigen_values, eigen_vectors), 0)

    def test_lowest_invalid_operator(self):
        """Test for get_lowest_n() with invalid linear operator."""
        with self.assertRaises(ValueError):
            Davidson(None, numpy.eye(self.matrix.shape[0], 8))

    def test_lowest_zero_n(self):
        """Test for get_lowest_n() with invalid n_lowest."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(0)

    def test_lowest_invalid_shape(self):
        """Test for get_lowest_n() with invalid dimension for initial guess."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(
                1, numpy.ones((self.matrix.shape[0] * 2, 1), dtype=complex))

    def test_get_lowest_n_trivial_guess(self):
        """Test for get_lowest_n() with trivial initial guess."""
        with self.assertRaises(ValueError):
            self.davidson.get_lowest_n(
                1, numpy.zeros((self.matrix.shape[0], 1), dtype=complex))

    def test_get_lowest_fail(self):
        """Test for get_lowest_n() with n_lowest = 1."""
        n_lowest = 1
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=2)

        self.assertTrue(not success)
        self.assertTrue(numpy.allclose(eigen_values,
                                       numpy.array([1.41556103])))

    def test_get_lowest_with_default(self):
        """Test for get_lowest_n() with default n_lowest = 1."""
        numpy.random.seed(len(self.eigen_values))
        success, eigen_values, _ = self.davidson.get_lowest_n()

        self.assertTrue(success)
        self.assertTrue(numpy.allclose(eigen_values, self.eigen_values[:1]))

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
        self.davidson.options.max_subspace = 8
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
        self.assertTrue(
            numpy.allclose(eigen_values, self.eigen_values[:n_lowest]))

    def test_get_lowest_all(self):
        """Test for get_lowest_n() with n_lowest = 10."""
        n_lowest = 10
        initial_guess = self.initial_guess[:, :n_lowest]

        success, eigen_values, _ = self.davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=1)
        self.assertTrue(success)
        self.assertTrue(
            numpy.allclose(eigen_values, self.eigen_values[:n_lowest]))


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

    def test_get_lowest_z_real(self):
        """Test for get_lowest_n() for z with real eigenvectors only."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator('Z3') * self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)
        davidson.options.real_only = True

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
        # Real components only.
        self.assertTrue(numpy.allclose(numpy.real(eigen_vectors),
                                       eigen_vectors))

    def test_get_lowest_y_real_fail(self):
        """Test for get_lowest_n() for y with real eigenvectors only."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator('Y3') * self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)
        davidson.options.max_subspace = 11
        davidson.options.real_only = True

        n_lowest = 6
        # Guess vectors have both real and imaginary parts.
        numpy.random.seed(dimension)
        initial_guess = 1.0j * numpy.random.rand(dimension, n_lowest)
        numpy.random.seed(dimension * 2)
        initial_guess += numpy.random.rand(dimension, n_lowest)
        success, _, eigen_vectors = davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=10)

        self.assertFalse(success)

        # Not real components only.
        self.assertFalse(numpy.allclose(numpy.real(eigen_vectors),
                                        eigen_vectors))
    def test_get_lowest_y_real(self):
        """Test for get_lowest_n() for y with real eigenvectors only."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator('Y3') * self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)
        davidson.options.real_only = True

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
        # Not real components only.
        self.assertFalse(numpy.allclose(numpy.real(eigen_vectors),
                                        eigen_vectors))

    def test_get_lowest_y_complex(self):
        """Test for get_lowest_n() for y with complex eigenvectors."""
        dimension = 2 ** self.n_qubits
        qubit_operator = QubitOperator('Y3') * self.coefficient
        davidson = QubitDavidson(qubit_operator, self.n_qubits)
        davidson.options.real_only = True

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

class SparseDavidsonTest(unittest.TestCase):
    """"Tests for SparseDavidson class with sparse matrices."""

    def setUp(self):
        """Sets up all variables needed for SparseDavidson class."""
        logging.basicConfig(level=logging.INFO)
        self.dimension = 1000
        self.sparse_matrix = generate_sparse_matrix(self.dimension)
        self.davidson_options = DavidsonOptions(max_subspace=100,
                                                max_iterations=50,
                                                real_only=True)

        # Checks for built-in eigh() function.
        self.eigen_values, self.eigen_vectors = numpy.linalg.eigh(
            self.sparse_matrix)
        self.assertAlmostEqual(get_difference(
            self.sparse_matrix, self.eigen_values, self.eigen_vectors), 0)

        # Makes sure eigenvalues are sorted.
        self.eigen_values = sorted(self.eigen_values)

    def test_hermitain(self):
        """Test matrix used is Hermitian."""
        self.assertTrue(numpy.allclose(self.sparse_matrix,
                                       self.sparse_matrix.conj().T))

    def test_get_lowest_n_coo(self):
        """Test for get_lowest_n() as a coo_matrix."""
        davidson = SparseDavidson(scipy.sparse.coo_matrix(self.sparse_matrix),
                                  self.davidson_options)

        n_lowest = 2
        initial_guess = numpy.eye(self.dimension, n_lowest)
        success, eigen_values, eigen_vectors = davidson.get_lowest_n(
            n_lowest, initial_guess)

        expected_eigen_values = self.eigen_values[:n_lowest]

        self.assertTrue(success)
        self.assertLess(
            numpy.max(numpy.abs(eigen_values - expected_eigen_values)),
            self.davidson_options.eps)
        self.assertLess(
            get_difference(self.sparse_matrix, eigen_values, eigen_vectors),
            self.davidson_options.eps)

        # Real components only.
        self.assertTrue(numpy.allclose(numpy.real(eigen_vectors),
                                       eigen_vectors))

    def test_get_lowest_n_coo_complex(self):
        """Test for get_lowest_n() as a coo_matrix with real_only=False."""
        self.davidson_options.real_only = False
        davidson = SparseDavidson(scipy.sparse.coo_matrix(self.sparse_matrix),
                                  self.davidson_options)

        n_lowest = 2
        initial_guess = numpy.eye(self.dimension, n_lowest)
        success, eigen_values, eigen_vectors = davidson.get_lowest_n(
            n_lowest, initial_guess, max_iterations=30)

        expected_eigen_values = self.eigen_values[:n_lowest]

        self.assertTrue(success)
        self.assertLess(
            numpy.max(numpy.abs(eigen_values - expected_eigen_values)),
            self.davidson_options.eps)
        self.assertLess(
            get_difference(self.sparse_matrix, eigen_values, eigen_vectors),
            self.davidson_options.eps)

        # Real components only.
        self.assertTrue(numpy.allclose(numpy.real(eigen_vectors),
                                       eigen_vectors))


    def test_get_lowest_n(self):
        """Test for get_lowest_n() as a other sparse formats."""
        n_lowest = 2
        expected_eigen_values = self.eigen_values[:n_lowest]
        initial_guess = numpy.eye(self.dimension, n_lowest)

        for run_matrix in [
                scipy.sparse.bsr_matrix(self.sparse_matrix),
                scipy.sparse.csc_matrix(self.sparse_matrix),
                scipy.sparse.csr_matrix(self.sparse_matrix),
                scipy.sparse.dia_matrix(self.sparse_matrix),
                scipy.sparse.dok_matrix(self.sparse_matrix),
                scipy.sparse.lil_matrix(self.sparse_matrix),
        ]:
            davidson = SparseDavidson(run_matrix, self.davidson_options)
            success, eigen_values, eigen_vectors = davidson.get_lowest_n(
                n_lowest, initial_guess)

            self.assertTrue(success)
            self.assertLess(
                numpy.max(numpy.abs(eigen_values - expected_eigen_values)),
                self.davidson_options.eps)
            self.assertLess(
                get_difference(self.sparse_matrix, eigen_values, eigen_vectors),
                self.davidson_options.eps)

            # Real components only.
            self.assertTrue(numpy.allclose(numpy.real(eigen_vectors),
                                           eigen_vectors))


class DavidsonUtilityTest(unittest.TestCase):
    """"Tests for utility functions."""
    def test_append_random_vectors_0(self):
        """Test append_random_vectors() with too few columns."""
        vectors = numpy.zeros((10, 2), dtype=complex)
        self.assertTrue(numpy.allclose(
            append_random_vectors(vectors, 0), vectors))

    def test_append_random_vectors(self):
        """Test append_random_vectors()."""
        row = 10
        col = 2
        add = 1
        vectors = numpy.eye(row, col)
        new_vectors = append_random_vectors(vectors, add)

        # Identical for the first col columns.
        self.assertTrue(numpy.allclose(new_vectors[:, :col], vectors))

        # Orthonormal.
        self.assertTrue(numpy.allclose(
            numpy.dot(new_vectors.conj().T, new_vectors),
            numpy.eye(col + add, col + add)))

    def test_append_random_vectors_real(self):
        """Test append_random_vectors()."""
        row = 10
        col = 2
        add = 1
        vectors = numpy.eye(row, col)
        new_vectors = append_random_vectors(vectors, add, real_only=True)

        # Identical for the first col columns.
        self.assertTrue(numpy.allclose(new_vectors[:, :col], vectors))

        # Orthonormal.
        self.assertTrue(numpy.allclose(
            numpy.dot(new_vectors.conj().T, new_vectors),
            numpy.eye(col + add, col + add)))

        # Real.
        self.assertTrue(numpy.allclose(numpy.real(new_vectors), new_vectors))

    def test_append_vectors_big_col(self):
        """Test append_random_vectors() with too many failed trial."""
        row = 10
        vectors = numpy.eye(row, row)
        new_vectors = append_random_vectors(vectors, 1)

        self.assertTrue(numpy.allclose(new_vectors, vectors))

    def test_orthonormalize(self):
        """Test for orthonormalization with removing non-independent vectors."""
        sqrt_half = numpy.sqrt(0.5)
        expected_array = numpy.array([
            [sqrt_half, sqrt_half, 0],
            [sqrt_half, -sqrt_half, 0],
            [0, 0, 1],
        ])

        array = numpy.array([[1, 1, 10, 1], [1, -1, 10, 1], [0, 0, 2, 1]],
                            dtype=float)
        array[:, 0] *= sqrt_half
        array = orthonormalize(array, 1)
        self.assertTrue(numpy.allclose(array, expected_array))

    def test_orthonormalize_complex(self):
        """Test for orthonormalization with complex matrix."""
        sqrt_half = numpy.sqrt(0.5)
        expected_array = numpy.array([
            [sqrt_half * 1.0j, sqrt_half * 1.0j, 0],
            [sqrt_half * 1.0j, -sqrt_half * 1.0j, 0],
            [0, 0, 1],
        ], dtype=complex)

        array = numpy.array([[1.j, 1.j, 10], [1.j, -1.j, 10], [0, 0, 2]],
                            dtype=complex)
        array[:, 0] *= sqrt_half
        array = orthonormalize(array, 1)
        self.assertTrue(numpy.allclose(array, expected_array))
