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

"""This module is to find lowest eigenvalues with Davidson algorithm."""

import logging
import warnings

import numpy
import numpy.linalg
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from openfermion.utils._sparse_tools import get_linear_qubit_operator_diagonal
from openfermion.utils._linear_qubit_operator import \
    generate_linear_qubit_operator


class DavidsonError(Exception):
    """Exceptions."""
    pass


class DavidsonOptions(object):
    """Davidson algorithm iteration options."""

    def __init__(self, max_subspace=100, max_iterations=300, eps=1e-6,
                 real_only=False):
        """
        Args:
            max_subspace(int): Max number of vectors in the auxiliary subspace.
            max_iterations(int): Max number of iterations.
            eps(float): The max error for eigen vector error's elements during
                iterations: linear_operator * v - v * lambda.
            real_only(bool): Desired eigenvectors are real only or not. When one
                specifies the real_only to be true but it only has complex ones,
                no matter it converges or not, the returned vectors will be
                complex.
        """
        if max_subspace <= 2 or max_iterations <= 0 or eps <= 0:
            raise ValueError('Invalid values for max_subspace, max_iterations '
                             'and/ or eps: ({}, {}, {}).'.format(
                                 max_subspace, max_iterations, eps))

        self.max_subspace = max_subspace
        self.max_iterations = max_iterations
        self.eps = eps
        self.real_only = real_only

    def set_dimension(self, dimension):
        """
        Args:
            dimension(int): Dimension of the matrix, which sets a upper limit on
                the work space.
        """
        if dimension <= 0:
            raise ValueError('Invalid dimension: {}).'.format(dimension))
        self.max_subspace = min(self.max_subspace, dimension + 1)


class Davidson(object):
    """Davidson algorithm to get the n states with smallest eigenvalues."""

    def __init__(self, linear_operator, linear_operator_diagonal, options=None):
        """
        Args:
            linear_operator(scipy.sparse.linalg.LinearOperator): The linear
                operator which defines a dot function when applying on a vector.
            linear_operator_diagonal(numpy.ndarray): The linear operator's
                diagonal elements.
            options(DavidsonOptions): Iteration options.
        """
        if options is None:
            options = DavidsonOptions()

        if not isinstance(linear_operator,
                          (scipy.sparse.linalg.LinearOperator,
                           scipy.sparse.spmatrix)):
            raise ValueError(
                'linear_operator is not a LinearOperator: {}.'.format(type(
                    linear_operator)))

        self.linear_operator = linear_operator
        self.linear_operator_diagonal = linear_operator_diagonal

        self.options = options
        self.options.set_dimension(len(linear_operator_diagonal))

    def get_lowest_n(self, n_lowest=1, initial_guess=None, max_iterations=None):
        """
        Returns `n` smallest eigenvalues and corresponding eigenvectors for
            linear_operator.

        Args:
            n(int):
                The number of states corresponding to the smallest eigenvalues
                and associated eigenvectors for the linear_operator.
            initial_guess(numpy.ndarray[complex]): Initial guess of eigenvectors
                associated with the `n` smallest eigenvalues.
            max_iterations(int): Max number of iterations when not converging.

        Returns:
            success(bool): Indicates whether it converged, i.e. max elementwise
                error is smaller than eps.
            eigen_values(numpy.ndarray[complex]): The smallest n eigenvalues.
            eigen_vectors(numpy.ndarray[complex]): The smallest n eigenvectors
                  corresponding with those eigen values.
        """
        # Goes through a few checks and preprocessing before iterative
        # diagonalization.

        # 1. Checks for number of states desired, should be in the range of
        # [0, max_subspace).
        if n_lowest <= 0 or n_lowest >= self.options.max_subspace:
            raise ValueError('n_lowest {} is supposed to be in [1, {}).'.format(
                n_lowest, self.options.max_subspace))

        # 2. Checks for initial guess vectors' dimension is the same to that of
        # the operator.
        if initial_guess is None:
            initial_guess = generate_random_vectors(
                len(self.linear_operator_diagonal), n_lowest,
                real_only=self.options.real_only)
        if initial_guess.shape[0] != len(self.linear_operator_diagonal):
            raise ValueError('Guess vectors have a different dimension with '
                             'linear opearator diagonal elements: {} != {}.'
                             .format(initial_guess.shape[1],
                                     len(self.linear_operator_diagonal)))

        # 3. Makes sure real guess vector if real_only is specified.
        if self.options.real_only:
            if not numpy.allclose(numpy.real(initial_guess), initial_guess):
                warnings.warn('Initial guess is not real only!', RuntimeWarning)
                initial_guess = numpy.real(initial_guess)

        # 4. Checks for non-trivial (non-zero) initial guesses.
        if numpy.max(numpy.abs(initial_guess)) < self.options.eps:
            raise ValueError('Guess vectors are all zero! {}'.format(
                initial_guess.shape))
        initial_guess = scipy.linalg.orth(initial_guess)

        # 5. Makes sure number of initial guess vector is at least n_lowest.
        if initial_guess.shape[1] < n_lowest:
            initial_guess = append_random_vectors(
                initial_guess, n_lowest - initial_guess.shape[1],
                real_only=self.options.real_only)

        success = False
        num_iterations = 0
        guess_v = initial_guess
        guess_mv = None
        max_iterations = max_iterations or self.options.max_iterations
        while (num_iterations < max_iterations and not success):
            (eigen_values, eigen_vectors, mat_eigen_vectors, max_trial_error,
             guess_v, guess_mv) = self._iterate(n_lowest, guess_v, guess_mv)
            logging.info("Eigenvalues for iteration %d: %s, error is %f.",
                         num_iterations, eigen_values, max_trial_error)

            if max_trial_error < self.options.eps:
                success = True
                break

            # Make sure it keeps real components only.
            if self.options.real_only:
                guess_v = numpy.real(guess_v)

            # Deals with new directions to make sure they're orthonormal.
            # Also makes sure there're new directions added for the next
            # iteration, if not, add n_lowest random vectors.
            count_mvs = guess_mv.shape[1]
            guess_v = orthonormalize(guess_v, count_mvs, self.options.eps)
            if guess_v.shape[1] <= count_mvs:
                guess_v = append_random_vectors(
                    guess_v, n_lowest, real_only=self.options.real_only)


            # Limits number of vectors to self.options.max_subspace, in this
            # case, keep the following:
            # 1) first n_lowest eigen_vectors;
            # 2) first n_lowest matrix multiplication result for eigen_vectors;
            #
            # 3) new search directions which will be used for improvement for
            #    the next iteration.
            if guess_v.shape[1] >= self.options.max_subspace:
                guess_v = numpy.hstack([
                    eigen_vectors,
                    guess_v[:, count_mvs:],
                ])
                guess_mv = mat_eigen_vectors

                if self.options.real_only:
                    if (not numpy.allclose(numpy.real(guess_v), guess_v) or
                            not numpy.allclose(numpy.real(guess_mv), guess_mv)):
                        # Forces recalculation for matrix multiplication with
                        # vectors.
                        guess_mv = None

            num_iterations += 1

        if (self.options.real_only and
                not numpy.allclose(numpy.real(eigen_vectors), eigen_vectors)):
            warnings.warn('Unable to get real only eigenvectors, return '
                          'complex vectors instead with success state {}.'
                          .format(success), RuntimeWarning)
        return success, eigen_values, eigen_vectors

    def _iterate(self, n_lowest, guess_v, guess_mv=None):
        """One iteration with guess vectors.

        Args:
            n_lowest(int): The first n_lowest number of eigenvalues and
                eigenvectors one is interested in.
            guess_v(numpy.ndarray(complex)): Guess eigenvectors associated with
                the smallest eigenvalues.
            guess_mv(numpy.ndarray(complex)): Matrix applied on guess_v,
                therefore they should have the same dimension.

        Returns:
            trial_lambda(numpy.ndarray(float)): The minimal eigenvalues based on
                guess eigenvectors.
            trial_v(numpy.ndarray(complex)): New guess eigenvectors.
            trial_mv(numpy.ndarray(complex)): New guess eigenvectors' matrix
                multiplication result.
            max_trial_error(float): The max elementwise error for all guess
                vectors.

            guess_v(numpy.ndarray(complex)): Cached guess eigenvectors to avoid
               recalculation for the next iterations.
            guess_mv(numpy.ndarray(complex)): Cached guess vectors which is the
                matrix product of linear_operator with guess_v.
        """
        if guess_mv is None:
            guess_mv = self.linear_operator.dot(guess_v)
        dimension = guess_v.shape[1]

        # Note that getting guess_mv is the most expensive step.
        if guess_mv.shape[1] < dimension:
            guess_mv = numpy.hstack([guess_mv, self.linear_operator.dot(
                guess_v[:, guess_mv.shape[1] : dimension])])
        guess_vmv = numpy.dot(guess_v.conj().T, guess_mv)

        # Gets new set of eigenvalues and eigenvectors in the vmv space, with a
        # smaller dimension which is the number of vectors in guess_v.
        #
        # Note that we don't get the eigenvectors directly, instead we only get
        # a transformation based on the raw vectors, so that mv don't need to be
        # recalculated.
        trial_lambda, trial_transformation = numpy.linalg.eigh(guess_vmv)

        # Sorts eigenvalues in ascending order.
        sorted_index = list(reversed(trial_lambda.argsort()[::-1]))
        trial_lambda = trial_lambda[sorted_index]
        trial_transformation = trial_transformation[:, sorted_index]

        if len(trial_lambda) > n_lowest:
            trial_lambda = trial_lambda[:n_lowest]
            trial_transformation = trial_transformation[:, :n_lowest]

        # Estimates errors based on diagonalization in the smaller space.
        trial_v = numpy.dot(guess_v, trial_transformation)
        trial_mv = numpy.dot(guess_mv, trial_transformation)
        trial_error = trial_mv - trial_v * trial_lambda

        new_directions, max_trial_error = self._get_new_directions(
            trial_error, trial_lambda, trial_v)
        if new_directions:
            guess_v = numpy.hstack([guess_v, numpy.stack(new_directions).T])
        return (trial_lambda, trial_v, trial_mv, max_trial_error,
                guess_v, guess_mv)

    def _get_new_directions(self, error_v, trial_lambda, trial_v):
        """Gets new directions from error vectors.

        Args:
            error_v(numpy.ndarray(complex)): Error vectors from the guess
                eigenvalues and associated eigenvectors.
            trial_lambda(numpy.ndarray(float)): The n_lowest minimal guess
                eigenvalues.
            trial_v(numpy.ndarray(complex)): Guess eigenvectors associated with
                trial_lambda.

        Returns:
            new_directions(numpy.ndarray(complex)): New directions for searching
                for real eigenvalues and eigenvectors.
            max_trial_error(float): The max elementwise error for all guess
                vectors.
        """
        n_lowest = error_v.shape[1]

        max_trial_error = 0
        # Adds new guess vectors for the next iteration for the first n_lowest
        # directions.
        origonal_dimension = error_v.shape[0]

        new_directions = []
        for i in range(n_lowest):
            current_error_v = error_v[:, i]

            if numpy.max(numpy.abs(current_error_v)) < self.options.eps:
                # Already converged for this eigenvector, no contribution to
                # search for new directions.
                continue

            max_trial_error = max(max_trial_error,
                                  numpy.linalg.norm(current_error_v))
            diagonal_inverse = numpy.ones(origonal_dimension)
            for j in range(origonal_dimension):
                # Makes sure error vectors are bounded.
                diff_lambda = self.linear_operator_diagonal[j] - trial_lambda[i]
                if numpy.abs(diff_lambda) > self.options.eps:
                    diagonal_inverse[j] /= diff_lambda
                else:
                    diagonal_inverse[j] /= self.options.eps
            diagonal_inverse_error = diagonal_inverse * current_error_v
            diagonal_inverse_trial = diagonal_inverse * trial_v[:, i]
            new_direction = -current_error_v + (trial_v[:, i] * numpy.dot(
                trial_v[:, i].conj(), diagonal_inverse_error) / numpy.dot(
                    trial_v[:, i].conj(), diagonal_inverse_trial))

            new_directions.append(new_direction)
        return new_directions, max_trial_error


class QubitDavidson(Davidson):
    """Davidson algorithm applied to a QubitOperator."""

    def __init__(self, qubit_operator, n_qubits=None, options=None):
        """
        Args:
            qubit_operator(QubitOperator): A qubit operator which is a linear
                operator as well.
            n_qubits(int): Number of qubits.
            options(DavidsonOptions): Iteration options.
        """
        super(QubitDavidson, self).__init__(
            generate_linear_qubit_operator(qubit_operator, n_qubits, options),
            get_linear_qubit_operator_diagonal(qubit_operator, n_qubits),
            options=options)


class SparseDavidson(Davidson):
    """Davidson algorithm for a sparse matrix."""

    def __init__(self, sparse_matrix, options=None):
        """
        Args:
            sparse_matrix(scipy.sparse.spmatrix): A sparse matrix in scipy.
            options(DavidsonOptions): Iteration options.
        """
        super(SparseDavidson, self).__init__(
            sparse_matrix, sparse_matrix.diagonal(), options=options)


def generate_random_vectors(row, col, real_only=False):
    """Generates orthonormal random vectors with col columns.

    Args:
        row(int): Number of rows for the vectors.
        col(int): Number of columns for the vectors.
        real_only(bool): Real vectors or complex ones.

    Returns:
        random_vectors(numpy.ndarray(complex)): Orthonormal random vectors.
    """
    random_vectors = numpy.random.rand(row, col)
    if not real_only:
        random_vectors = random_vectors + numpy.random.rand(row, col) * 1.0j
    random_vectors = scipy.linalg.orth(random_vectors)
    return random_vectors


def append_random_vectors(vectors, col, max_trial=3, real_only=False):
    """Appends exactly col orthonormal random vectors for vectors.

    Assumes vectors is already orthonormal.

    Args:
        vectors(numpy.ndarray(complex)): Orthonormal original vectors to be
            appended.
        col(int): Number of columns to be appended.
        real_only(bool): Real vectors or complex ones.

    Returns:
        vectors(numpy.ndarray(complex)): Orthonormal vectors with n columns.
    """
    if col <= 0:
        return vectors

    vector_columns = vectors.shape[1]
    total_columns = min(vector_columns + col, vectors.shape[0] + 1)

    num_trial = 0
    while vector_columns < total_columns:
        num_trial += 1

        vectors = numpy.hstack([vectors, generate_random_vectors(
            vectors.shape[0], total_columns - vector_columns, real_only)])
        vectors = orthonormalize(vectors, vector_columns)

        # Checks whether there are any new vectors added successfully.
        if vectors.shape[1] == vector_columns:
            if num_trial > max_trial:
                warnings.warn('Unable to generate specified number of random '
                              'vectors {}: returning {} in total.'.format(
                                  col, vector_columns), RuntimeWarning)
                break
        else:
            num_trial = 1
            vector_columns = vectors.shape[1]
    return vectors

def orthonormalize(vectors, num_orthonormals=1, eps=1e-6):
    """Orthonormalize vectors, so that they're all normalized and orthogoal.

    The first vector is the same to that of vectors, while vector_i is
    orthogonal to vector_j, where j < i.

    Args:
        vectors(numpy.ndarray(complex)): Input vectors to be
            orthonormalized.
        num_orthonormals(int): First `num_orthonormals` columns are already
            orthonormal, so that one doesn't need to make any changes.
        eps(float): criterion of elements' max absolute value for zero vectors.

    Returns:
        ortho_normals(numpy.ndarray(complex)): Output orthonormal vectors.
    """
    ortho_normals = vectors
    count_orthonormals = num_orthonormals
    # Skip unchanged ones.
    for i in range(num_orthonormals, vectors.shape[1]):
        vector_i = vectors[:, i]
        # Makes sure vector_i is orthogonal to all processed vectors.
        for j in range(i):
            vector_i -= ortho_normals[:, j] * numpy.dot(
                ortho_normals[:, j].conj(), vector_i)

        # Makes sure vector_i is normalized.
        if numpy.max(numpy.abs(vector_i)) < eps:
            continue
        ortho_normals[:, count_orthonormals] = (vector_i /
                                                numpy.linalg.norm(vector_i))
        count_orthonormals += 1
    return ortho_normals[:, :count_orthonormals]
