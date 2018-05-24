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
from __future__ import absolute_import

import numpy
import numpy.linalg
import scipy
import scipy.linalg
import scipy.sparse.linalg


# Exceptions.
class DavidsonError(Exception):
    """Exceptions."""
    pass


class Davidson(object):
    """Davidson algorithm to get the n states with smallest energies."""

    def __init__(self, linear_operator, linear_operator_diagonal, eps=1e-6):
        """
        Args:
            linear_operator(scipy.sparse.linalg.LinearOperator): The linear
                operator which defines a dot function when applying on a vector.
            linear_operator_diagonal(numpy.ndarray): The linear operator's
                diagonal elements.
            eps(float): The max error for eigen vectors' elements during
                iterations.
        """
        if not isinstance(linear_operator, scipy.sparse.linalg.LinearOperator):
            raise ValueError(
                'linear_operator is not a LinearOperator: {}.'.format(type(
                    linear_operator)))

        self.linear_operator = linear_operator
        self.linear_operator_diagonal = linear_operator_diagonal
        self.eps = eps

    def get_lowest_n(self, n_lowest=1, initial_guess=None, max_iterations=300):
        """
        Returns `n` smallest eigenvalues and corresponding eigenvectors for
            linear_operator.

        Args:
            n(int): The number of states corresponding to the smallest eigenvalues
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
        if n_lowest <= 0:
            raise ValueError('n_lowest is supposed to be positive: {}.'.format(
                n_lowest))

        success = False
        num_iterations = 0
        guess_v = initial_guess
        guess_mv = None
        while (num_iterations < max_iterations and not success):
            eigen_values, eigen_vectors, max_trial_error, guess_v, guess_mv = (
                self._iterate(n_lowest, guess_v, guess_mv))

            if max_trial_error < self.eps:
                success = True
                break

            # Deals with new directions to make sure they're orthonormal.
            count_mvs = guess_mv.shape[1]
            guess_v = self._orthonormalize(guess_v, count_mvs)
            if guess_v.shape[1] <= count_mvs:
                raise ValueError(
                    'Not able to work with new directions: {} vs {}.'.format(
                        guess_v.shape, count_mvs))
            num_iterations += 1
        return success, eigen_values, eigen_vectors


    def _orthonormalize(self, vectors, num_orthonormals=1):
        """Orthonormalize vectors, so that they're all normalized and orthogoal.

        The first vector is the same to that of vectors, while vector_i is
        orthogonal to vector_j, where j < i.

        Args:
            vectors(numpy.ndarray(complex)): Input vectors to be
                orthonormalized.
            num_orthonormals(int): First `num_orthonormals` columns are already
                orthonormal, so that one doesn't need to make any changes.

        Returns:
            ortho_normals(numpy.ndarray(complex)): Output orthonormal vectors.
        """
        num_vectors = vectors.shape[1]
        if num_vectors == 0:
            raise ValueError(
                'vectors is not supposed to be empty: {}.'.format(vectors.shape))

        ortho_normals = numpy.copy(vectors)
        count_orthonormals = num_orthonormals
        # Skip unchanged ones.
        for i in range(num_orthonormals, num_vectors):
            vector_i = vectors[:, i]
            # Makes sure vector_i is orthogonal to all processed vectors.
            for j in range(i):
                vector_i -= ortho_normals[:, j] * numpy.dot(ortho_normals[:, j],
                                                            vector_i)

            # Makes sure vector_i is normalized.
            v_norm = numpy.linalg.norm(vector_i)
            if v_norm < self.eps:
                continue
            ortho_normals[:, count_orthonormals] = vector_i / v_norm
            count_orthonormals += 1
        return ortho_normals[:, :count_orthonormals]

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
            max_trial_error(float): The max elementwise error for all guess
                vectors.

            guess_v(numpy.ndarray(complex)): Cached guess eigenvectors to avoid
               recalculation for the next iterations.
            guess_mv(numpy.ndarray(complex)): Cached guess vectors which is the
                matrix product of linear_operator with guess_v.
        """
        # TODO: optimize for memory usage, so that one can limit max number of
        # guess vectors to keep.

        if guess_mv is None:
            guess_v = scipy.linalg.orth(guess_v)
            guess_mv = self.linear_operator * guess_v
        origonal_dimension, dimension = guess_v.shape

        # Note that getting guess_mv is the most expensive step.
        if guess_mv.shape[1] < dimension:
            guess_mv = numpy.hstack([guess_mv, self.linear_operator *
                                     guess_v[:, guess_mv.shape[1] : dimension]])
        guess_vmv = numpy.dot(guess_v.conj().T, guess_mv)

        # Gets new set of eigenvalues and eigenvectors in the vmv space, with a
        # smaller dimension which is the number of vectors in guess_v.
        #
        # Note that we don't get the eigenvectors directly, instead we only get
        # a transformation based on the raw vectors, so that mv don't need to be
        # recalculated.
        trial_lambda, trial_transformation = numpy.linalg.eig(guess_vmv)

        # Sorts eigenvalues in ascending order.
        sorted_index = list(reversed(trial_lambda.argsort()[::-1]))
        trial_lambda = trial_lambda[sorted_index]
        trial_transformation = trial_transformation[:, sorted_index]

        # Estimates errors based on diagonalization in the smaller space.
        trial_v = numpy.dot(guess_v, trial_transformation)
        trial_mv = numpy.dot(guess_mv, trial_transformation)
        trial_error = trial_mv - numpy.dot(trial_v, numpy.diag(trial_lambda))

        max_trial_error = 0
        # Adds new guess vectors for the next iteration for the first n_lowest
        # directions.
        new_directions = []
        for i in range(n_lowest):
            new_direction = trial_error[:, i]
            i_trial_error = numpy.max(numpy.abs(new_direction))
            max_trial_error = max(max_trial_error, i_trial_error)

            if i_trial_error < self.eps:
                # Already converged for this eigenvector, no contribution to
                # search for new directions.
                continue

            for j in range(origonal_dimension):
                diff_lambda = self.linear_operator_diagonal[j] - trial_lambda[i]
                if numpy.abs(diff_lambda) > self.eps:
                    new_direction[j] /= diff_lambda
                else:
                    # Makes sure error vectors are bounded.
                    new_direction[j] /= self.eps
            new_directions.append(new_direction)

        if new_directions:
            guess_v = numpy.hstack([guess_v, numpy.stack(new_directions).T])
        if len(trial_lambda) > n_lowest:
            trial_lambda = trial_lambda[:n_lowest]
            trial_v = trial_v[:, :n_lowest]
        return trial_lambda, trial_v, max_trial_error, guess_v, guess_mv
