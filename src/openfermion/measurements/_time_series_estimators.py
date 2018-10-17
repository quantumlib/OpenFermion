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

"""
Classes for time series estimators for quantum phase estimation.
Estimators described in arXiv:1809.09697, in particular in Sec.II.A.
"""

import numpy as np
import scipy as sp


class TimeSeriesEstimator(object):
    """
    Time-series estimator: a QPE estimator based on the analysis
    of input data as a time series. Takes in data from repeated
    QPE experiments, and returns an estimate of the underlying
    eigenvalues.

    Each QPE experiment consists of multiple rounds, and each round
    consists of the application of U^k conditional on an ancilla qubit,
    an additional rotation by e^{i*beta*sigma_z} on the ancilla qubit,
    and measurement of the ancilla qubit in the X-basis returning a
    measurement m. This is done on a starting state sum_j a_j|phi_j>,
    where |phi_j> is an eigenstate of U with eigenvalue phi_j.

    The simplest method to use this class assumes the above data for
    each experiment is stored in a list of one dictionary for
    each round, using the labels
        'num_rotations': k
        'final_rotation': beta
        'measurement': m
    In particular, this class requires that all experiments be one round
    long (this data format allows consistency with the Bayesian and
    multi-round time series estimators).
    Then, max_experiment_length must be set to max(k) (also note that this
    estimator requires a minimum of one experiment for each k=1,...,max(k)).
    num_freqs can be set to the number of expected phi_j in the problem;
    if num_freqs is less than that only the first few (ordered by |a_j|^2)
    will be estimated, if num_freqs is greater then the remaining frequencies
    will be garbage.

    Usage consists of the following commands:

    >> estimator = TimeSeriesEstimator(max_experiment_length, num_freqs_max)
    >> for experiment in experiment_list:
    >>     estimator.update(experiment)
    >> angles = estimator.estimate()

    angles will now contain the best estimates for phi_j. To obtain
    estimates for |a_j|^2, one can set the flag return_amplitudes=True in
    the call to estimator.estimate().
    """
    def __init__(self,
                 max_experiment_length,
                 num_freqs_max,
                 depolarizing_noise=False,
                 singular_values_cutoff=1e-12,
                 store_history=False,
                 automatic_average=False):
        """
        Args:
            max_experiment_length (int): the largest number (K) of unitary operations
                to be performed during the experiment. At least two experiments
                must be made at each k=1,...,K, with two different values
                of the final rotation beta (not separated by pi), in order to
                obtain an estimate from this estimator.
            num_freqs_max (int): the maximum number of eigenvalues phi_j to
                estimate. Must be less than max_experiment_length without
                depolarizing noise and less than max_experiment_length/2 with.
            depolarizing_noise (bool): whether or not to account for depolarizing
                noise in the model.
            singular_values_cutoff (float): the value below which singular values
                are treated as zero (for counting the number of eigenvalues
                present in the problem).
            store_history (bool): whether to save the estimated eigenvalues whenever
                they are estimated.
            automatic_average (bool): whether to reestimate the eigenvalues whenever
                new data is added.
        """
        if num_freqs_max * (1+depolarizing_noise) > max_experiment_length:
            raise ValueError('''Maximum experiment length of {}
                             is too short to estimate {} frequencies'''.format(
                                 max_experiment_length, num_freqs_max))

        self._max_experiment_length = max_experiment_length
        self._num_freqs_max = num_freqs_max
        self._depolarizing_noise = depolarizing_noise
        self._singular_values_cutoff = singular_values_cutoff
        self._automatic_average = automatic_average
        self._store_history = store_history

        self.reset()

    def reset(self):
        """
        Resets estimator for new experiment
        """
        self._re_abs = np.zeros(self._max_experiment_length+1)
        self._im_abs = np.zeros(self._max_experiment_length+1)
        self._re_est = np.zeros(self._max_experiment_length+1)
        self._im_est = np.zeros(self._max_experiment_length+1)
        self._function_estimate = np.zeros(self._max_experiment_length+1)
        self._function_estimate[0] = 1  # we get this point for free.
        self._num_freqs = self._num_freqs_max
        self._update_flag = True

        if self._store_history:
            self.average_history = []

    def update(self, experiment_data):
        """
        Takes new experiment data in dictionary format. Note - storing data
        in this format is significantly worse than tallying results and
        inserting with update_mass (provided here to work similarly
        to the Bayesian estimators).

        Args:
            experiment_data (list of dictionaries): information
                about each round in the experiment. Each dictionary
                must contain the entries 'final_rotation', 'measurement',
                and 'num_rotations'. For the single-round estimator,
                experiment_data must be of length 1.

        Returns:
            (bool) success of update (always True for this estimator)
        """
        if len(experiment_data) > 1:
            raise NotImplementedError('''Estimator not implemented for
                                      multi-round experiments''')
        round_data = experiment_data[0]
        angle = round_data['final_rotation'] +\
            np.pi * round_data['measurement']
        k = round_data['num_rotations']

        self._re_abs[k] += np.cos(angle)**2
        self._im_abs[k] += np.sin(angle)**2
        self._re_est[k] += np.cos(angle)
        self._im_est[k] -= np.sin(angle)

        self._update_flag = False

        if self._automatic_average is True:
            self.estimate()

        return True

    def update_mass(self, f_data):
        """
        Takes data to update g(k) from multiple experiments.
        The preferable form to input data, but requires that
        experiments take a final rotation of either 0 or pi/2.

        Args:
            f_data (numpy array): f_data[k,a,m] = number of m=0,1
                measurements following k controled unitary rotations
                and a final rotation of a=0,1*(pi/2).

        Returns:
            (bool): success of update (always True for this estimator)
        """
        self._re_abs += f_data[:, 0, 0] + f_data[:, 0, 1]
        self._re_est += f_data[:, 0, 0] - f_data[:, 0, 1]
        self._im_abs += f_data[:, 1, 0] + f_data[:, 1, 1]
        self._im_est -= f_data[:, 1, 0] - f_data[:, 1, 1]

        self._update_flag = False

        if self._automatic_average is True:
            self.estimate()

        return True

    def _update_function(self):
        """
        Calculates the estimate of the function from the individual counts
        """
        function_real = self._re_est/self._re_abs
        function_imag = self._im_est/self._im_abs

        self._function_estimate = function_real + 1j * function_imag
        self._function_estimate[0] = 1

        if self._depolarizing_noise is False:
            self._function_estimate = np.hstack(
                [self._function_estimate[:0:-1].conj(),
                 self._function_estimate])

        self._update_flag = True

    def _make_hankel_matrices(self):
        """
        Makes the two Hankel matrices
        """
        if self._update_flag is False:
            self._update_function()

        hankel0 = sp.linalg.hankel(
            c=self._function_estimate[:self._num_freqs],
            r=self._function_estimate[self._num_freqs-1:-1])

        hankel1 = sp.linalg.hankel(
            c=self._function_estimate[1:self._num_freqs+1],
            r=self._function_estimate[self._num_freqs:])

        self._hankel0 = hankel0.T
        self._hankel1 = hankel1.T

    def _make_translation_matrix(self):
        """
        Makes the (transposed) translation matrix from the two Hankel matrices.
        """
        self._shift_matrix = sp.linalg.lstsq(self._hankel0, self._hankel1)[0]

    def _get_num_frequencies(self):
        """
        Finds the number of singular values above cutoff in
        the Hankel matrices - this corresponds to our best estimate
        of the number of frequencies in the problem. If the Hankel
        matrices are full-rank, then we possibly have too many
        frequencies to estimate, and low-amplitude estimates
        shouldn't be trusted (in general we advise not trusting
        any results with amplitude estimate below 1/max_experiment_length)
        """
        self._num_freqs = self._num_freqs_max
        if self._singular_values_cutoff > 0:
            self._make_hankel_matrices()
            singular_values = np.linalg.svd(self._hankel0)[1]
            self._num_freqs = sum(singular_values >
                                  self._singular_values_cutoff)
            assert self._num_freqs <= self._num_freqs_max

    def _get_amplitudes(self):
        """
        Obtains the amplitudes of the problem by a least squares fit
        of the target function to the estimated function.
        """
        if self._depolarizing_noise is True:
            self._generation_matrix = np.array([
                [np.exp(1j*k*angle) for angle in self.angles]
                for k in range(0, self._max_experiment_length+1)])
        else:
            self._generation_matrix = np.array([
                [np.exp(1j*k*angle) for angle in self.angles]
                for k in range(-self._max_experiment_length,
                               self._max_experiment_length+1)])

        self.amplitudes = sp.linalg.lstsq(self._generation_matrix,
                                          self._function_estimate)[0]

    def estimate(self, return_amplitudes=False):
        """
        Estimates all frequencies as required.

        Args:
            return_amplitudes (bool): whether to return both the corresponding
                amplitude estimates along with the angle estimates.
        Returns
            angles (array): the estimated target frequencies
            amplitudes(optional, array): The corresponding amplitude estimates
        """
        self._get_num_frequencies()
        self._make_hankel_matrices()
        self._make_translation_matrix()
        self.angles = np.angle(np.linalg.eigvals(self._shift_matrix.T))
        self._get_amplitudes()
        self.amplitudes, self.angles = zip(
            *sorted(zip(self.amplitudes, self.angles),
                    key=lambda x: np.real(x[0]), reverse=True))

        if self._store_history:
            self.average_history.append(np.array(self.angles))

        if return_amplitudes:
            return self.angles, self.amplitudes
        return self.angles


class TimeSeriesMultiRoundEstimator(TimeSeriesEstimator):
    """
    Time series estimator for multi-round QPE experiments.
    Generates g(k) from a specific set of multi-round experiments
    and then performs a time-series analysis as in the single-round
    case, but weighted by the calculated variance in individual
    points.

    Requires a very specific set of multi-round experiments
    (N/2 rounds with 0 final rotation, N/2 rounds with pi/2
    final rotation). Unfortunately this doesn't seem very easily
    generalizeable.
    """
    def __init__(self, **kwargs):

        super(TimeSeriesMultiRoundEstimator, self).__init__(
            depolarizing_noise=True, **kwargs)

        self._choose_vec = np.array([
            sp.special.comb(self._max_experiment_length, k)
            for k in range(self._max_experiment_length+1)])

    def reset(self):
        """
        Resets estimator for new experiment
        """
        self._hamming_mat = np.zeros([self._max_experiment_length//2+1,
                                      self._max_experiment_length//2+1])
        self._function_estimate = np.zeros(self._max_experiment_length + 1)
        self._num_experiments = 0
        self._num_freqs = self._num_freqs_max
        self._update_flag = True
        if self._store_history:
            self.average_history = []

    def update(self, experiment_data):
        """
        Adds data in json format.

        Args:
            experiment_data (list of dictionaries):
                arguments and measurements of each round.
                arguments must contain 'measurement', 'final_rotation'.
                and must have an equal number of 0 and pi/2
                final rotations (restricted by our experiment design).

        Returns:
            (bool) success of update (always True for this estimator)
        """
        if len(experiment_data) != self._max_experiment_length:
            raise NotImplementedError('''Estimator only accepts
                                    experiments of length {}'''.format(
                                        self._max_experiment_length))
        if len([round_data for round_data in experiment_data
                if round_data['final_rotation'] == 0]) !=\
                self._max_experiment_length//2:
            raise NotImplementedError('''Half of the experiments
                                    must have final rotation of 0.''')

        hw0 = sum([round_data['measurement']
                   for round_data in experiment_data
                   if round_data['final_rotation'] == 0])
        hw1 = sum([round_data['measurement']
                   for round_data in experiment_data
                   if round_data['final_rotation'] != 0])

        self._hamming_mat[hw0, hw1] += 1
        self._num_experiments += 1
        self._update_flag = False

        if self._automatic_average is True:
            self.estimate()

        return True

    def update_mass(self, hamming_mat):
        """
        Takes data in the form hamming_mat[hw0,hw1] = number
        of observed data points with hw0 measurements of '1' after a 0
        rotation and hw1 measurements of '1' after a pi/2 rotation

        Args:
            hamming_mat (numpy array): register of multiple measurements as
                described above.

        Returns:
            (bool): success of update (always True for this estimator)
        """
        self._hamming_mat += hamming_mat
        self._num_experiments += np.sum(hamming_mat)
        self._update_flag = False

        if self._automatic_average is True:
            self.estimate()

        return True

    def _gen_comb(self, N, n):
        """
        A generalized combinatorial function that returns zero
        outside of the usual bounds (required for our summation).

        Args:
            N, n (int): combinatorial input
        Returns:
            (int): N choose n.
        """
        if n > N or n < 0:
            return 0
        return sp.special.comb(N, n)

    def _p_function(self, R, l, m):
        """
        Function to calculate the recurring combinatorial factors
        used in our chosen generation of g(k). See paper for details of input
        and output.

        Args:
            R,l,m (int): see arXiv:1809.09697 for details
        Returns:
            (int): calculation of subterms in eq.25 of arXiv:1809.09697.
        """
        res = 0
        for p in range(0, l//2 + 1):
            res += self._gen_comb(m, 2*p) * self._gen_comb(R//2 - m, l - 2*p)
        res = 2 * res / self._gen_comb(R//2, l) - 1
        return res

    def _update_function(self):
        """
        Calculates the estimate of the function g(k) from the input counts,
        following eq.25 of arXiv:1809.09697.
        """
        hamming_mat = self._hamming_mat / self._num_experiments
        hamming_mat_var =\
            (hamming_mat + 1/self._num_experiments) *\
            (1 - hamming_mat + 1/self._num_experiments) /\
            self._num_experiments

        R = self._max_experiment_length
        self._function_estimate = np.zeros(R//2+1, dtype=complex)
        function_var = np.zeros(R//2+1)
        for k in range(R//2+1):
            for n in range(R//2+1):
                for m in range(R//2+1):
                    factor = 0
                    for l in range(k+1):
                        factor += (-1j)**(k-l) * self._gen_comb(k, l) *\
                            self._p_function(R, l, m) *\
                            self._p_function(R, k-l, n)

                    self._function_estimate[k] += factor * hamming_mat[m, n]
                    function_var[k] += np.abs(factor)**2 *\
                        hamming_mat_var[m, n]

        self._function_std = np.sqrt(function_var)

    def _make_hankel_matrices(self):
        """
        Makes the two Hankel matrices and stores them,
        along with the standard deviation in the target Hankel matrix.
        """
        if self._update_flag is False:
            self._update_function()

        hankel0 = sp.linalg.hankel(
            c=self._function_estimate[:self._num_freqs],
            r=self._function_estimate[self._num_freqs-1:-1])

        hankel1 = sp.linalg.hankel(
            c=self._function_estimate[1:self._num_freqs+1],
            r=self._function_estimate[self._num_freqs:])

        hankel1_std = sp.linalg.hankel(
            c=self._function_std[1:self._num_freqs+1],
            r=self._function_std[self._num_freqs:])

        self._hankel0 = hankel0.T
        self._hankel1 = hankel1.T
        self._hankel1_std = hankel1_std.T

    def _make_translation_matrix(self):
        """
        Makes the (transposed) translation matrix from the two Hankel matrices.
        Weights according to the standard deviation of hankel1.
        """
        shift_columns = []
        for j in range(self._num_freqs):
            w_mat = np.diag(1/self._hankel1_std[:, j])
            new_column = sp.linalg.lstsq(
                np.dot(w_mat, self._hankel0),
                np.dot(w_mat, self._hankel1[:, j]))[0]
            shift_columns.append(new_column)
        self._shift_matrix = np.array(shift_columns)

    def _get_amplitudes(self):
        """
        Obtains the amplitudes of the problem by a least squares fit
        of the target function to the estimated function.
        """
        self._generation_matrix = np.array([
            [np.exp(1j*k*angle) for angle in self.angles]
            for k in range(0, self._max_experiment_length//2+1)])

        self.amplitudes = sp.linalg.lstsq(self._generation_matrix,
                                          self._function_estimate)[0]
