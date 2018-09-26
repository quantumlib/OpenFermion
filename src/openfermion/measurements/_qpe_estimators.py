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

'''
Classes for estimators for quantum phase estimation. Estimators described
in arXiv:XXXX.XXXX
'''

import numpy as np
import scipy as sp
from numpy import cos, sin, pi
from scipy import sparse, optimize
import warnings
from openfermion.config import EQ_TOLERANCE


class ProbabilityDist:
    '''
    Stores a estimate of multiple phases
    $P(\phi_j)=p_0+\sum_{r=0}^N (p_{2j}cos(r*\phi_j)+p_{2j-1}sin(r*\phi_j))$
    and the probability $A_j=|a_j|^2$, with $a_j$ the amplitude of the
    eigenstate $|\phi_j\>$ in the starting state of the experiment.
    '''

    def __init__(self,
                 amplitude_guess=[1],
                 amplitude_vars=[[1]],
                 num_vectors=1,
                 num_freqs=1000,
                 max_n=1,
                 vector_guess=None):

        '''
        Args:
            amplitude_guess: estimates for amplitudes of different
                eigenstates in the initial state ($A_j$)
            amplitude_vars: variance on the amplitude estimates
            num_vectors: number of phases $\phi_j$ to estimate
            num_freqs: number of frequencies in Fourier representation.
            max_n: maximum number of unitary evolutions before measurement.
                dictates how many matrices for updates are made and stored.
            vector_guess: a prior estimate of the phases $\phi_j$. If none,
                assumes flat.
        '''

        # Store size data
        self._num_vectors = num_vectors
        self._num_freqs = num_freqs

        # Store prior information
        if vector_guess is not None and (
                num_vectors != vector_guess.shape[1]
                or 2*num_freqs+1 != vector_guess.shape[0]):
            raise ValueError('''Prior distribution of phases has shape {},
                required shape is {}'''.format(
                    vector_guess.shape, (2*num_freqs+1, num_vectors)))

        self._vector_guess = vector_guess

        if num_vectors != len(amplitude_guess):
            raise ValueError('''I need {} amplitudes, but you have
                given me {}.'''.format(num_vectors, len(amplitude_guess)))

        self._amplitude_guess = amplitude_guess
        self._amplitude_vars = amplitude_vars

        if any([self._amplitude_guess[j] -
                self._amplitude_guess[j+1] < EQ_TOLERANCE
                for j in range(num_vectors - 1)]):
            raise ValueError('''Starting amplitudes must be different.''')

        # Store max_n
        self._max_n = max_n

        # Create matrices for multiplication
        self._make_matrices()
        self._make_projector()
        self.reset()

    def reset(self):

        '''
        Resets the distribution.
        '''

        if self._vector_guess is not None:
            self._fourier_vectors = np.array(self._vector_guess)
        else:
            # Initialize probability distributions
            self._fourier_vectors = np.zeros((self._num_freqs*2+1,
                                             self._num_vectors))
            for j in range(self._num_vectors):
                self._fourier_vectors[0, j] = 1

        # Initialize p_vec with some information to break the symmetry
        self._p_vecs = [np.array(self._amplitude_guess)]
        self._amplitude_estimates = np.array(self._amplitude_guess)
        self._inv_covar_mat = np.linalg.inv(np.array(self._amplitude_vars))

    def get_real_dist(self, x_vec=np.linspace(-pi, pi, 101)):
        '''
        Generates the real distribution over a sequence of points

        Args:
            x_vec: the points at which the distribution is generated
        Returns:
            y_vecs: the value of the distributions at the given x points.
        '''

        # Init storage to return
        y_vecs = []

        for j in range(self._num_vectors):

            # Get vector and evaluate corresponding function
            # at the chosen set of points
            vector = self._fourier_vectors[:, j]
            y_vec = [self._make_dist(vector, theta) for theta in x_vec]

            # Store and plot data
            y_vecs.append(y_vec)

        return y_vecs

    def _Holevo_centers(self, vectors=None):
        '''
        Gets the Holevo average phase from a distribution.
        Args:
            vectors: vectors to calculate the average of. If None,
                defaults to the vectors stored in the class.
        Returns:
            centers: the corresponding average angle of the
                distribution.

        '''
        if vectors is None:
            vectors = self._fourier_vectors

        centers = np.angle(vectors[2, :] + 1j*vectors[1, :])
        return centers

    def _Holevo_variances(self, vectors=None, maxvar=4*np.pi**2/12):
        '''
        Gets the Holevo variance from a distribution

        Args:
            vectors: Fourier vectors to calculate the variance of.
                If none, defaults to the vectors stored in the class.
            maxvar: The maximum allowed variance. Defaults to the
                maximum of a uniform distribution on [-pi,pi].
        Returns:
            variances: The Holevo variances of each vector.
        '''
        variances = []

        if vectors is None:
            vectors = self._fourier_vectors

        for j in range(self._num_vectors):
            if (vectors[2, j]**2 + vectors[1, j]**2) == 0:

                variances.append(maxvar)

            else:
                variances.append(min(
                    maxvar,
                    4 / (vectors[2, j]**2+vectors[1, j]**2) - 1))
        return variances

    def _make_matrices(self):
        ''' Function to make the matrices that define multiplication
        by sin(theta) and cos(theta). We store these as sparse matrices
        in CSR format, and initialize them in (row,col,data) format.
        We assume here that self.max_n is less than self.num_freqs.
        '''

        # self._matrices[n][a] contains matrix M^a(n+1)
        # following our paper
        self._matrices = []

        for n in range(1, self._max_n+1):

            # We generate the matrices ourselves in
            # COO format, and then create a
            # CSR matrix directly

            cos_matrix, sin_matrix = self._make_matrix(n)

            # Append to list
            self._matrices.append([cos_matrix, sin_matrix])

    def _make_matrix(self, n):

        # Initialize with terms for when j = 0
        data_cos = [2]
        row_ind_cos = [2*n]
        col_ind_cos = [0]

        data_sin = [-2]
        row_ind_sin = [2*n-1]
        col_ind_sin = [0]

        # First do terms when j < n
        for j in range(1, n):

            data_cos += [1, 1, -1, 1]
            row_ind_cos += [2*j+2*n-1, 2*j+2*n, 2*n-2*j-1, 2*n-2*j]
            col_ind_cos += [2*j-1, 2*j, 2*j-1, 2*j]

            data_sin += [1, -1, -1, -1]
            row_ind_sin += [2*j+2*n, 2*j+2*n-1, 2*n-2*j, 2*n-2*j-1]
            col_ind_sin += [2*j-1, 2*j, 2*j-1, 2*j]

        # Next do terms when j = n
        data_cos += [1, 1, 1]
        row_ind_cos += [4*n-1, 0, 4*n]
        col_ind_cos += [2*n-1, 2*n, 2*n]

        data_sin += [-1, 1, -1]
        row_ind_sin += [0, 4*n, 4*n-1]
        col_ind_sin += [2*n-1, 2*n-1, 2*n]

        # Finish with terms when j > n
        for j in range(n+1, self._num_freqs+1):

            data_cos += [1, 1]
            row_ind_cos += [2*j-2*n-1, 2*j-2*n]
            col_ind_cos += [2*j-1, 2*j]

            data_sin += [-1, 1]
            row_ind_sin += [2*j-2*n, 2*j-2*n-1]
            col_ind_sin += [2*j-1, 2*j]

            if j + n <= self._num_freqs:

                data_cos += [1, 1]
                row_ind_cos += [2*j+2*n-1, 2*j+2*n]
                col_ind_cos += [2*j-1, 2*j]

                data_sin += [1, -1]
                row_ind_sin += [2*j+2*n, 2*j+2*n-1]
                col_ind_sin += [2*j-1, 2*j]

        # Make sparse matrices
        cos_matrix = sparse.csr_matrix(
            (data_cos, (row_ind_cos, col_ind_cos)))
        sin_matrix = sparse.csr_matrix(
            (data_sin, (row_ind_sin, col_ind_sin)))

        return cos_matrix, sin_matrix

    def _vector_product(self, vectors, round_data):

        '''
        Calculates cos^2(n\phi/2+\beta/2)P(\phi) for an input
        n, beta and P(\phi).

        Args:
            vectors: the input vectors to be multiplied

            round_data: dictionary containing data about rotation
                in particular, round data needs 'num_rotations',
                'final_rotation', and 'measurement'.

        Returns:
            updated vectors: the vector representation
                of P_{k,\beta}(m|\phi_j)P_{prior}(\phi_j) for each
                input vector P_{prior}(\phi_j)
        '''

        # Extract the pieces we want from the round_data
        beta = round_data['final_rotation'] -\
            round_data['measurement'] * pi
        n = round_data['num_rotations']

        # Get the matrices for multiplication
        if n <= self._max_n:
            cos_matrix = self._matrices[n-1][0]
            sin_matrix = self._matrices[n-1][1]
        else:
            cos_matrix, sin_matrix = self._make_matrix(n)

        return 0.5*vectors +\
            0.25*cos(beta)*cos_matrix.dot(vectors) +\
            0.25*sin(beta)*sin_matrix.dot(vectors)

    def _mlikelihood(self, a_vec):
        # Calculates the negative likelihood of a set of amplitudes
        # generating the observed measurements

        return -sum([np.log(p_vec.dot(a_vec[:len(p_vec)]))
                    for p_vec in self._p_vecs])

    def _diff_mlikelihood(self, a_vec):
        # The derivative of the above function
        return -sum([p_vec/(p_vec.dot(a_vec[:len(p_vec)]))
                    for p_vec in self._p_vecs]) +\
            self._dinit_mlikelihood(a_vec)

    def _single_diff(self, p_vec, a_vec):
        # The first derivative of a single term in the
        # likelihood function.
        return -p_vec / np.dot(p_vec, a_vec[:len(p_vec)])

    def _jacobian_term(self, p_vec, a_vec):
        # The second derivative of the above function,
        # evaluated for a single p vector - to evaluate
        # and store
        return np.dot(p_vec[:, np.newaxis], p_vec[np.newaxis, :]) /\
            np.dot(p_vec, a_vec[:len(p_vec)]) ** 2

    def _calculate_jacobian(self):
        '''
        Recalculates the Jacobian based on all previously
        calculated p_vecs.
        '''

        self._jacobian = sum(
            [self._jacobian_term(p_vec, self._amplitude_estimates)
             for p_vec in self._p_vecs]) + self._jinit_mlikelihood()

    def _init_mlikelihood(self, a_vec):
        # A normal distributed initial guess at the amplitudes
        return np.dot(0.5*(a_vec-self._amplitude_guess)[np.newaxis, :],
                      np.dot(self._inv_covar_mat,
                             (a_vec-self._amplitude_guess)[:, np.newaxis]))

    def _dinit_mlikelihood(self, a_vec):
        # The derivative of the above likelihood function
        return np.dot(self._inv_covar_mat, (a_vec-self._amplitude_guess))

    def _jinit_mlikelihood(self):
        # The jacobian of the above likelihood function
        return self._inv_covar_mat

    def _make_projector(self):
        # Calculates the projector onto the plane \sum_j|a_j|^2=0

        num_vectors = self._num_vectors
        self._projector = np.identity(num_vectors) -\
            1/num_vectors * np.ones([num_vectors, num_vectors])

    def _make_dist(self, vector, angle):
        '''
        converts a frequency vector into a function evaluated at theta
        Args:
            vector: a fourier vector P(\phi)
            angle: the angle to evaluate P at

        Returns:
            P(angle)
        '''

        res = vector[0]
        for j in range(1, self._num_freqs):
            res += vector[2*j-1] * sin(j*angle)
            res += vector[2*j] * cos(j*angle)
        return res / (2*pi)


class BayesEstimator(ProbabilityDist):
    '''
    Class to estimate a QPE experiment in the absence of error
    '''

    def __init__(self,
                 amplitude_guess=[1],
                 amplitude_vars=[[1]],
                 num_vectors=1,
                 num_freqs=1000,
                 max_n=1,
                 vector_guess=None,
                 full_update_with_failure=False,
                 store_history=True,
                 amplitude_approx_cutoff=100):

        '''
        Args:
            amplitude_guess: estimates for amplitudes of different
                eigenstates in the initial state ($A_j$)
            amplitude_vars: variance on the amplitude estimates
            num_vectors: number of phases $\phi_j$ to estimate
            num_freqs: number of frequencies in Fourier representation.
            max_n: maximum number of unitary evolutions before measurement.
                dictates how many matrices for updates are made and stored.
            vector_guess: a prior estimate of the phases $\phi_j$. If none,
                assumes flat.
            full_update_with_failure: chooses whether to perform
                a full single step of SLSQP whenever the amplitude
                optimization returns an unphysical result (negative
                amplitude squared), or to just enforce physicality
                by hand.
            store_history: whether to store the history of
                estimated values of the estimator. This is only
                a few numbers per update, but it could be costly.
            amplitude_approx_cutoff: a cutoff between estimating the amplitudes
                via full SLSQP and approximating with single steps of Newton's
                method. Increasing may lead to higher accuracy,
                decreasing will lead to faster runtimes.
        '''
        self._amplitude_approx_cutoff = amplitude_approx_cutoff
        self._full_update_with_failure = full_update_with_failure
        self._store_history = store_history

        super(BayesEstimator, self).__init__(
            amplitude_guess=amplitude_guess,
            amplitude_vars=amplitude_vars,
            num_vectors=num_vectors,
            num_freqs=num_freqs,
            max_n=max_n,
            vector_guess=vector_guess)

    def reset(self):
        '''
        Resets estimator to initial state.
        '''

        super(BayesEstimator, self).reset()

        # The following stores the history of the estimator
        # for analysis purposes

        if self._store_history:
            self.averages = []
            self.variances = []
            self.log_Bayes_factor_history = []
            self.amplitudes_history = []

        self.log_Bayes_factor = 0
        self.num_dsets = 0

    def update(self, experiment_data, force_accept=False):
        '''
        Performs the expected Bayesian update, and stores result.

        Args:
            experiment_data: a list of dictionaries, containing
                the data from each round of the experiment. Each round requires
                entries: 'num_rotations', 'measurement', and 'final_rotation'.

            force_accept: whether to insist that the estimator
                accepts this update, even when the result doesn't seem
                physical (due to numerical error or experimental error
                the estimator may think some experiments are so unlikely
                they couldn't have occurred).

        Returns:
            success: boolean value determining whether or not the update
                was successful.

        Raises:
            warnings: if the resulting distribution does not seem
                correct and force_accept=False, will reject the update,
                warn the user, and return False.
        '''

        if len(experiment_data) == 0:
            warnings.warn('This doesnt seem to be an experiment')

        # Get set of products of P_prior(theta_j)p(\vec{m},theta_j)
        temp_vectors = self._calc_vectors(experiment_data=experiment_data)

        # Sanity check
        if any(temp_vectors[0, :] < -1e-10):
            warnings.warn('''New normalization coefficients should be
                          positive! I have {}. I will reject the result of this
                          update unless force_accept=True, but this
                          could imply a previous failure in
                          estimating.'''.format(temp_vectors[0, :]))
            if force_accept is not True:
                return False

        # Getting the integral of these functions from -pi,pi is trivial.
        normalization = sum(temp_vectors[0, :]*self._amplitude_estimates)

        # Sanity check - the normalization should never be negative here
        if normalization < 0:
            warnings.warn('''Negative normalization. amplitudes are {}
                          and vector coefficients are {}. I will reject the
                          result of this update unless force_accept=True
                          but this could imply a previous failure in
                          estimating.'''.format(
                            self._amplitude_estimates, temp_vectors[0, :]))
            if force_accept is not True:
                return False

        # Go through set of vectors, update one at a time
        new_vectors = np.zeros(temp_vectors.shape)
        for j in range(self._num_vectors):

            # Calculation the contribution of this wavefunction to the estimate
            overlap = temp_vectors[0, j]
            nf = overlap * self._amplitude_estimates[j]

            # Update vectors
            new_vectors[:, j] = (((normalization - nf) *
                                  self._fourier_vectors[:, j]) +
                                 (self._amplitude_estimates[j] *
                                  temp_vectors[:, j])) / normalization

        # Check that we still have a positive variance with these
        # new vectors - otherwise throw them out.
        for var in self._Holevo_variances(vectors=new_vectors):
            if var < -EQ_TOLERANCE:
                warnings.warn('''Some variances in the new distribution
                    are negative - var={}. I will reject the result of this
                    update unless force_accept=True, but this could imply a
                    previous failure in estimating.'''.format(var))
                if force_accept is not True:
                    return False

        # Success! Update our distributions
        self._fourier_vectors = new_vectors
        self.log_Bayes_factor += np.log(normalization)
        self.num_dsets += 1

        # Store the probabilities of each vector to have contributed
        # to the observed experiment, to assist in amplitude updates.
        # Important to copy temp_vectors to prevent memory leak.
        self._p_vecs.append(np.copy(temp_vectors[0, :]))

        # Update amplitudes
        if self.num_dsets > self._amplitude_approx_cutoff:
            self._update_amplitudes_approx()
        elif self.num_dsets == self._amplitude_approx_cutoff:
            self._calculate_jacobian()
            self._update_amplitudes()
        else:
            self._update_amplitudes()

        # Store data if required
        if self._store_history:
            self.averages.append(self.estimate())
            self.variances.append(self.estimate_variance())
            self.log_Bayes_factor_history.append(self.log_Bayes_factor)
            self.amplitudes_history.append(self._amplitude_estimates)

        return True

    def _calc_vectors(self, experiment_data):
        '''
        Every update takes the form
        P(\vec{m}|\vec{phi},\vec{A})=\sum_jA_jp(\vec{m}|\phi_j).
        This function should return the set of p(\vec{m}|\phi_j)
        distributions in vector form.

        Takes a set of conditional probability updates and
        multiplies them with the initial distributions, returning the result.
        We assume that the conditional probabilities look like
        cos^2(phi + beta)

        Args:
            experiment_data: list of dictionaries containing data from each
                QPE round in the experiment.
        '''

        # Copy vectors
        temp_vectors = self._fourier_vectors.copy()

        # Loop over updates
        for round_data in experiment_data:

            # Update vectors
            temp_vectors = self._vector_product(vectors=temp_vectors,
                                                round_data=round_data)

            # the zeroth coefficient of any distribution is always decreased
            # by multiplication by cos^2(theta), but should always be positive.
            # This implies that if we have a negative zeroth coefficient,
            # it's the result of numerical error and should be dropped.
            for j in range(temp_vectors.shape[1]):
                if temp_vectors[0, j] <= 0:
                    temp_vectors[:, j] = np.zeros((self._num_freqs*2+1))

        return temp_vectors

    def _update_amplitudes_approx(self):
        '''
        Updates amplitudes approximately via Newton's method.
        To be used after ~100 amplitude estimates or so, when
        the approximation is close
        '''

        if self._num_vectors == 1:
            self._amplitude_estimates = np.array([1])
            return
        try:
            self._jacobian += self._jacobian_term(
                self._p_vecs[-1], self._amplitude_estimates)
        except:
            self._calculate_jacobian()

        d_amp = np.dot(
            self._projector, np.dot(
                np.linalg.inv(self._jacobian),
                self._single_diff(self._p_vecs[-1], self._amplitude_estimates)))

        temp_ae = self._amplitude_estimates - d_amp

        # Check that we fit within the boundaries of
        # A_i > 0 for each i.
        if min(temp_ae) < 0:

            # We have two options here - either repeat the costly
            # full optimization using SLSQP, or project onto the allowed space.

            # Repeat full update
            if self._full_update_with_failure:
                self._update_amplitudes()
                self._calculate_jacobian()
                return

            # Projecting onto the allowed space
            d_amp = d_amp / np.max(np.abs(d_amp / self._amplitude_estimates))
            temp_ae = self._amplitude_estimates - d_amp

            # Prevent numerical errors
            for j in range(len(temp_ae)):
                if temp_ae[j] < 0:
                    temp_ae[j] = 0
            temp_ae = temp_ae / sum(temp_ae)

        self._amplitude_estimates = temp_ae

    def _update_amplitudes(self):

        # Amplitudes are updated as the maximum likelihood estimator.
        # We want to use the SLSQP function from scipy as it both
        # takes constraints and bounds, and uses a Jacobian which
        # we can calculate trivially.

        if self._num_vectors == 1:
            self._amplitude_estimates = np.array([1])
            return

        res = optimize.minimize(
                fun=self._mlikelihood,
                method='SLSQP',
                x0=self._amplitude_estimates,
                jac=self._diff_mlikelihood,
                bounds=[(0, 1) for _ in range(self._num_vectors)],
                constraints={'type': 'eq',
                             'fun': lambda x: 1-sum(x)})

        if res['success']:
            self._amplitude_estimates = res['x']
        else:
            warnings.warn('''The amplitude update failed. This estimation
                should probably no longer be trusted.''')

    def gen_distribution(self, experiment_data):
        '''
        Generates the posterior distribution estimate
        of the eigenspectrum of the wavefunction.

        Args:
            experiment_data: list of dictionaries containing
                data for each round of the experiment.
        Returns:
            distribution: the Fourier vector containing
                the predicted diagonal of the post-experiment
                density matrix.
        '''
        return np.dot(self._calc_vectors(experiment_data),
                      self._amplitude_estimates)

    def estimate(self):
        '''
        Returns:
            the best current estimate of the eigenvalues
        '''
        return self._Holevo_centers()

    def estimate_variance(self):
        '''
        Returns:
            the estimated variance in the best current
            estimate of the eigenvalues.
        '''
        return self._Holevo_variances()


class BayesDepolarizingEstimator(BayesEstimator):
    '''
    Bayesian estimator that includes a depolarizing noise
    channel parametrized by epsilon_B and epsilon_D.
    '''

    def __init__(self,
                 K1=np.inf,
                 Kerr=np.inf,
                 **kwargs):

        '''
        Args:
            K1: T1/t', where t' is the length of time
                over which the system can decay.
            Kerr: Terr/T_U, where T_u is the length of
                a single unitary circuit, and Terr is
                the coherence time of the system.
        '''

        self.K1 = K1
        self.Kerr = Kerr

        super(BayesDepolarizingEstimator, self).__init__(**kwargs)

    def _epsilon_D_function(self, n):
        '''
        epsilon D is the depolarizing channel - an n-dependent
        probability of failing and returning a completely random
        result.

        Args:
            n: the number of unitary rotations performed
        Returns:
            epsilon_D: the probability of failure during this experiment.
        '''
        epsilon_D = 1 - np.exp(-n / self.K1)
        return epsilon_D

    def _epsilon_B_function(self):
        '''
        epsilon_B is the T1 channel - an n-indepnedent probability
        of the ancilla failing and returning 0. This decay can only occur
        at the end of the experiment (as T1 decay while the ancilla is
        in a coherent state will result in a random result being returned).

        Returns:
            epsilon_B: the probability of failure at the end of this
                experiment.
        '''
        epsilon_B = 1 - np.exp(-1 / self.Kerr)
        return epsilon_B

    def _vector_product(self, vectors, round_data):

        '''
        Calculates cos^2(n\phi/2+\beta/2)P(\phi) for an input
        n, beta and P(\phi).

        Args:
            vectors: the input vectors to be multiplied

            round_data: dictionary containing data about rotation
                in particular, round data needs 'num_rotations',
                'final_rotation', and 'measurement'.
                Can contain additional value of 'true_measurement'
                in case ancilla qubits are not reset between rounds.

        Returns:
            updated vectors: the vector representation
                of P_{k,\beta}(m|\phi_j)P_{prior}(\phi_j) for each
                input vector P_{prior}(\phi_j)
        '''

        # Extract the pieces we want from the round_data
        beta = round_data['final_rotation'] -\
            round_data['measurement'] * pi
        n = round_data['num_rotations']
        try:
            mr = round_data['true_measurement']
        except KeyError:
            mr = round_data['measurement']

        # Get the correct cos matrix
        if n < self._max_n:
            cos_matrix = self._matrices[n-1][0]
            sin_matrix = self._matrices[n-1][1]
        else:
            cos_matrix, sin_matrix = self._make_matrix(n)

        epsilon_D = self._epsilon_D_function(n)
        epsilon_B = self._epsilon_B_function()

        return ((1 - epsilon_D) * (
                0.5*vectors +
                0.25*cos(beta)*cos_matrix.dot(vectors) +
                0.25*sin(beta)*sin_matrix.dot(vectors)) +
                vectors * (0.5*epsilon_D +
                           (-1)**mr * 0.5 * epsilon_B))


class TimeSeriesEstimator:

    def __init__(self,
                 max_experiment_length,
                 num_freqs_max,
                 depolarizing_noise=False,
                 singular_values_cutoff=1e-12,
                 store_history=False,
                 automatic_average=False):

        '''
        Time-series estimator: a QPE estimator based on a time-series
            analysis of g(k)=\sum_jA_je^{ik\phi_j}, where
            A_j=|<\phi_j|\Psi>|^2 is the probabilistic overlap
            with the starting state |\Psi>, and \phi_j is the unitary
            eigenvalue U|\phi_j>=e^{i\phi_j}|\phi_j>.

        Args:
            max_experiment_length: the largest number (K) of unitary operations
                to be performed during the experiment. At least two experiments
                must be made at each k=1,...,K, with two different values
                of the final rotation beta (not separated by pi), in order to
                obtain an estimate from this estimator.
            num_freqs_max: the maximum number of eigenvalues \phi_j to
                estimate. Must be less than max_experiment_length without
                depolarizing noise and less than max_experiment_length/2 with.
            depolarizing_noise: whether or not to account for depolarizing
                noise in the model.
            singular_values_cutoff: the value below which singular values
                are treated as zero (for counting the number of eigenvalues
                present in the problem).
            store_history: whether to save the estimated eigenvalues whenever
                they are estimated.
            automatic_average: whether to reestimate the eigenvalues whenever
                new data is added.
        '''

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
        '''
        Resets estimator for new experiment
        '''
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
        '''
        Takes new experiment data in dictionary format. Note - storing data
        in this format is significantly worse than tallying results and
        inserting with update_mass (provided here to work similarly
        to the Bayesian estimators).

        Args:
            experiment_data: list of dictionaries containing information
                about each round in the experiment. Each dictionary
                must contain the entries 'final_rotation', 'measurement',
                and 'num_rotations'. For the single-round estimator,
                experiment_data must be of length 1.

        Returns:
            success of update (always True for this estimator)
        '''
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
        '''
        Takes data to update g(k) in the form  The preferable form to input
        data, but requires that experiments take a final rotation of either
        0 or pi/2.

        Args:
            f_data: f_data[k,a,m] = number of m=0,1
                measurements following k controled unitary rotations
                and a final rotation of a=0,1*(pi/2).

        Returns:
            success of update (always True for this estimator)
        '''
        self._re_abs += f_data[:, 0, 0] + f_data[:, 0, 1]
        self._re_est += f_data[:, 0, 0] - f_data[:, 0, 1]
        self._im_abs += f_data[:, 1, 0] + f_data[:, 1, 1]
        self._im_est -= f_data[:, 1, 0] - f_data[:, 1, 1]

        self._update_flag = False

        if self._automatic_average is True:
            self.estimate()

        return True

    def _update_function(self):
        '''
        Calculates the estimate of the function from the individual counts
        '''

        function_real = self._re_est/self._re_abs
        function_imag = self._im_est/self._im_abs

        self._function_estimate = function_real + 1j * function_imag
        self._function_estimate[0] = 1

        if self._depolarizing_noise is False:
            self._function_estimate = np.hstack(
                [self._function_estimate[:0:-1].conj(),
                 self._function_estimate])

        self._update_flag = True

    def _make_Hankel_matrices(self):
        '''
        Makes the two Hankel matrices
        '''

        if self._update_flag is False:
            self._update_function()

        F0 = sp.linalg.hankel(
            c=self._function_estimate[:self._num_freqs],
            r=self._function_estimate[self._num_freqs-1:-1])

        F1 = sp.linalg.hankel(
            c=self._function_estimate[1:self._num_freqs+1],
            r=self._function_estimate[self._num_freqs:])

        self._F0 = F0.T
        self._F1 = F1.T

    def _make_translation_matrix(self):
        '''
        Makes the (transposed) translation matrix from the two Hankel matrices.
        '''

        self._TT_matrix = sp.linalg.lstsq(self._F0, self._F1)[0]

    def _get_num_frequencies(self):
        '''
        Finds the number of singular values above cutoff in
        the Hankel matrices - this corresponds to our best estimate
        of the number of frequencies in the problem. If the Hankel
        matrices are full-rank, then we possibly have too many
        frequencies to estimate, and low-amplitude estimates
        shouldn't be trusted (in general we advise not trusting
        any results with amplitude estimate below 1/max_experiment_length)
        '''

        self._num_freqs = self._num_freqs_max
        if self._singular_values_cutoff > 0:
            self._make_Hankel_matrices()
            singular_values = np.linalg.svd(self._F0)[1]
            self._num_freqs = sum(singular_values >
                                  self._singular_values_cutoff)
            if self._num_freqs > self._num_freqs_max:
                self._num_freqs = self._num_freqs_max

    def _get_amplitudes(self):
        '''
        Obtains the amplitudes of the problem by a least squares fit
        of the target function to the estimated function.
        '''

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
        '''
        Estimates all frequencies as required.

        Args:
            return_amplitudes: whether to return both the corresponding
                amplitude estimates along with the angle estimates.
        Returns
            angles: the estimated target frequencies
            amplitudes(optional): The corresponding amplitude estimates
        '''
        self._get_num_frequencies()
        self._make_Hankel_matrices()
        self._make_translation_matrix()
        self.angles = np.angle(np.linalg.eigvals(self._TT_matrix.T))
        self._get_amplitudes()
        self.amplitudes, self.angles = zip(*sorted(zip(
            self.amplitudes, self.angles),
            key=lambda x: np.real(x[0]), reverse=True))

        if self._store_history:
            self.average_history.append(np.array(self.angles))

        if return_amplitudes:
            return self.angles, self.amplitudes
        else:
            return self.angles


class TimeSeriesMultiRoundEstimator(TimeSeriesEstimator):

    '''
    Time series estimator for multi-round QPE experiments.
    Generates g(k) from a specific set of multi-round experiments
    and then performs a time-series analysis as in the single-round
    case, but weighted by the calculated variance in individual
    points.

    Requires a very specific set of multi-round experiments
    (N/2 rounds with 0 final rotation, N/2 rounds with pi/2
    final rotation). Unfortunately this doesn't seem very easily
    generalizeable.
    '''

    def __init__(self, **kwargs):

        super(TimeSeriesMultiRoundEstimator, self).__init__(
            depolarizing_noise=True, **kwargs)

        self._choose_vec = np.array([
            sp.special.comb(self._max_experiment_length, k)
            for k in range(self._max_experiment_length+1)])

    def reset(self):
        '''
        Resets estimator for new experiment
        '''

        self._hamming_mat = np.zeros([self._max_experiment_length//2+1,
                                     self._max_experiment_length//2+1])
        self._function_estimate = np.zeros(self._max_experiment_length + 1)
        self._num_experiments = 0
        self._num_freqs = self._num_freqs_max
        self._update_flag = True
        if self._store_history:
            self.average_history = []

    def update(self, experiment_data):
        '''
        Adds data in json format.

        Args:
            experiment_data: list of dictionaries containing
                arguments and measurements of each round.
                arguments must contain 'measurement', 'final_rotation'.
                and must have an equal number of 0 and pi/2
                final rotations (restricted by our experiment design).

        Returns:
            success of update (always True for this estimator)
        '''

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
        '''
        Takes data in the form hamming_mat[hw0,hw1] = number
        of observed data points with hw0 measurements of '1' after a 0
        rotation and hw1 measurements of '1' after a pi/2 rotation

        Args:
            hamming_mat: register of multiple measurements as
                described above.

        Returns:
            success of update (always True for this estimator)
        '''
        self._hamming_mat += hamming_mat
        self._num_experiments += np.sum(hamming_mat)
        self._update_flag = False

        if self._automatic_average is True:
            self.estimate()

        return True

    def _gen_comb(self, N, n):
        '''
        A generalized combinatorial function that returns zero
        outside of the usual bounds (required for our summation).

        Args:
            N, n: combinatorial input
        Returns:
            N choose n.
        '''
        if n > N or n < 0:
            return 0
        return sp.special.comb(N, n)

    def _p_function(self, R, l, m):
        '''
        Function to calculate the recurring combinatorial factors
        used in our chosen generation of g(k). See paper for details of input
        and output.

        Args:
            R,l,m: see arXiv:XXXX.XXXX for details
        Returns:
            calculation of eq.XXX from arXiv:XXXX.XXXX.
        '''
        res = 0
        for p in range(0, l//2 + 1):
            res += self._gen_comb(m, 2*p) * self._gen_comb(R//2 - m, l - 2*p)
        res = 2 * res / self._gen_comb(R//2, l) - 1
        return res

    def _update_function(self):
        '''
        Calculates the estimate of the function g(k) from the input counts,
        following eq.XXX of arXiv:XXXX.XXXX.
        '''
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

    def _make_Hankel_matrices(self):
        '''
        Makes the two Hankel matrices and stores them,
        along with the standard deviation in the target Hankel matrix.
        '''

        if self._update_flag is False:
            self._update_function()

        F0 = sp.linalg.hankel(
            c=self._function_estimate[:self._num_freqs],
            r=self._function_estimate[self._num_freqs-1:-1])

        F1 = sp.linalg.hankel(
            c=self._function_estimate[1:self._num_freqs+1],
            r=self._function_estimate[self._num_freqs:])

        F1std = sp.linalg.hankel(
            c=self._function_std[1:self._num_freqs+1],
            r=self._function_std[self._num_freqs:])

        self._F0 = F0.T
        self._F1 = F1.T
        self._F1std = F1std.T

    def _make_translation_matrix(self):
        '''
        Makes the (transposed) translation matrix from the two Hankel matrices.
        Weights according to the standard deviation of F1.
        '''

        TT_columns = []
        for j in range(self._num_freqs):
            w_mat = np.diag(1/self._F1std[:, j])
            new_column = sp.linalg.lstsq(
                np.dot(w_mat, self._F0),
                np.dot(w_mat, self._F1[:, j]))[0]
            TT_columns.append(new_column)
        self._TT_matrix = np.array(TT_columns)

    def _get_amplitudes(self):
        '''
        Obtains the amplitudes of the problem by a least squares fit
        of the target function to the estimated function.
        '''

        self._generation_matrix = np.array([
            [np.exp(1j*k*angle) for angle in self.angles]
            for k in range(0, self._max_experiment_length//2+1)])

        self.amplitudes = sp.linalg.lstsq(self._generation_matrix,
                                          self._function_estimate)[0]
