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
in arXiv:1809.09697 (in particular in App.C and App.C.1)
'''
import warnings
import numpy
from numpy import cos, sin, pi
from scipy import sparse, optimize
from openfermion.config import EQ_TOLERANCE


class FourierProbabilityDist(object):
    '''
    Stores a estimate of multiple phases in the form arXiv:1809.09697 App.C.
    In particular, stores the Fourier representation as a sum of sine and
    cosine waves (Eq. (C1)), and provides methods for manipulation of these.
    '''

    def __init__(self,
                 amplitude_guess=None,
                 amplitude_vars=None,
                 num_vectors=1,
                 num_freqs=1000,
                 max_n=1,
                 vector_guess=None):
        '''
        Args:
            amplitude_guess: estimates for amplitudes of different
                eigenstates in the initial state.
            amplitude_vars: variance on the amplitude estimates
            num_vectors: number of phases to estimate
            num_freqs: number of frequencies in Fourier representation.
            max_n: maximum number of unitary evolutions before measurement.
                dictates how many matrices for updates are made and stored.
            vector_guess: a prior estimate of the phases. If none,
                assumes flat.

        Raises:
            ValueError: if prior distribution does not have required shape.
            ValueError: if insufficient amplitude guesses are given.
            ValueError: if starting amplitudes do not break symmetry.
        '''

        # Store size data
        self._num_vectors = num_vectors
        self._num_freqs = num_freqs

        if amplitude_guess is None:
            amplitude_guess = [1]
        if amplitude_vars is None:
            amplitude_vars = [[1]]

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

        if any([numpy.abs(self._amplitude_guess[j] -
                          self._amplitude_guess[j+1]) < EQ_TOLERANCE
                for j in range(num_vectors - 1)]):
            raise ValueError('''Starting amplitudes must be different.''')

        # Store max_n
        self._max_n = max_n

        # Set un-created variables to be made elsewhere in the class to None
        self._jacobian = None
        self._amplitude_estimates = None
        self._fourier_vectors = None

        # Create matrices for multiplication
        self._make_matrices()
        self._make_projector()
        self.reset()

    def reset(self):
        '''
        Resets the distribution.
        '''
        if self._vector_guess is not None:
            self._fourier_vectors = numpy.array(self._vector_guess)
        else:
            # Initialize probability distributions
            self._fourier_vectors = numpy.zeros((self._num_freqs*2+1,
                                                 self._num_vectors))
            for j in range(self._num_vectors):
                self._fourier_vectors[0, j] = 1

        # Initialize p_vec with some information to break the symmetry
        self._p_vecs = [numpy.array(self._amplitude_guess)]
        self._amplitude_estimates = numpy.array(self._amplitude_guess)
        self._inv_covar_mat = numpy.linalg.inv(
            numpy.array(self._amplitude_vars))

    def get_real_dist(self, x_vec=numpy.linspace(-pi, pi, 101)):
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

    def _holevo_centers(self, vectors=None):
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

        centers = numpy.angle(vectors[2, :] + 1j*vectors[1, :])
        return centers

    def _holevo_variances(self, vectors=None, maxvar=4*numpy.pi**2/12):
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

        for index in range(1, self._max_n+1):

            # We generate the matrices ourselves in
            # COO format, and then create a
            # CSR matrix directly

            cos_matrix, sin_matrix = self._make_matrix(index)

            # Append to list
            self._matrices.append([cos_matrix, sin_matrix])

    def _make_matrix(self, nrot):
        '''
        Makes matrices that sends a Fourier representation of p(phi)
        to the Fourier representation of cos^2(n*phi/2+beta/2)*p(phi).

        See App. C in arXiv:1809.09697

        Args:
            nrot: the number of rotations n in the above formula.
        '''

        # Initialize with terms for when j = 0
        data_cos = [2]
        row_ind_cos = [2*nrot]
        col_ind_cos = [0]

        data_sin = [-2]
        row_ind_sin = [2*nrot-1]
        col_ind_sin = [0]

        # First do terms when j < n
        for j in range(1, nrot):

            data_cos += [1, 1, -1, 1]
            row_ind_cos += [2*j+2*nrot-1, 2*j+2*nrot, 2*nrot-2*j-1, 2*nrot-2*j]
            col_ind_cos += [2*j-1, 2*j, 2*j-1, 2*j]

            data_sin += [1, -1, -1, -1]
            row_ind_sin += [2*j+2*nrot, 2*j+2*nrot-1, 2*nrot-2*j, 2*nrot-2*j-1]
            col_ind_sin += [2*j-1, 2*j, 2*j-1, 2*j]

        # Next do terms when j = n
        data_cos += [1, 1, 1]
        row_ind_cos += [4*nrot-1, 0, 4*nrot]
        col_ind_cos += [2*nrot-1, 2*nrot, 2*nrot]

        data_sin += [-1, 1, -1]
        row_ind_sin += [0, 4*nrot, 4*nrot-1]
        col_ind_sin += [2*nrot-1, 2*nrot-1, 2*nrot]

        # Finish with terms when j > n
        for j in range(nrot+1, self._num_freqs+1):

            data_cos += [1, 1]
            row_ind_cos += [2*j-2*nrot-1, 2*j-2*nrot]
            col_ind_cos += [2*j-1, 2*j]

            data_sin += [-1, 1]
            row_ind_sin += [2*j-2*nrot, 2*j-2*nrot-1]
            col_ind_sin += [2*j-1, 2*j]

            if j + nrot <= self._num_freqs:

                data_cos += [1, 1]
                row_ind_cos += [2*j+2*nrot-1, 2*j+2*nrot]
                col_ind_cos += [2*j-1, 2*j]

                data_sin += [1, -1]
                row_ind_sin += [2*j+2*nrot, 2*j+2*nrot-1]
                col_ind_sin += [2*j-1, 2*j]

        # Make sparse matrices
        cos_matrix = sparse.csr_matrix(
            (data_cos, (row_ind_cos, col_ind_cos)))
        sin_matrix = sparse.csr_matrix(
            (data_sin, (row_ind_sin, col_ind_sin)))

        return cos_matrix, sin_matrix

    def _vector_product(self, vectors, round_data):
        '''
        Calculates cos^2(n*phi/2+beta/2)P(phi) for an input
        n, beta and P(phi).

        Args:
            vectors: the input vectors to be multiplied

            round_data: dictionary containing data about rotation
                in particular, round data needs 'num_rotations',
                'final_rotation', and 'measurement'.

        Returns:
            updated vectors: the vector representation
                of P_{k,beta}(m|phi_j)P_{prior}(phi_j) for each
                input vector P_{prior}(phi_j)
        '''

        # Extract the pieces we want from the round_data
        beta = round_data['final_rotation']
        meas = round_data['measurement']
        nrot = round_data['num_rotations']

        # Get the matrices for multiplication
        if nrot <= self._max_n:
            cos_matrix = self._matrices[nrot-1][0]
            sin_matrix = self._matrices[nrot-1][1]
        else:
            cos_matrix, sin_matrix = self._make_matrix(nrot)

        return 0.5 * vectors +\
            0.25 * cos(beta - meas * pi) * cos_matrix.dot(vectors) +\
            0.25 * sin(beta - meas * pi) * sin_matrix.dot(vectors)

    def _mlikelihood(self, a_vec):
        # Calculates the negative likelihood of a set of amplitudes
        # generating the observed measurements
        return -sum([numpy.log(p_vec.dot(a_vec[:len(p_vec)]))
                     for p_vec in self._p_vecs]) +\
            self._init_mlikelihood(a_vec)

    def _diff_mlikelihood(self, a_vec):
        # The derivative of the above function
        return -sum([p_vec/(p_vec.dot(a_vec[:len(p_vec)]))
                     for p_vec in self._p_vecs]) +\
            self._dinit_mlikelihood(a_vec)

    def _single_diff(self, p_vec, a_vec):
        # The first derivative of a single term in the
        # likelihood function.
        return -p_vec / numpy.dot(p_vec, a_vec[:len(p_vec)])

    def _jacobian_term(self, p_vec, a_vec):
        # The second derivative of the above function,
        # evaluated for a single p vector - to evaluate
        # and store
        return numpy.dot(p_vec[:, numpy.newaxis], p_vec[numpy.newaxis, :]) /\
            numpy.dot(p_vec, a_vec[:len(p_vec)]) ** 2

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
        return numpy.dot(
            0.5*(a_vec-self._amplitude_guess)[numpy.newaxis, :],
            numpy.dot(self._inv_covar_mat,
                      (a_vec-self._amplitude_guess)[:, numpy.newaxis])).item()

    def _dinit_mlikelihood(self, a_vec):
        # The derivative of the above likelihood function
        return numpy.dot(self._inv_covar_mat, (a_vec-self._amplitude_guess))

    def _jinit_mlikelihood(self):
        # The jacobian of the above likelihood function
        return self._inv_covar_mat

    def _make_projector(self):
        # Calculates the projector onto the plane sum_j|a_j|^2=0
        num_vectors = self._num_vectors
        self._projector = numpy.identity(num_vectors) -\
            1/num_vectors * numpy.ones([num_vectors, num_vectors])

    def _make_dist(self, vector, angle):
        '''
        converts a frequency vector into a function evaluated at theta
        Args:
            vector: a fourier vector P(phi)
            angle: the angle to evaluate P at

        Returns:
            P(angle)
        '''
        res = vector[0]
        for j in range(1, self._num_freqs):
            res += vector[2*j-1] * sin(j*angle)
            res += vector[2*j] * cos(j*angle)
        return res / (2*pi)


class BayesEstimator(FourierProbabilityDist):
    '''
    Class to estimate a QPE experiment in the absence of error
    '''

    def __init__(self,
                 amplitude_guess=None,
                 amplitude_vars=None,
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
            num_vectors: number of phases to estimate
            num_freqs: number of frequencies in Fourier representation.
            max_n: maximum number of unitary evolutions before measurement.
                dictates how many matrices for updates are made and stored.
            vector_guess: a prior estimate of the phases. If none,
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
            self.log_bayes_factor_history = []
            self.amplitudes_history = []

        self.log_bayes_factor = 0
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
        if not experiment_data:
            warnings.warn('This doesnt seem to be an experiment')

        # Get set of products of P_prior(phi_j)p(m,phi_j)
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

        # Go through set of vectors, update one at a time
        new_vectors = numpy.zeros(temp_vectors.shape)
        for j in range(self._num_vectors):

            # Calculation the contribution of this wavefunction to the estimate
            overlap = temp_vectors[0, j]
            norm_factor = overlap * self._amplitude_estimates[j]

            # Update vectors
            new_vectors[:, j] = (((normalization - norm_factor) *
                                  self._fourier_vectors[:, j]) +
                                 (self._amplitude_estimates[j] *
                                  temp_vectors[:, j])) / normalization

        # Check that we still have a positive variance with these
        # new vectors - otherwise throw them out.
        for var in self._holevo_variances(vectors=new_vectors):
            if var < -EQ_TOLERANCE:
                warnings.warn('''Some variances in the new distribution
                    are negative - var={}. I will reject the result of this
                    update unless force_accept=True, but this could imply a
                    previous failure in estimating.'''.format(var))
                if force_accept is not True:
                    return False

        # Success! Update our distributions
        old_fourier_vectors = self._fourier_vectors
        self._fourier_vectors = new_vectors
        self.log_bayes_factor += numpy.log(normalization)
        self.num_dsets += 1

        # Store the probabilities of each vector to have contributed
        # to the observed experiment, to assist in amplitude updates.
        # Important to copy temp_vectors to prevent memory leak.
        self._p_vecs.append(numpy.copy(temp_vectors[0, :]))

        # Update amplitudes
        old_amplitudes = numpy.array(self._amplitude_estimates)
        if self.num_dsets > self._amplitude_approx_cutoff:
            self._update_amplitudes_approx()
        elif self.num_dsets == self._amplitude_approx_cutoff:
            self._calculate_jacobian()
            self._update_amplitudes()
        else:
            self._update_amplitudes()

        if any(self._amplitude_estimates < 0):
            warnings.warn('''Im getting negative amplitudes for
                the distribution - amplitudes = {}. I will reject
                this update unless force_accept=True, but this
                could indicate a previous failure in estimation.
                '''.format(self._amplitude_estimates))
            if force_accept is not True:
                self._amplitude_estimates = old_amplitudes
                self._fourier_vectors = old_fourier_vectors
                return False

        # Store data if required
        if self._store_history:
            self.averages.append(self.estimate())
            self.variances.append(self.estimate_variance())
            self.log_bayes_factor_history.append(self.log_bayes_factor)
            self.amplitudes_history.append(self._amplitude_estimates)

        return True

    def _calc_vectors(self, experiment_data):
        '''
        Every update takes the form P(m|phi,A)=sum_jA_jp(m|phi_j).
        This function should return the set of p(m|phi_j)
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
                    temp_vectors[:, j] = numpy.zeros((self._num_freqs*2+1))

        return temp_vectors

    def _update_amplitudes_approx(self):
        '''
        Updates amplitudes approximately via Newton's method.
        To be used after ~100 amplitude estimates or so, when
        the approximation is close
        '''
        if self._num_vectors == 1:
            self._amplitude_estimates = numpy.array([1])
            return

        self._jacobian += self._jacobian_term(
            self._p_vecs[-1], self._amplitude_estimates)

        d_amp = numpy.dot(
            self._projector, numpy.dot(
                numpy.linalg.inv(self._jacobian),
                self._single_diff(
                    self._p_vecs[-1], self._amplitude_estimates)))

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
            d_amp = d_amp / numpy.max(
                numpy.abs(d_amp / self._amplitude_estimates))
            temp_ae = self._amplitude_estimates - d_amp

            # Prevent numerical errors
            for j, val in enumerate(temp_ae):
                temp_ae[j] = max(val, 0)
            temp_ae = temp_ae / sum(temp_ae)

        self._amplitude_estimates = temp_ae

    def _update_amplitudes(self):
        '''
        Amplitudes are updated as the maximum likelihood estimator.
        We want to use the SLSQP function from scipy as it both
        takes constraints and bounds, and uses a Jacobian which
        we can calculate trivially.
        '''
        if self._num_vectors == 1:
            self._amplitude_estimates = numpy.array([1])
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

    def estimate(self, return_amplitudes=False):
        '''
        Returns:
            the best current estimate of the eigenvalues
        '''
        eigenvalues = self._holevo_centers()
        if return_amplitudes:
            amplitudes = self.estimate_amplitudes()
            return eigenvalues, amplitudes
        return eigenvalues

    def estimate_variance(self):
        '''
        Returns:
            the estimated variance in the best current
            estimate of the eigenvalues.
        '''
        return self._holevo_variances()

    def estimate_amplitudes(self):
        '''
        Returns:
            a copy of the estimated amplitudes.
        '''
        return numpy.array(self._amplitude_estimates)


class BayesDepolarizingEstimator(BayesEstimator):
    '''
    Bayesian estimator that includes a depolarizing noise
    channel parametrized by epsilon_B and epsilon_D.
    '''

    def __init__(self,
                 k_1=numpy.inf,
                 k_err=numpy.inf,
                 **kwargs):
        '''
        Args:
            k1: T1/t', where t' is the length of time
                over which the system can decay.
            kerr: Terr/T_U, where T_u is the length of
                a single unitary circuit, and Terr is
                the coherence time of the system.
        '''
        self.k_1 = k_1
        self.k_err = k_err

        super(BayesDepolarizingEstimator, self).__init__(**kwargs)

    def _epsilon_d_function(self, nrot):
        '''
        epsilon d is the depolarizing channel - an n-dependent
        probability of failing and returning a completely random
        result.

        Args:
            n: the number of unitary rotations performed
        Returns:
            epsilon_d: the probability of failure during this experiment.
        '''
        epsilon_d = 1 - numpy.exp(-nrot / self.k_err)
        return epsilon_d

    def _epsilon_b_function(self):
        '''
        epsilon_b is the T1 channel - an n-indepnedent probability
        of the ancilla failing and returning 0. This decay can only occur
        at the end of the experiment (as T1 decay while the ancilla is
        in a coherent state will result in a random result being returned).

        Returns:
            epsilon_b: the probability of failure at the end of this
                experiment.
        '''
        epsilon_b = 1 - numpy.exp(-1 / self.k_1)
        return epsilon_b

    def _vector_product(self, vectors, round_data):
        '''
        Calculates cos^2(n*phi/2+beta/2)P(phi) for an input
        n, beta and P(phi).

        Args:
            vectors: the input vectors to be multiplied

            round_data: dictionary containing data about rotation
                in particular, round data needs 'num_rotations',
                'final_rotation', and 'measurement'.
                Can contain additional value of 'true_measurement'
                in case ancilla qubits are not reset between rounds.

        Returns:
            updated vectors: the vector representation
                of P_{k,beta}(m|phi_j)P_{prior}(phi_j) for each
                input vector P_{prior}(phi_j)
        '''

        # Extract the pieces we want from the round_data
        beta = round_data['final_rotation']
        meas = round_data['measurement']
        nrot = round_data['num_rotations']
        try:
            meas_real = round_data['true_measurement']
        except KeyError:
            meas_real = round_data['measurement']

        # Get the correct cos matrix
        if nrot < self._max_n:
            cos_matrix = self._matrices[nrot-1][0]
            sin_matrix = self._matrices[nrot-1][1]
        else:
            cos_matrix, sin_matrix = self._make_matrix(nrot)

        epsilon_d = self._epsilon_d_function(nrot)
        epsilon_b = self._epsilon_b_function()

        return (
            (1 - epsilon_d) * (
                0.5 * vectors +
                0.25 * cos(beta - meas * pi) * cos_matrix.dot(vectors) +
                0.25 * sin(beta - meas * pi) * sin_matrix.dot(vectors)) +
            vectors * (0.5 * epsilon_d +
                       (-1)**meas_real * 0.5 * epsilon_b))
