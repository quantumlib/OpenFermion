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
Classes to store probability distributions for estimation
of random variables.
"""
import numpy
from scipy import sparse
from numpy import cos, sin, pi


class FourierProbabilityDist(object):
    """
    Stores a multivariant Fourier representation of a periodic function:

    f(phi_0,phi_1,...) = sum_j A_j f_j
    f_j = sum_n (c_{j,2n} cos(n*phi_j) + c_{j,2n-1} sin(n*phi_j))

    In particular, this class stores values of A_j, c_{j,2n},
    c_{j,2n+1} from the above equation.

    It also provides routines for multiplying f by functions of the form
    cos^2(n*phi_j + beta), and for calculating maximum-likelihood
    distributions of the A_j variables.

    More details can be found in arXiv:1809.09697, Appendix C.
    """

    def __init__(self,
                 init_amplitude_guess=None,
                 init_amplitude_vars=None,
                 num_vectors=1,
                 num_freqs=1000,
                 vector_guess=None):
        """
        Args:
            init_amplitude_guess (numpy array or list):
                We take an initial normal distribution for the amplitudes
                which requires for each a mean and a variance - 'guess' is
                the mean
            init_amplitude_vars (numpy array or list): variance of the
                estimated distribution in init_amplitude_guess
            num_vectors (int): number of phases phi_j to estimate
            num_freqs (int): number of wave components cos(n*phi_j)
                to store coefficients for (this provides a cut-off to
                the sensitivity of the function).
            max_n (int): the maximum n required by the user in
                multiplying the stored function functions of the form
                cos(n*phi_j + beta).
            vector_guess (numpy array): a prior estimate of the function,
                given in terms of the Fourier components c_{j,n}.
                If None, estimated as a flat function over [-pi,pi].

        Raises:
            ValueError: if prior distribution does not have required shape.
            ValueError: if insufficient amplitude guesses are given.
            ValueError: if starting amplitudes do not break symmetry.
        """

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

        if init_amplitude_guess is None:
            init_amplitude_guess = [1]
        if init_amplitude_vars is None:
            init_amplitude_vars = [[0]]

        if num_vectors != len(init_amplitude_guess):
            raise ValueError('''I need {} amplitudes, but you have
                given me {}.'''.format(num_vectors, len(init_amplitude_guess)))

        if any([numpy.isclose(init_amplitude_guess[j],
                              init_amplitude_guess[j+1])
                for j in range(num_vectors - 1)]):
            raise ValueError('''Starting amplitudes must be different.''')

        self._init_amplitude_guess = init_amplitude_guess
        self._init_amplitude_vars = init_amplitude_vars

        # The current estimate of the amplitudes is just the mean
        # of the given guess.
        self._amplitude_estimates = numpy.array(init_amplitude_guess)

        if self._vector_guess is not None:
            self._fourier_vectors = numpy.array(vector_guess)
        else:
            # Initialize probability distributions
            self._fourier_vectors = numpy.zeros((self._num_freqs*2+1,
                                                 self._num_vectors))
            for j in range(self._num_vectors):
                self._fourier_vectors[0, j] = 1

        self._p_vecs = []

        self._inv_covar_mat = numpy.linalg.inv(
            numpy.array(self._init_amplitude_vars))

    def get_real_dist(self, x_vec=None):
        """
        Generates the real distribution over a sequence of points

        Args:
            x_vec (numpy array or list): the points at which the distribution is generated.
                If None, defaults to 101 points between -pi and pi.
        Returns:
            y_vecs (numpy array): the value of each distribution at the given x points.
                y_vecs[j,k] = f_j(x_vec[k])
        """
        # Set default x vector
        if x_vec is None:
            x_vec = numpy.linspace(-pi, pi, 101)

        # Init storage to return
        y_vecs = []

        for j in range(self._num_vectors):

            # Get vector and evaluate corresponding function
            # at the chosen set of points
            vector = self._fourier_vectors[:, j]
            y_vec = [self._make_dist(vector, theta) for theta in x_vec]

            # Store and plot data
            y_vecs.append(y_vec)

        return numpy.array(y_vecs)

    def _holevo_centers(self, vectors=None):
        """
        Gets the Holevo average phase from a distribution.
        Args:
            vectors (numpy array): vectors to calculate the average of. If None,
                defaults to the vectors stored in the class.
        Returns:
            centers (numpy array): the corresponding average angle of the
                distribution.
        """
        if vectors is None:
            vectors = self._fourier_vectors

        centers = numpy.angle(vectors[2, :] + 1j*vectors[1, :])
        return centers

    def _holevo_variances(self, vectors=None, maxvar=4*numpy.pi**2/12):
        """
        Gets the Holevo variance from a distribution

        Args:
            vectors (numpy array): Fourier vectors to calculate the variance of.
                If none, defaults to the vectors stored in the class.
            maxvar (float): The maximum allowed variance. Defaults to the
                maximum of a uniform distribution on [-pi,pi].
        Returns:
            variances (list): The Holevo variances of each vector.
        """
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

    def _make_dist(self, vector, angle):
        """
        converts a frequency vector into a function evaluated at theta
        Args:
            vector (numpy array): a fourier vector P(phi)
            angle (float): the angle to evaluate P at

        Returns:
            (float) P(angle)
        """
        res = vector[0]
        for j in range(1, self._num_freqs):
            res += vector[2*j-1] * sin(j*angle)
            res += vector[2*j] * cos(j*angle)
        return res / (2*pi)

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
        # Recalculates the Jacobian based on all previously
        # calculated p_vecs.
        self._jacobian = sum(
            [self._jacobian_term(
                p_vec,
                self._amplitude_estimates)
             for p_vec in self._p_vecs]) + self._jinit_mlikelihood()

    def _init_mlikelihood(self, a_vec):
        # A normal distributed initial guess at the amplitudes
        return numpy.dot(
            0.5*(a_vec-self._init_amplitude_guess)[
                numpy.newaxis, :],
            numpy.dot(self._inv_covar_mat,
                      (a_vec-self._init_amplitude_guess)[
                        :, numpy.newaxis])).item()

    def _dinit_mlikelihood(self, a_vec):
        # The derivative of the above likelihood function
        return numpy.dot(
            self._inv_covar_mat,
            (a_vec-self._init_amplitude_guess))

    def _jinit_mlikelihood(self):
        # The jacobian of the above likelihood function
        return self._inv_covar_mat


def make_fourier_matrices(max_n, num_freqs):
    """
    Function to make the matrices that define multiplication
    by sin(theta) and cos(theta). We store these as sparse matrices
    in CSR format, and initialize them in (row,col,data) format.
    We assume here that self.max_n is less than self.num_freqs.
    """

    # matrices[n][a] contains matrix M^a(n+1)
    # following arXiv:1809.09697, app.C
    matrices = []

    for index in range(1, max_n+1):

        # We generate the matrices ourselves in
        # COO format, and then create a
        # CSR matrix directly

        cos_matrix, sin_matrix = make_fourier_matrix(index, num_freqs)

        # Append to list
        matrices.append([cos_matrix, sin_matrix])
    return matrices

def make_fourier_matrix(nrot, num_freqs):
    """
    Makes matrices that sends a Fourier representation of p(phi)
    to the Fourier representation of cos^2(n*phi/2+beta/2)*p(phi).

    See App. C in arXiv:1809.09697

    Args:
        nrot (n): the number of rotations n in the above formula.
    Returns:
        cos_matrix (scipy.sparse.csr_matrix) matrix for multiplication
            by cos(phi)
        sin_matrix (scipy.sparse.csr_matrix) matrix for multiplication
            by sin(phi)
    """

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
    for j in range(nrot+1, num_freqs+1):

        data_cos += [1, 1]
        row_ind_cos += [2*j-2*nrot-1, 2*j-2*nrot]
        col_ind_cos += [2*j-1, 2*j]

        data_sin += [-1, 1]
        row_ind_sin += [2*j-2*nrot, 2*j-2*nrot-1]
        col_ind_sin += [2*j-1, 2*j]

        if j + nrot <= num_freqs:

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
