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

"""LinearQubitOperator is a linear operator from QubitOperator."""

import functools
import logging
import multiprocessing

import numpy
import numpy.linalg
import scipy
import scipy.sparse
import scipy.sparse.linalg

from openfermion.utils import count_qubits


class LinearQubitOperatorOptions(object):
    """Options for LinearQubitOperator."""

    def __init__(self, processes=10, pool=None):
        """
        Args:
            processes(int): Number of processors to use.
            pool(multiprocessing.Pool): A pool of workers.
        """
        if processes <= 0:
            raise ValueError('Invalid number of processors specified {} <= 0'
                             .format(processes))

        self.processes = min(processes, multiprocessing.cpu_count())
        self.pool = pool

    def get_processes(self, num):
        """Number of real processes to use."""
        return max(min(num, self.processes), 1)

    def get_pool(self, num=None):
        """Gets a pool of workers to do some parallel work.

        pool will be cached, which implies that one should be very clear how
        many processes one needs, as it's allocated at most once. Subsequent
        calls of get_pool() will reuse the cached pool.

        Args:
            num(int): Number of workers one needs.
        Returns:
            pool(multiprocessing.Pool): A pool of workers.
        """
        processes = self.get_processes(num or self.processes)
        logging.info("Calling multiprocessing.Pool(%d)", processes)
        return multiprocessing.Pool(processes)


class LinearQubitOperator(scipy.sparse.linalg.LinearOperator):
    """A LinearOperator implied from a QubitOperator.

    The idea is that a single i_th qubit operator, O_i, is a 2-by-2 matrix, to
    be applied on a vector of length n_hilbert / 2^i, performs permutations and/
    or adds an extra factor for its first half and the second half, e.g. a `Z`
    operator keeps the first half unchanged, while adds a factor of -1 to the
    second half, while an `I` keeps it both components unchanged.

    Note that the vector length is n_hilbert / 2^i, therefore when one works on
    i monotonically (in increasing order), one keeps splitting the vector to the
    right size and then apply O_i on them independently.

    Also note that operator O_i, is an *envelop operator* for all operators
    after it, i.e. {O_j | j > i}, which implies that starting with i = 0, one
    can split the vector, apply O_i, split the resulting vector (cached) again
    for the next operator."""

    def __init__(self, qubit_operator, n_qubits=None):
        """
        Args:
            qubit_operator(QubitOperator): A qubit operator to be applied on
                vectors.
            n_qubits(int): The total number of qubits
        """
        calculated_n_qubits = count_qubits(qubit_operator)
        if n_qubits is None:
            n_qubits = calculated_n_qubits
        elif n_qubits < calculated_n_qubits:
            raise ValueError('Invalid number of qubits specified '
                             '{} < {}.'.format(n_qubits, calculated_n_qubits))

        n_hilbert = 2 ** n_qubits
        super(LinearQubitOperator, self).__init__(
            shape=(n_hilbert, n_hilbert), dtype=complex)
        self.qubit_operator = qubit_operator
        self.n_qubits = n_qubits

    def _matvec(self, x):
        """Matrix-vector multiplication for the LinearQubitOperator class.

        Args:
          x(numpy.ndarray): 1D numpy array.

        Returns:
          retvec(numpy.ndarray): same to the shape of input vector of x.
        """
        retvec = numpy.zeros(x.shape, dtype=complex)
        # Loop through the terms.
        for qubit_term in self.qubit_operator.terms:
            vecs = [x]
            tensor_factor = 0
            coefficient = self.qubit_operator.terms[qubit_term]

            for pauli_operator in qubit_term:
                # Split vector by half and half for each bit.
                if pauli_operator[0] > tensor_factor:
                    vecs = [v for iter_v in vecs for v in numpy.split(
                        iter_v, 2 ** (pauli_operator[0] - tensor_factor))]

                # Note that this is to make sure that XYZ operations always work
                # on vector pairs.
                vec_pairs = [numpy.split(v, 2) for v in vecs]

                # There is an non-identity op here, transform the vector.
                xyz = {
                    'X' : lambda vps: [[vp[1], vp[0]] for vp in vps],
                    'Y' : lambda vps: [[-1j * vp[1], 1j * vp[0]] for vp in vps],
                    'Z' : lambda vps: [[vp[0], -vp[1]] for vp in vps],
                }
                vecs = [v for vp in xyz[pauli_operator[1]](vec_pairs)
                        for v in vp]
                tensor_factor = pauli_operator[0] + 1

            # No need to check tensor_factor, i.e. to deal with bits left.
            retvec += coefficient * numpy.concatenate(vecs)
        return retvec


class ParallelLinearQubitOperator(scipy.sparse.linalg.LinearOperator):
    """A LinearOperator from a QubitOperator with multiple processors."""

    def __init__(self, qubit_operator, n_qubits=None, options=None):
        """
        Args:
            qubit_operator(QubitOperator): A qubit operator to be applied on
                vectors.
            n_qubits(int): The total number of qubits
            options(LinearQubitOperatorOptions): Options for the LinearOperator.
        """
        n_qubits = n_qubits or count_qubits(qubit_operator)
        n_hilbert = 2 ** n_qubits
        super(ParallelLinearQubitOperator, self).__init__(
            shape=(n_hilbert, n_hilbert), dtype=complex)

        self.qubit_operator = qubit_operator
        self.n_qubits = n_qubits
        self.options = options or LinearQubitOperatorOptions()

        self.qubit_operator_groups = list(qubit_operator.get_operator_groups(
            self.options.processes))
        self.linear_operators = [LinearQubitOperator(operator, n_qubits)
                                 for operator in self.qubit_operator_groups]

    def _matvec(self, x):
        """Matrix-vector multiplication for the LinearQubitOperator class.

        Args:
          x(numpy.ndarray): 1D numpy array.

        Returns:
          retvec(numpy.ndarray): same to the shape of input vector of x.
        """
        if not self.linear_operators:
            return numpy.zeros(x.shape)

        pool = self.options.get_pool(len(self.linear_operators))
        vecs = pool.imap_unordered(apply_operator,
                                   [(operator, x)
                                    for operator in self.linear_operators])
        pool.close()
        pool.join()
        return functools.reduce(numpy.add, vecs)


def apply_operator(args):
    """Helper funtion to apply opeartor to a vector."""
    operator, vec = args
    return operator * vec


def generate_linear_qubit_operator(qubit_operator, n_qubits=None, options=None):
    """ Generates a LinearOperator from a QubitOperator.

    Args:
        qubit_operator(QubitOperator): A qubit operator to be applied on
            vectors.
        n_qubits(int): The total number of qubits
        options(LinearQubitOperatorOptions): Options for the
            ParallelLinearQubitOperator.
    Returns:
        linear_operator(scipy.sparse.linalg.LinearOperator): A linear operator.
    """
    if options is None:
        linear_operator = LinearQubitOperator(qubit_operator, n_qubits)
    else:
        linear_operator = ParallelLinearQubitOperator(
            qubit_operator, n_qubits, options)
    return linear_operator
