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
import scipy.sparse
import scipy.sparse.linalg

from openfermion.utils.operator_utils import count_qubits
from openfermion.config import get_available_cpu_count


def _bit_parity(values):
    """Returns the parity of the population count of each uint64 value."""
    values = values ^ (values >> numpy.uint64(32))
    values = values ^ (values >> numpy.uint64(16))
    values = values ^ (values >> numpy.uint64(8))
    values = values ^ (values >> numpy.uint64(4))
    values = values ^ (values >> numpy.uint64(2))
    values = values ^ (values >> numpy.uint64(1))
    return values & numpy.uint64(1)


class LinearQubitOperatorOptions:
    """Options for LinearQubitOperator."""

    def __init__(self, processes=10, pool=None):
        """
        Args:
            processes(int): Number of processors to use.
            pool(multiprocessing.Pool): A pool of workers.
        """
        if processes <= 0:
            raise ValueError('Invalid number of processors specified {} <= 0'.format(processes))

        self.processes = min(processes, get_available_cpu_count())
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

    Each term of the QubitOperator is a tensor product of Pauli operators and
    acts on an amplitude vector as a signed permutation. Qubit q corresponds to
    bit (n_qubits - 1 - q) of the amplitude index. The qubits carrying an X or Y
    flip that bit, so the amplitude at index c moves to index c ^ x_mask, where
    x_mask collects those bits. The qubits carrying a Y or Z multiply the
    amplitude by -1 whenever the matching index bit is set, and every Y adds a
    further factor of 1j. Applying a term is therefore a reindexing of the vector
    together with these per-index signs, accumulated over all terms."""

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
            raise ValueError(
                'Invalid number of qubits specified '
                '{} < {}.'.format(n_qubits, calculated_n_qubits)
            )

        n_hilbert = 2**n_qubits
        super().__init__(shape=(n_hilbert, n_hilbert), dtype=complex)
        self.qubit_operator = qubit_operator
        self.n_qubits = n_qubits

    def _matvec(self, x):
        """Matrix-vector multiplication for the LinearQubitOperator class.

        Args:
          x(numpy.ndarray): 1D numpy array.

        Returns:
          retvec(numpy.ndarray): same to the shape of input vector of x.
        """
        arr = numpy.asarray(x)
        vec = arr.reshape(-1)
        indices = None
        retvec = numpy.zeros(vec.size, dtype=complex)

        for qubit_term, coefficient in self.qubit_operator.terms.items():
            x_mask = 0
            z_mask = 0
            y_count = 0
            for qubit, action in qubit_term:
                bit = 1 << (self.n_qubits - 1 - qubit)
                if action == 'X':
                    x_mask ^= bit
                elif action == 'Y':
                    x_mask ^= bit
                    z_mask ^= bit
                    y_count += 1
                else:
                    z_mask ^= bit

            amplitudes = coefficient * (1, 1j, -1, -1j)[y_count % 4] * vec
            if z_mask or x_mask:
                if indices is None:
                    indices = numpy.arange(vec.size, dtype=numpy.uint64)
            if z_mask:
                signs = 1 - 2 * _bit_parity(indices & numpy.uint64(z_mask)).astype(numpy.int8)
                amplitudes = signs * amplitudes
            if x_mask:
                retvec[indices ^ numpy.uint64(x_mask)] += amplitudes
            else:
                retvec += amplitudes

        return retvec.reshape(arr.shape)


class ParallelLinearQubitOperator(scipy.sparse.linalg.LinearOperator):
    """A LinearOperator from a QubitOperator with multiple processors."""

    _start_method_set = False

    def __init__(self, qubit_operator, n_qubits=None, options=None):
        """
        Args:
            qubit_operator(QubitOperator): A qubit operator to be applied on
                vectors.
            n_qubits(int): The total number of qubits
            options(LinearQubitOperatorOptions): Options for the LinearOperator.
        """
        n_qubits = n_qubits or count_qubits(qubit_operator)
        n_hilbert = 2**n_qubits
        super().__init__(shape=(n_hilbert, n_hilbert), dtype=complex)

        self.qubit_operator = qubit_operator
        self.n_qubits = n_qubits
        self.options = options or LinearQubitOperatorOptions()

        if not ParallelLinearQubitOperator._start_method_set:
            multiprocessing.set_start_method('forkserver', force=True)
            ParallelLinearQubitOperator._start_method_set = True

        self.qubit_operator_groups = list(
            qubit_operator.get_operator_groups(self.options.processes)
        )
        self.linear_operators = [
            LinearQubitOperator(operator, n_qubits) for operator in self.qubit_operator_groups
        ]

    def _matvec(self, x):
        """Matrix-vector multiplication for the LinearQubitOperator class.

        Args:
          x(numpy.ndarray): 1D numpy array.

        Returns:
          retvec(numpy.ndarray): same to the shape of input vector of x.
        """
        if not self.linear_operators:
            return numpy.zeros(x.shape)

        if self.options.processes <= 1:
            return functools.reduce(numpy.add, (operator * x for operator in self.linear_operators))

        pool = self.options.get_pool(len(self.linear_operators))
        vecs = pool.imap_unordered(
            apply_operator, [(operator, x) for operator in self.linear_operators]
        )
        pool.close()
        pool.join()
        return functools.reduce(numpy.add, vecs)


def apply_operator(args):
    """Helper function to apply operator to a vector."""
    operator, vec = args
    return operator * vec


def generate_linear_qubit_operator(qubit_operator, n_qubits=None, options=None):
    """Generates a LinearOperator from a QubitOperator.

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
        linear_operator = ParallelLinearQubitOperator(qubit_operator, n_qubits, options)
    return linear_operator
