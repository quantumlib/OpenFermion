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

"""Tests for linear_qubit_operator.py."""

import multiprocessing
import unittest

import numpy
import scipy.sparse.linalg

from openfermion.ops import QubitOperator
from openfermion.utils._linear_qubit_operator import (
    LinearQubitOperator,
    LinearQubitOperatorOptions,
    ParallelLinearQubitOperator,
    apply_operator,
    generate_linear_qubit_operator,
)
from openfermion.utils._sparse_tools import qubit_operator_sparse

class LinearQubitOperatorOptionsTest(unittest.TestCase):
    """Tests for LinearQubitOperatorOptions class."""

    def setUp(self):
        """LinearQubitOperatorOptions test set up."""
        self.processes = multiprocessing.cpu_count()
        self.options = LinearQubitOperatorOptions(self.processes)

    def test_init(self):
        """Tests __init__()."""
        self.assertEqual(self.options.processes, self.processes)
        self.assertIsNone(self.options.pool)

    def test_get_processes_small(self):
        """Tests get_processes() with a small num."""
        num = 1
        self.assertEqual(self.options.get_processes(num), num)

    def test_get_processes_large(self):
        """Tests get_processes() with a large num."""
        self.assertEqual(self.options.get_processes(2*self.processes),
                         self.processes)

    def test_invalid_processes(self):
        """Tests with invalid processes since it's not positive."""
        with self.assertRaises(ValueError):
            LinearQubitOperatorOptions(0)

    def test_get_pool(self):
        """Tests get_pool() without a num."""
        self.assertIsNone(self.options.pool)

        pool = self.options.get_pool()
        self.assertIsNotNone(pool)

    def test_get_pool_with_num(self):
        """Tests get_processes() with a num."""
        self.assertIsNone(self.options.pool)

        pool = self.options.get_pool(2)
        self.assertIsNotNone(pool)

class LinearQubitOperatorTest(unittest.TestCase):
    """Tests for LinearQubitOperator class."""

    def test_init(self):
        """Tests __init__()."""
        qubit_operator = QubitOperator('Z2')
        n_qubits = 3
        linear_operator = LinearQubitOperator(qubit_operator)

        self.assertEqual(linear_operator.qubit_operator, qubit_operator)
        self.assertEqual(linear_operator.n_qubits, n_qubits)

        # Checks type.
        self.assertTrue(isinstance(linear_operator,
                                   scipy.sparse.linalg.LinearOperator))

    def test_matvec_wrong_n(self):
        """Testing with wrong n_qubits."""
        with self.assertRaises(ValueError):
            LinearQubitOperator(QubitOperator('X3'), 1)

    def test_matvec_wrong_vec_length(self):
        """Testing with wrong vector length."""
        with self.assertRaises(ValueError):
            _ = LinearQubitOperator(QubitOperator('X3')) * numpy.zeros(4)

    def test_matvec_0(self):
        """Testing with zero term."""
        qubit_operator = QubitOperator.zero()

        vec = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
        matvec_expected = numpy.zeros(vec.shape)

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(qubit_operator, 3) * vec, matvec_expected))

    def test_matvec_x(self):
        """Testing product with X."""
        vec = numpy.array([1, 2, 3, 4])
        matvec_expected = numpy.array([2, 1, 4, 3])

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('X1')) * vec,
            matvec_expected))

    def test_matvec_y(self):
        """Testing product with Y."""
        vec = numpy.array([1, 2, 3, 4], dtype=complex)
        matvec_expected = 1.0j * numpy.array([-2, 1, -4, 3], dtype=complex)

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('Y1')) * vec,
            matvec_expected))

    def test_matvec_z(self):
        """Testing product with Z."""
        vec = numpy.array([1, 2, 3, 4])
        matvec_expected = numpy.array([1, 2, -3, -4])

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('Z0'), 2) * vec,
            matvec_expected))

    def test_matvec_z3(self):
        """Testing product with Z^n."""
        vec = numpy.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        matvec_expected = numpy.array(
            [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16])

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('Z3')) * vec,
            matvec_expected))

    def test_matvec_zx(self):
        """Testing with multiple factors."""
        vec = numpy.array([1, 2, 3, 4])
        matvec_expected = numpy.array([2, 1, -4, -3])

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('Z0 X1')) * vec,
            matvec_expected))

    def test_matvec_multiple_terms(self):
        """Testing with multiple terms."""
        qubit_operator = (QubitOperator.identity() + 2 * QubitOperator('Y2') +
                          QubitOperator(((0, 'Z'), (1, 'X')), 10.0))

        vec = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
        matvec_expected = (10 * numpy.array([3, 4, 1, 2, -7, -8, -5, -6]) +
                           2j * numpy.array([-2, 1, -4, 3, -6, 5, -8, 7]) + vec)

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(qubit_operator) * vec, matvec_expected))

    def test_matvec_compare(self):
        """Compare LinearQubitOperator with qubit_operator_sparse."""
        qubit_operator = QubitOperator('X0 Y1 Z3')
        mat_expected = qubit_operator_sparse(qubit_operator)

        self.assertTrue(numpy.allclose(numpy.transpose(
            numpy.array([LinearQubitOperator(qubit_operator) * v
                         for v in numpy.identity(16)])),
                                       mat_expected.A))

class ParallelLinearQubitOperatorTest(unittest.TestCase):
    """Tests for ParallelLinearQubitOperator class."""

    def setUp(self):
        """ParallelLinearQubitOperator test set up."""
        self.qubit_operator = (QubitOperator('Z3') + QubitOperator('Y0') +
                               QubitOperator('X1'))
        self.n_qubits = 4
        self.linear_operator = ParallelLinearQubitOperator(self.qubit_operator)

        # Vectors for calculations.
        self.vec = numpy.array(range(2 ** self.n_qubits))

        expected_matvec = numpy.array([
            0, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15,
        ])
        expected_matvec = expected_matvec + numpy.array([
            -8j, -9j, -10j, -11j, -12j, -13j, -14j, -15j,
            0j, 1j, 2j, 3j, 4j, 5j, 6j, 7j,
        ])
        expected_matvec += numpy.array([
            4, 5, 6, 7, 0, 1, 2, 3,
            12, 13, 14, 15, 8, 9, 10, 11,
        ])
        self.expected_matvec = expected_matvec

    def test_init(self):
        """Tests __init__()."""
        self.assertEqual(self.linear_operator.qubit_operator,
                         self.qubit_operator)
        self.assertEqual(self.linear_operator.n_qubits, self.n_qubits)
        self.assertIsNone(self.linear_operator.options.pool)

        cpu_count = multiprocessing.cpu_count()
        default_processes = min(cpu_count, 10)
        self.assertEqual(self.linear_operator.options.processes,
                         default_processes)

        # Generated variables.
        self.assertEqual(len(self.linear_operator.qubit_operator_groups),
                         min(multiprocessing.cpu_count(), 3))
        self.assertEqual(QubitOperator.accumulate(
            self.linear_operator.qubit_operator_groups), self.qubit_operator)

        for linear_operator in self.linear_operator.linear_operators:
            self.assertEqual(linear_operator.n_qubits, self.n_qubits)
            self.assertTrue(isinstance(linear_operator,
                                       LinearQubitOperator))

        # Checks type.
        self.assertTrue(isinstance(self.linear_operator,
                                   scipy.sparse.linalg.LinearOperator))


    def test_matvec(self):
        """Tests _matvec() for matrix multiplication with a vector."""

        self.assertIsNone(self.linear_operator.options.pool)
        self.assertTrue(numpy.allclose(self.linear_operator * self.vec,
                                       self.expected_matvec))

    def test_matvec_0(self):
        """Testing with zero term."""
        qubit_operator = QubitOperator.zero()

        vec = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
        matvec_expected = numpy.zeros(vec.shape)

        self.assertTrue(numpy.allclose(
            ParallelLinearQubitOperator(qubit_operator, 3) * vec,
            matvec_expected))
        self.assertIsNone(self.linear_operator.options.pool)

    def test_closed_workers_not_reused(self):
        qubit_operator = QubitOperator('X0')
        parallel_qubit_op = ParallelLinearQubitOperator(qubit_operator, 1,
                options=LinearQubitOperatorOptions(processes=2))
        state = [1.0, 0.0]
        parallel_qubit_op.dot(state)
        parallel_qubit_op.dot(state)
        self.assertIsNone(parallel_qubit_op.options.pool)

class UtilityFunctionTest(unittest.TestCase):
    """Tests for utility functions."""

    def test_apply_operator(self):
        """Tests apply_operator() since it's executed on other processors."""
        vec = numpy.array([1, 2, 3, 4])
        matvec_expected = numpy.array([2, 1, 4, 3])

        self.assertTrue(numpy.allclose(
            apply_operator((LinearQubitOperator(QubitOperator('X1')), vec)),
            matvec_expected))

    def test_generate_linear_operator(self):
        """Tests generate_linear_qubit_operator()."""
        qubit_operator = (QubitOperator('Z3') + QubitOperator('X1') +
                          QubitOperator('Y0'))
        n_qubits = 6

        # Checks types.
        operator = generate_linear_qubit_operator(qubit_operator, n_qubits)
        self.assertTrue(isinstance(operator, LinearQubitOperator))
        self.assertFalse(isinstance(operator, ParallelLinearQubitOperator))

        operator_again = generate_linear_qubit_operator(
            qubit_operator, n_qubits, options=LinearQubitOperatorOptions(2))
        self.assertTrue(isinstance(operator_again, ParallelLinearQubitOperator))
        self.assertFalse(isinstance(operator_again, LinearQubitOperator))

        # Checks operators are equivalent.
        numpy.random.seed(n_qubits)
        vec = numpy.random.rand(2 ** n_qubits, 1)
        self.assertTrue(numpy.allclose(operator * vec, operator_again * vec))
