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
from __future__ import absolute_import, division

import unittest
import numpy
import scipy.sparse.linalg

from openfermion.ops import QubitOperator
from openfermion.utils._linear_qubit_operator import (
    ParallelLinearQubitOperator,
    LinearQubitOperatorOptions,
    LinearQubitOperator,
)
from openfermion.utils._sparse_tools import qubit_operator_sparse

class LinearQubitOperatorOptionsTest(unittest.TestCase):
    """Tests for LinearQubitOperatorOptions class."""

    def setUp(self):
        """LinearQubitOperatorOptions test set up."""
        self.processes = 6
        self.options = LinearQubitOperatorOptions(self.processes)

    def test_init(self):
        """Tests __init__()."""
        self.assertEqual(self.options.processes, self.processes)

    def test_get_processes_small(self):
        """Tests get_processes() with a small num."""
        num = 2
        self.assertEqual(self.options.get_processes(num), num)

    def test_get_processes_large(self):
        """Tests get_processes() with a large num."""
        self.assertEqual(self.options.get_processes(20), self.processes)

    def test_invalid_processes(self):
        """Tests with invalid processes since it's not positive."""
        with self.assertRaises(ValueError):
            LinearQubitOperatorOptions(0)

class LinearQubitOperatorTest(unittest.TestCase):
    """Tests for LinearQubitOperator class."""

    def test_init(self):
        """Tests __init__()."""
        self.qubit_operator = QubitOperator('Z2')
        self.n_qubits = 3
        self.linear_operator = LinearQubitOperator(self.qubit_operator)

        self.assertEqual(self.linear_operator.qubit_operator,
                         self.qubit_operator)
        self.assertEqual(self.linear_operator.n_qubits, self.n_qubits)

        # Checks type.
        self.assertTrue(isinstance(self.linear_operator,
                                   scipy.sparse.linalg.LinearOperator))

    def test_matvec_wrong_n(self):
        """Testing with wrong n_qubits."""
        with self.assertRaises(ValueError):
            LinearQubitOperator(QubitOperator('X3'), 1)

    def test_matvec_wrong_vec_length(self):
        """Testing with wrong vector length."""
        with self.assertRaises(ValueError):
            LinearQubitOperator(QubitOperator('X3')) * numpy.zeros(4)

    def test_matvec_0(self):
        """Testing with zero term."""
        qubit_operator = QubitOperator.zero()

        vec = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
        matvec_expected = numpy.zeros(vec.shape)

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(qubit_operator, 3) * vec, matvec_expected))

    def test_matvec_x(self):
        vec = numpy.array([1, 2, 3, 4])
        matvec_expected = numpy.array([2, 1, 4, 3])

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('X1')) * vec,
            matvec_expected))

    def test_matvec_y(self):
        vec = numpy.array([1, 2, 3, 4], dtype=complex)
        matvec_expected = 1.0j * numpy.array([-2, 1, -4, 3], dtype=complex)

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('Y1')) * vec,
            matvec_expected))

    def test_matvec_z(self):
        vec = numpy.array([1, 2, 3, 4])
        matvec_expected = numpy.array([1, 2, -3, -4])

        self.assertTrue(numpy.allclose(
            LinearQubitOperator(QubitOperator('Z0'), 2) * vec,
            matvec_expected))

    def test_matvec_z3(self):
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
        self.assertEqual(self.linear_operator.qubit_operator, self.qubit_operator)
        self.assertEqual(self.linear_operator.n_qubits, self.n_qubits)
        self.assertEqual(self.linear_operator.options.processes, 10)

        # Generated variables.
        self.assertEqual(len(self.linear_operator.qubit_operator_groups), 3)
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

        self.assertTrue(numpy.allclose(self.linear_operator * self.vec,
                                       self.expected_matvec))
