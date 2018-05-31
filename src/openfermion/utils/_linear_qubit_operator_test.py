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

    def setUp(self):
        """LinearQubitOperator test set up."""
        self.qubit_operator = QubitOperator('Z2')
        self.n_qubits = 3
        self.linear_operator = LinearQubitOperator(self.qubit_operator)

    def test_init(self):
        """Tests __init__()."""
        self.assertEqual(self.linear_operator.qubit_operator,
                         self.qubit_operator)
        self.assertEqual(self.linear_operator.n_qubits, self.n_qubits)

        # Checks type.
        self.assertTrue(isinstance(self.linear_operator,
                                   scipy.sparse.linalg.LinearOperator))

    def test_matvec(self):
        """Tests _matvec() for matrix multiplication with a vector."""
        vec = numpy.array(range(2 ** self.n_qubits))
        expected_matvec = numpy.array([0, -1, 2, -3, 4, -5, 6, -7])

        self.assertTrue(numpy.allclose(self.linear_operator * vec,
                                       expected_matvec))


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
