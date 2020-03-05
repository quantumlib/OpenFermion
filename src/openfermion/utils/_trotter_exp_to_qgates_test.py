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
"""Tests for _trotter_exp_to_qgates.py"""

import unittest

from openfermion.utils._trotter_exp_to_qgates import *
from openfermion.utils._trotter_exp_to_qgates import (
    _third_order_trotter_helper)
from openfermion.utils import count_qubits


class TrottQasmTest(unittest.TestCase):

    def setUp(self):
        # First qubit operator example
        self.opA = QubitOperator('X0 Z1 Y3', 0.5)
        self.opB = 0.6 * QubitOperator('Z3 Z4')
        self.op_id = QubitOperator('', 1.0)
        self.qo1 = self.opA + self.opB

    def test_exceptions(self):
        # Test exceptions in trotter_operator_grouping()

        with self.assertRaises(ValueError):
            _ = [
                i for i in trotter_operator_grouping(self.qo1, trotter_order=0)
            ]

        with self.assertRaises(ValueError):
            _ = [
                i for i in trotter_operator_grouping(self.qo1, trotter_order=4)
            ]

        with self.assertRaises(TypeError):
            _ = [i for i in trotter_operator_grouping(42)]

        emptyQO = QubitOperator()
        with self.assertRaises(TypeError):
            _ = [i for i in trotter_operator_grouping(emptyQO)]

        emptyTO = []
        with self.assertRaises(TypeError):
            _ = [
                i for i in trotter_operator_grouping(self.qo1,
                                                     term_ordering=emptyTO)
            ]

        # Too few ops for 2nd-order
        with self.assertRaises(ValueError):
            _ = [
                i for i in trotter_operator_grouping(self.opA, trotter_order=2)
            ]

        # Too few ops for 3rd-order
        with self.assertRaises(ValueError):
            _ = [
                i for i in trotter_operator_grouping(self.opA, trotter_order=3)
            ]

    def compare_qubop_lists(self, gold, res):
        # Compare lists of operators. Used in most test functions.
        for a, b in zip(gold, res):
            self.assertEqual(len(a.terms), 1)
            self.assertEqual(len(b.terms), 1)
            for key in set(a.terms.keys()).union(b.terms.keys()):
                self.assertEqual(key in a.terms, key in b.terms)
                self.assertAlmostEqual(a.terms[key], b.terms[key])

    def test_3rd_order_helper_2ops(self):
        # Test 3rd-order helper, H=A+B
        op_a = QubitOperator('X0 Z1 Y2', 0.1)
        op_b = QubitOperator('Z2', 0.1)

        gold = []
        gold.append(op_a * (7./24))
        gold.append(op_b * (2./3))
        gold.append(op_a * (3./4))
        gold.append(op_b * (-2./3))
        gold.append(op_a * (-1./24))
        gold.append(op_b * (1.))

        # Second arg must be in list form
        res = _third_order_trotter_helper([op_a, op_b])

        # Assert each term in list of QubitOperators is correct
        self.compare_qubop_lists(gold, res)

    def test_3rd_order_helper_3ops(self):
        # Test 3rd-order helper, H=A+B+C
        op_a = QubitOperator('X0', 1.)
        op_b = QubitOperator('Z2', 1.)
        op_c = QubitOperator('Z3', 1.)

        gold = []
        gold.append(op_a * 7./24)
        gold.append(op_b * 7./36)
        gold.append(op_c * 4./9)
        gold.append(op_b * 1./2)
        gold.append(op_c * -4./9)
        gold.append(op_b * -1./36)
        gold.append(op_c * 2./3)
        gold.append(op_a * 3./4)
        gold.append(op_b * -7./36)
        gold.append(op_c * -4./9)
        gold.append(op_b * -1./2)
        gold.append(op_c * 4./9)
        gold.append(op_b * 1./36)
        gold.append(op_c * -2./3)
        gold.append(op_a * -1./24)
        gold.append(op_b * 7./24)
        gold.append(op_c * 2./3)
        gold.append(op_b * 3./4)
        gold.append(op_c * -2./3)
        gold.append(op_b * -1./24)
        gold.append(op_c * 1.0)

        # Second arg must be in list form
        res = _third_order_trotter_helper([op_a, op_b, op_c])

        # Assert each term in list of QubitOperators is correct
        self.compare_qubop_lists(gold, res)

    def test_trott_ordering_3rd_ord(self):
        # Test 3rd-order Trotterization, H=A+B+C
        op_a = QubitOperator('X0', 1.)
        op_b = QubitOperator('Z2', 1.)
        op_c = QubitOperator('Z3', 1.)
        ham = op_a + op_b + op_c

        # Result from code
        res = [op for op in trotter_operator_grouping(ham, trotter_order=3)]

        gold = []
        gold.append(op_a * 7./24)
        gold.append(op_b * 7./36)
        gold.append(op_c * 4./9)
        gold.append(op_b * 1./2)
        gold.append(op_c * -4./9)
        gold.append(op_b * -1./36)
        gold.append(op_c * 2./3)
        gold.append(op_a * 3./4)
        gold.append(op_b * -7./36)
        gold.append(op_c * -4./9)
        gold.append(op_b * -1./2)
        gold.append(op_c * 4./9)
        gold.append(op_b * 1./36)
        gold.append(op_c * -2./3)
        gold.append(op_a * -1./24)
        gold.append(op_b * 7./24)
        gold.append(op_c * 2./3)
        gold.append(op_b * 3./4)
        gold.append(op_c * -2./3)
        gold.append(op_b * -1./24)
        gold.append(op_c * 1.0)

        # Assert each term in list of QubitOperators is correct
        self.compare_qubop_lists(gold, res)

    def test_trott_ordering_2nd_ord(self):
        # Test 2nd-order Trotter ordering
        op_a = QubitOperator('X0', 1.)
        op_b = QubitOperator('Z2', 1.)
        op_c = QubitOperator('Z3', 1.)
        ham = op_a + op_b + op_c
        _ = [op for op in trotter_operator_grouping(ham, trotter_order=2)]
        gold = []
        gold.append(op_a * 0.5)
        gold.append(op_b * 0.5)
        gold.append(op_c * 1.0)
        gold.append(op_b * 0.5)
        gold.append(op_a * 0.5)

    def test_get_trott_qubops(self):
        # Testing with trotter number of 2 (first-order)
        res = [op for op in trotter_operator_grouping(
            self.qo1, trotter_number=2, trotter_order=1, term_ordering=None,
            k_exp=1.0)]

        gold = []
        gold.append(0.5 * self.opA)
        gold.append(0.5 * self.opB)
        gold.append(0.5 * self.opA)
        gold.append(0.5 * self.opB)

        # Assert each term in list of QubitOperators is correct
        self.compare_qubop_lists(gold, res)

    def test_qasm_string_Z(self):
        # Testing for correct QASM string output w/ Pauli-X

        # Number of qubits
        qasmstr = str(count_qubits(self.opB)) + "\n"

        # Write each QASM operation
        qasmstr += "\n".join(trotterize_exp_qubop_to_qasm(self.opB))

        # Correct string
        strcorrect = '''5
CNOT 3 4
Rz 0.6 4
CNOT 3 4'''

        self.assertEqual(qasmstr, strcorrect)

    def test_qasm_string_XYZ(self):
        # Testing for correct QASM string output w/ Pauli-{X,Y,Z}
        # QubitOperator('X0 Z1 Y3', 0.5)

        # Number of qubits
        qasmstr = str(count_qubits(self.opA)) + "\n"

        # Write each QASM operation
        qasmstr += "\n".join(trotterize_exp_qubop_to_qasm(self.opA))

        # Correct string
        strcorrect = '''4
H 0
Rx 1.5707963267948966 3
CNOT 0 1
CNOT 1 3
Rz 0.5 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.5707963267948966 3'''

        self.assertEqual(qasmstr, strcorrect)

    def test_qasm_string_Controlled_XYZ(self):
        # Testing for correct QASM string output w/ Pauli-{X,Y,Z}
        # QubitOperator('X0 Z1 Y3', 0.5) and a controlled ancilla

        # Number of qubits
        qasmstr = str(count_qubits(self.opA)+1) + "\n"

        # Write each QASM operation
        qasmstr += "\n".join(
            trotterize_exp_qubop_to_qasm(self.opA, ancilla='ancilla'))

        # Correct string
        strcorrect = '''5
H 0
Rx 1.5707963267948966 3
CNOT 0 1
CNOT 1 3
C-Phase -1.0 ancilla 3
Rz 0.5 ancilla
CNOT 1 3
CNOT 0 1
H 0
Rx -1.5707963267948966 3'''

        self.assertEqual(qasmstr, strcorrect)

    def test_qasm_string_Controlled_XYZ_ancilla_list(self):
        # Testing for correct QASM string output w/ Pauli-{X,Y,Z}
        # QubitOperator('X0 Z1 Y3', 0.5) and a controlled ancilla

        # Number of qubits
        qasmstr = str(count_qubits(self.opA)+1) + "\n"

        qubit_list = ['q0', 'q1', 'q2', 'q3']

        # Write each QASM operation
        qasmstr += "\n".join(
            trotterize_exp_qubop_to_qasm(self.opA, ancilla='ancilla',
                                         qubit_list=qubit_list))

        # Correct string
        strcorrect = '''5
H q0
Rx 1.5707963267948966 q3
CNOT q0 q1
CNOT q1 q3
C-Phase -1.0 ancilla q3
Rz 0.5 ancilla
CNOT q1 q3
CNOT q0 q1
H q0
Rx -1.5707963267948966 q3'''

        self.assertEqual(qasmstr, strcorrect)

    def test_qasm_string_controlled_identity(self):
        qasmstr = str(count_qubits(self.op_id)+1) + "\n"

        # Write each QASM operation
        qasmstr += "\n".join(
            trotterize_exp_qubop_to_qasm(self.op_id, ancilla='ancilla'))

        strcorrect = '''1
Rz 1.0 ancilla'''

        self.assertEqual(qasmstr, strcorrect)

    def test_qasm_string_identity(self):
        qasmstr = str(count_qubits(self.op_id)+1) + "\n"

        # Write each QASM operation
        qasmstr += "\n".join(
            trotterize_exp_qubop_to_qasm(self.op_id))

        strcorrect = '''1\n'''

        self.assertEqual(qasmstr, strcorrect)

    def test_qasm_string_multiple_operator(self):
        # Testing for correct QASM string output
        # w/ sum of Pauli-{X,Y,Z} and Pauli-{Z}
        # QubitOperator('X0 Z1 Y3', 0.5) + QubitOperator('Z3 Z4', 0.6)

        # Number of qubits
        qasmstr = str(count_qubits(self.qo1)) + "\n"

        # Write each QASM operation
        qasmstr += "\n".join(trotterize_exp_qubop_to_qasm(self.qo1))

        # Two possibilities for the correct string depending on
        # the order in which the operators loop.
        strcorrect1 = '''5
CNOT 3 4
Rz 0.6 4
CNOT 3 4
H 0
Rx 1.5707963267948966 3
CNOT 0 1
CNOT 1 3
Rz 0.5 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.5707963267948966 3'''

        strcorrect2 = '''5
H 0
Rx 1.5707963267948966 3
CNOT 0 1
CNOT 1 3
Rz 0.5 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.5707963267948966 3
CNOT 3 4
Rz 0.6 4
CNOT 3 4'''
        try:
            self.assertEqual(qasmstr, strcorrect1)
        except:
            self.assertEqual(qasmstr, strcorrect2)
