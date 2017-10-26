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
import os
import cStringIO

from openfermion.ops import QubitOperator
from _trotter_exp_to_qgates import *


class TrottQasmTest(unittest.TestCase):

    def setUp(self):
        # First qubit operator example
        self.opA = QubitOperator('X0 Z1 Y3', 0.5)
        self.opB = 0.6 * QubitOperator('Z3 Z4')
        self.qo1 = self.opA + self.opB

    def compare_qubop_lists(self, gold, res):
        for a, b in zip(gold, res):
            self.assertEqual(len(a.terms), 1)
            self.assertEqual(len(b.terms), 1)
            self.assertEqual(a.terms.keys()[0], b.terms.keys()[0])
            self.assertAlmostEqual(a.terms.values()[0], b.terms.values()[0])

    def test_3rd_order_helper_2ops(self):
        # Test 3rd-order helper, H=A+B
        op_a = QubitOperator('X0 Z1 Y2', 0.1)
        op_b = QubitOperator('Z2', 0.1)
        a = 'X0 Z1 Y2'
        b = 'Z2'

        gold = []
        gold.append(op_a * (7./24))
        gold.append(op_b * (2./3))
        gold.append(op_a * (3./4))
        gold.append(op_b * (-2./3))
        gold.append(op_a * (-1./24))
        gold.append(op_b * (1.))

        # Second arg must be in list form
        res = third_order_trotter_helper(op_a, [op_b])

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
        res = third_order_trotter_helper(op_a, [op_b, op_c])

        # Assert each term in list of QubitOperators is correct
        self.compare_qubop_lists(gold, res)

    def test_get_trott_qubops(self):
        # Testing the Trotter decomposition function
        res = get_trotterized_qubops(
                self.qo1,
                trotter_number=2,
                trotter_order=1,
                term_ordering=None,
                k_exp=None)

        gold = []
        gold.append(0.5*self.opA)
        gold.append(0.5*self.opB)
        gold.append(0.5*self.opA)
        gold.append(0.5*self.opB)

        # Assert each term in list of QubitOperators is correct
        self.compare_qubop_lists(gold, res)

    def test_sgl_pauli_exp(self):
        # Writes out quantum gates for exponentiation of single
        # Pauli-string.

        op = QubitOperator('X0 Z1 Y3', 0.5)

        strioQasm = cStringIO.StringIO()

        exp_sgl_pauli_string_to_qasm(strioQasm, op)

        strcorrect = '''H 0
Rx 1.57079632679 3
CNOT 0 1
CNOT 1 3
Rz 0.5 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.57079632679 3
'''
        self.assertEqual(strioQasm.getvalue(), strcorrect)

    def test_complete_to_qasm(self):
        # Test of highest-level function in module
        strioQasm = cStringIO.StringIO()

        trotterize_exp_qubop_to_qasm(
                strioQasm,
                self.qo1,
                trotter_number=1,
                trotter_order=2,
                term_ordering=None,
                k_exp=None)

        strcorrect = '''5
# ***
H 0
Rx 1.57079632679 3
CNOT 0 1
CNOT 1 3
Rz 0.25 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.57079632679 3
CNOT 3 4
Rz 0.6 4
CNOT 3 4
H 0
Rx 1.57079632679 3
CNOT 0 1
CNOT 1 3
Rz 0.25 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.57079632679 3
'''
        self.assertEqual(strioQasm.getvalue(), strcorrect)

if __name__ == '__main__':
    unittest.main()
