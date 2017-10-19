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
Python's 'unittest' is run with either:
* python {this file's name}.py
* python -m unittest discover simple_example  # (for "discovering"
		all test files)
'''




import unittest
import os
import cStringIO

from openfermion.ops import QubitOperator
#import openfermion.utils._trotter_qasm_conversion as tqc
from openfermion.utils._trotter_qasm_conversion import *
#from _trotter_qasm_conversion import *


class TrottQasmTest(unittest.TestCase):

	def setUp(self):
		# First qubit operator example
		self.qo1 = QubitOperator('X0 Z1 Y3', 0.5) + 0.6 * QubitOperator('Z3 Z4')


	def test_simple1_trot1(self):


		# strioQasm = cStringIO.StringIO()
		strioQasm = "test.txt"
		print_to_qasm(strioQasm,self.qo1)

		# print strioQasm.getvalue()

		strcorrect = \
'''5
# ***
H 0
Rx 1.57079632679 3
CNOT 0 1
CNOT 1 3
Rz 0.5 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.57079632679 3
CNOT 3 4
Rz 0.6 4
CNOT 3 4
'''
		myfile = open('test.txt')
		data = myfile.read()
		self.assertEqual(data,strcorrect)
		myfile.close()
		try:
			os.remove('test.txt')
		except OSError:
			pass


	def test_simple1_trot2(self):

		strioQasm = 'test.txt'

		# Two differences from prev example: trotter_steps and k_exp
		print_to_qasm(strioQasm,self.qo1,trotter_steps=2, k_exp=0.1)

		strcorrect = \
'''5
# ***
H 0
Rx 1.57079632679 3
CNOT 0 1
CNOT 1 3
Rz 0.025 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.57079632679 3
CNOT 3 4
Rz 0.03 4
CNOT 3 4
H 0
Rx 1.57079632679 3
CNOT 0 1
CNOT 1 3
Rz 0.025 3
CNOT 1 3
CNOT 0 1
H 0
Rx -1.57079632679 3
CNOT 3 4
Rz 0.03 4
CNOT 3 4
'''

		myfile = open('test.txt')
		data = myfile.read()
		self.assertEqual(data,strcorrect)
		myfile.close()		
		try:
			os.remove('test.txt')
		except OSError:
			pass



	def test_order2_slice1(self):

		strioQasm = 'test.txt'

		# Second-order trotterization
		print_to_qasm(strioQasm,self.qo1,trotter_order=2,trotter_steps=1)


		myfile = open('test.txt')
		data = myfile.read()


		strcorrect = \
'''5
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

		self.assertEqual(data,strcorrect)
		myfile.close()		
		try:
			os.remove('test.txt')
		except OSError:
			pass





if __name__ == '__main__':
    unittest.main()









