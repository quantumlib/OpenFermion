#	Licensed under the Apache License, Version 2.0 (the "License");
#	you may not use this file except in compliance with the License.
#	You may obtain a copy of the License at
#
#		http://www.apache.org/licenses/LICENSE-2.0
#
#	Unless required by applicable law or agreed to in writing, software
#	distributed under the License is distributed on an "AS IS" BASIS,
#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#	See the License for the specific language governing permissions and
#	limitations under the License.

from openfermion.ops import QubitOperator
import numpy as np

'''
Description:
		Approximates the exponential of an operator composed of Pauli
		matrices, by Trotterization. Outputs QASM format to a python
		stream.

		Change-of-basis gates:
		H			: Z to X basis (and back)
		Rx(pi/2)	: Z to Y basis ..... 
		Rx(-pi/2) 	: Y to Z basis ..... 

'''




def third_order_trotter_helper(hamiltonian_dict,trotterized_list,op_a, op_b):
	'''
	This function recursively trotterizes a QubitOperator according
		to the scheme:
		e^(A+B) = e^(7/24 * A) e^(2/3 * B) e^(3/4 * A) e^(-1/24 * A) e^(B)

	Used in trotterize() function.

	Params: 
		trotterized_list: list that the function populates with trotterized
				QubitOperators
		op_a: first term in hamiltonian as a QubitOperator
		op_b: the rest of the terms in the hamiltonian as a list of
				QubitOperator term dictionary keys

	Returns: None
	'''
	trotterized_list.append(7.0/24.0 * op_a)
	if len(op_b) == 1:
		trotterized_list.append(2.0 / 3.0 * QubitOperator(op_b[0],
			hamiltonian_dict[op_b[0]]))
	else:
		btemp = op_b[1:]
		for i in btemp:
			hamiltonian_dict[i] *= 2.0 / 3.0
		atemp = QubitOperator(op_b[0],hamiltonian_dict[op_b[0]])
		atemp *= 2.0 / 3.0
		third_order_trotter_helper(trotterized_list, atemp, btemp)
	trotterized_list.append(3.0/4.0 * op_a)

	if op_b.terms.size == 1:
		trotterized_list.append(-2.0/3.0 * QubitOperator(op_b[0],
			hamiltonian_dict[op_b[0]]))
	else:
		btemp = op_b[1:]
		for i in btemp:
			hamiltonian_dict[i] *= -2.0 / 3.0
		atemp = QubitOperator(op_b[0],hamiltonian_dict[op_b[0]])
		atemp *= -2.0 / 3.0
		third_order_trotter_helper(trotterized_list,atemp,btemp)
	trotterized_list.append(-1.0 / 24.0 * op_a)

	if op_b.terms.size == 1:
		trotterized_list.append(QubitOperator(op_b[0],
			hamiltonian_dict[op_b[0]]))
	else:
		btemp = op_b[1:]
		atemp = QubitOperator(op_b[0],hamiltonian_dict[op_b[0]])
		third_order_trotter_helper(trotterized_list, atemp, btemp)




def trotterize(
	hamiltonian,trotter_steps = 1, 
	trotter_order = 1, 
	term_ordering = [], 
	k_exp = 1.):
	'''
	This function trotterizes a hamiltonian to an order 1-3 and 
	for a given number of trotter steps

	Params:
		hamiltonian: full hamiltonian as a QubitOperator.
		trotter_steps: number of trotter steps as an integer.
		order: order of trotterization as an integer from 1-3.
		term_ordering (optional): list of tuples (QubitOperator terms
				dictionary keys) that specifies order of terms when
				trotterizing.
		k_exp (optional): exponential factor to all terms when trotterizing.
	
	Returns:
		tuple: (A list of single Pauli-term QubitOperators, the number of 
		qubits required by the hamiltonian as an integer)
	
	Raises:
		ValueError if order > 3 or order <= 0 or trotter_steps > 100,
			TypeError for incorrect type 
	'''
	

	if trotter_order > 3 or trotter_order <= 0:
		raise ValueError("Invalid trotter order: " + str(order))
	elif trotter_steps <= 0 or trotter_steps > 100: 
		raise ValueError("Invalid number of trotter steps: " + str(trotter_steps))
		
	if not isinstance(hamiltonian, QubitOperator):
		raise TypeError("Hamiltonian must be a QubitOperator.")
	if not hamiltonian.terms:
		raise TypeError("Hamiltonian must be a non-empty QubitOperator.")
	if not isinstance(trotter_steps, int):
		raise TypeError("trotter_steps must be an int")
	if not isinstance(trotter_order, int):
		raise TypeError("trotter_order must be an int")
	if not isinstance(k_exp, float):
		raise TypeError("k_exp must be a float")
		
		
	num_qubits = 0
	for term in hamiltonian.terms.items():
		for item in term[0]:
			if item[0] > num_qubits:
				num_qubits = item[0]
	num_qubits += 1
	ret_val = []
	
	if len(term_ordering) == 0:
		for term in hamiltonian.terms.items():
			term_ordering.append(term[0])
	
	if trotter_order == 1:
		for step in range(trotter_steps):
			for op in term_ordering:
				ret_val.append(QubitOperator(op,hamiltonian.terms[op]) 
					* k_exp/trotter_steps)

	elif trotter_order == 2:
		if len(term_ordering) < 2:
			raise ValueError("Not enough terms in the Hamiltonian to do "+\
				"second order trotterization")
		for op in term_ordering[:-1]:
			ret_val.append(QubitOperator(op,hamiltonian.terms[op]) * k_exp 
				/ (2.0 * trotter_steps))
		ret_val.append(QubitOperator(term_ordering[-1],
			hamiltonian.terms[term_ordering[-1]] * k_exp / trotter_steps))

		for op in reversed(term_ordering[:-1]):
			ret_val.append(QubitOperator(op,hamiltonian.terms[op]) * k_exp 
				/ (2.0 * trotter_steps))

		for step in range(trotter_steps - 1):
			ret_val.extend(ret_val)

	elif trotter_order == 3:
		if len(term_ordering) < 2:
			raise ValueError("Not enough terms in the Hamiltonian to do "+\
				"third order trotterization")
		hamiltonian /= trotter_steps
		hamiltonian *= k_exp
		atemp = QubitOperator(term_ordering[0],hamiltonian.terms[term_ordering[0]])
		btemp = term_ordering[1:]
		third_order_trotter_helper(hamiltonian.terms, ret_val,atemp,btemp)

		for step in range(trotter_steps - 1):
			ret_val.extend(ret_val)
	return (ret_val,num_qubits)



def print_qubit_op_to_qasm(ostrm,qubop):
	'''
	This function exponentiates a single-Pauli-string QubitOperator
	and prints it in QASM format to a supplied output stream

	Params:
		ostrm: file output stream
		qubop: single Pauli-term QubitOperator to be printed
		kexp (optional): float to be multiplied to the coefficient 
	
	Returns:
		None
	'''


	for term in qubop.terms:

		termCoeff = qubop.terms[term]

		# List of operators and list of qubit ids
		ops = list()
		qids = list()
		strBas1 = ""  # Basis rotations 1
		strBas2 = ""  # Basis rotations 2
		nl = "\n"  # New line

		for p in term:	# p = single pauli term
			qid = p[0]	# Qubit index
			pop = p[1]	# Pauli op

			qids.append(qid)  # Qubit index
			ops.append(pop)	 # Pauli operator

			if pop == 'X':
				strBas1 += "H " + str(qid) + nl	 # Hadamard
				strBas2 += "H " + str(qid) + nl	 # Hadamard
			elif pop == 'Y':
				strBas1 += "Rx 1.57079632679 " + str(qid) + nl # Z --> Y
				strBas2 += "Rx -1.57079632679 " + str(qid) + nl	 # Y --> Z

		# Prep for CNOTs
		cnotPairs = np.vstack((qids[:-1], qids[1:]))
		cnots1 = ""
		cnots2 = ""
		for i in range(cnotPairs.shape[1]):
			pair = cnotPairs[:, i]
			cnots1 += "CNOT " + str(pair[0]) + " " + str(pair[1]) + nl
		for i in np.arange(cnotPairs.shape[1])[::-1]:
			pair = cnotPairs[:, i]
			cnots2 += "CNOT " + str(pair[0]) + " " + str(pair[1]) + nl

		# Exponentiating each Pauli string requires five parts

		# 1. Perform basis rotations
		ostrm.write(strBas1)

		# 2. First set CNOTs
		# Store in string, add later
		ostrm.write(cnots1)

		# 3. Rotation (Note kexp & Ntrot)
		ostrm.write("Rz " + str(termCoeff) + " "+ str(qids[-1])+ nl)

		# 4. Second set of CNOTs
		ostrm.write(cnots2)

		# 5. Rotate back to Z basis
		ostrm.write(strBas2)


def write_qubit_ops(op_list,ostrm):
	'''
	This function takes in a list of QubitOperators and sends them
	to be exponentiated and printed to a QASM file one at a time

	Params: 
	op_list: list of QubitOperators
	ostrm: output stream (e.g. file or StringIO object)

	Returns:
	None
    '''

	for op in op_list:
		print_qubit_op_to_qasm(ostrm,op)






def print_to_qasm(
	file_path, 
	hamiltonian, 
	trotter_steps=1, 
	trotter_order=1, 
	term_ordering = [], 
	k_exp = 1.):
	'''
    	This function trotterizes a Qubit hamiltonian and prints it to a QASM file
    
    	Params:
	    	file_path: (string) absolute file path and file name to write
	    			the QASM file to
	    	hamiltonian: QubitOperator to be trotterized and printed
	    	trotter_steps (optional): number of trotter steps (slices) for
	    			trotterization as an integer
	    	trotter_order (optional): order of trotterization as an integer
	    	term_ordering (optional): list of tuples (QubitOperator terms
	    			dictionary keys) that specifies order of terms when
					trotterizing
	    	k_exp (optional): exponential factor to all terms when trotterizing
    	
    	Returns: None
    	
    	Raises: IOError if invalid file path, TypeError if incorrect types
    
    	'''

    
	# Assert that hamiltonian is a QubitOperator
	if not isinstance(hamiltonian, QubitOperator):
		raise TypeError("Hamiltonian must be a QubitOperator.")
	if not hamiltonian.terms:
		raise TypeError("Hamiltonian must be a non-empty QubitOperator.")
	if not isinstance(trotter_steps, int):
		raise TypeError("trotter_steps must be an int")
	if not isinstance(trotter_order, int):
		raise TypeError("trotter_order must be an int")
	if not isinstance(k_exp, float):
		raise TypeError("k_exp must be a float")

	# Attempt to open file path
	try:
		outfile = open(file_path, 'w+')
	except IOError:
		print("File path error.")
		return

	trotterized_ham, num_qubits = trotterize(
		hamiltonian,
		trotter_steps,
		trotter_order,
		term_ordering,
		k_exp)

	outfile.write(str(num_qubits)+'\n')
	outfile.write("# ***\n")
	write_qubit_ops(trotterized_ham,outfile)
	outfile.close()

