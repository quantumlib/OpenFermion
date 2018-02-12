import numpy

from _binary_operator import SymbolicBinary
from openfermion.ops._code_operator import BinaryCode
from openfermion.ops._qubit_operator import QubitOperator




def _extract(term):
    """
    Extractor superoperator helper function.

    Args:
        term: (SymbolicBinary) a single summand from a binary expression

    Returns: (QubitOperator)

    """
    if len(term) == 1:
        term = term[0]
        if term[1] == 'W':
            return QubitOperator('Z' + str(term[0]))
        elif term[1] == '1':
            return -1  # we only have 1/0s in SymbolicBinaries
            #            and zeros are not represented
    if len(term) > 1:
        return dissolve(term)


def extractor(binary_op):
    """
    Applies the extraction superoperator to a binary expression to obtain the
    corresponding qubit operators. 

    Args:
        binary_op: (SymbolicBinary) the binary term

    Returns: (QubitOperator)  the qubit operator corresponding to the 
    binary terms 

    """
    return_fn = 1
    for term in binary_op.terms:
        return_fn *= _extract(term)
    return return_fn


def dissolve(term):
    """
    Decomposition helper. Takes a product of binary variables 
    and outputs the Pauli-string sum that corresponds to the 
    decomposed multi-qubit operator. 
.    Args:
        term: (SymbolicBinary) product of binary variables like 'w0 w2 w3' 

    Returns: (QubitOperator) superposition of Pauli-strings

    """
    prod = 2.0
    for var in term:
        if var[1] != 'W':
            raise ValueError('dissolve works on symbols W')
        prod *= (QubitOperator((), 0.5) - QubitOperator(
            'Z' + str(var[0]), 0.5))
    return QubitOperator((), 1.) - prod


def make_parity_list(code):
    """
    Create the parity list from the decoder of the input code.
    The output parity list has a similar structure as code.decoder. 
    Args:
        code: (BinaryCode)

    Returns: the parity list (list of SymbolicBinaries)
    Raises: (TypeError) argument is not BinaryCode

    """
    if not isinstance(code, BinaryCode):
        raise TypeError('argument is not BinaryCode')
    parity_binaries = [SymbolicBinary()]
    for index in numpy.arange(code.modes - 1):
        parity_binaries += [parity_binaries[-1] + code.decoder[index]]
    return parity_binaries


def code_transform(hamiltonian, code):
    """Transforms a Hamiltonian of Fermions into a Hamiltonian of qubits,
    via a binary code. 
    The role of the binary code is to relate the 
    occupation vectors (v0 v1 v2 ... vN-1) that span the fermionic
    basis, where e.g. the vector (1,0,1,1,0,0) yields the state
    
    a_0^\dagger  a_2^\dagger  a_3^\dagger |vac>,
    
    to the qubit basis, spanned by binary vectors
    (w0, w1, w2, ..., wn-1) determining a product state 
    |w0> |w1> |w2> ... |wn-1>. 
    
    The binary code has to provide an analytic relation between 
    between the binary vectors (v0, v1, ..., vN-1) and 
    (w0, w1, ..., wn-1), and possibly has the property 
    N>n, when the Fermion basis is smaller than the fermionic Fock space.
    In this way, the code_transform function can transform Fermion operators
    to qubit operators for customized and qubit-saving mappings.
    NOTE: Logic multi-qubit operators are decomposed 
    (e.g. CPhase(1,2) = 0.5 * (1 + Z1 + Z2 - Z1 Z2 ) ), which might increase
    the number of Hamiltonian terms drastically. 
    
    Args:
        hamiltonian: (FermionOperator) the fermionic Hamiltonian
        code: (BinaryCode) the binary code to transform the Hamiltonian
    
    Returns: (QubitOperator) the transformed Hamiltonian
        
    
    """
    new_hamiltonian = QubitOperator()
    parity_list = make_parity_list(code)

    for term in hamiltonian.terms:
        num_operators = len(term)

        # the updated parity and occupation account for sign changes due
        # to changed occupations mid-way in the term
        updated_parity = 0
        updated_occupation = numpy.ones(num_operators, dtype=int)
        for main_index in numpy.arange(1, num_operators + 1):

            tmp_occupation = 0
            for runner_index in numpy.arange(1, main_index):
                if term[-runner_index][0] == term[-main_index][0]:
                    tmp_occupation += 1  # delta

                if term[-runner_index][0] < term[-main_index][
                    0]: updated_parity += 1  # theta
            updated_occupation[-main_index] = (-1) ** (tmp_occupation)
        # making the parity and projection operator(s)
        transformed_term = QubitOperator((), (-1) ** (updated_parity))
        parity_term = SymbolicBinary()

        for index in numpy.arange(num_operators):
            transformed_term *= .5 * (
                QubitOperator(()) - updated_occupation[index] * (-1) ** (
                term[index][1]) * extractor(code.decoder[term[index][0]]))
            parity_term += parity_list[term[index][0]]  # update
        transformed_term *= extractor(parity_term)

        # making the update operator
        update_operator = QubitOperator(())
        changed_occupation_vector = numpy.zeros(code.modes, dtype=int)

        for op in term: changed_occupation_vector[op[0]] += 1
        changed_qubit_vector = code.encoder.dot(
            numpy.array(changed_occupation_vector)) % 2

        for index in numpy.arange(code.qubits):
            if changed_qubit_vector[index]: update_operator *= QubitOperator(
                'X' + str(index))  

        transformed_term = hamiltonian.terms[
                               term] * update_operator * transformed_term
        new_hamiltonian += transformed_term
    return new_hamiltonian


if __name__ == '__main__':
    from decoder_encoder_functions import BK_code
    from openfermion.ops._fermion_operator import FermionOperator

    code1 = BinaryCode(numpy.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),
                       [SymbolicBinary('w0'), SymbolicBinary('w0 + w1 + 1'),
                        SymbolicBinary('w0 w1 w2')])
    print (BK_code(10).encoder.toarray())
    hamil1 = FO(' 1 2^', 1.0)
    print (code_transform(hamil1, code1))
