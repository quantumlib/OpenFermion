from openfermion.ops._fermion_operator import FermionOperator as FO
from openfermion.ops._qubit_operator import QubitOperator as QO
from _binary_operator import SymbolicBinary
from  code import BinaryCode
import numpy as np

def _extract(term):
    """
    extractor superoperator helper function

    Args:
        term: a single summand of th SymbolicBinary object

    Returns: a QubitOperator object or a constant

    """
    if len(term) == 1:
        term = term[0]
        if term[1]=='W':
            return QO('Z'+str(term[0]))
        elif term[1]=='1':
            return -1   # we only have 1/0s in SymbolicBinaries and zeros are not represented
    if len(term) >1:
        return dissolve(term)


def extractor(decode):
    """
    applies the extraction superoperator

    Args:
        decode: a SymbolicBinary object

    Returns: a QubitOperator object

    """
    return_fn = 1
    for term in decode.terms:
        return_fn*=_extract(term)
    return return_fn


def dissolve(term):
    """
    decomposition helper
    Args:
        term: a multi factor term

    Returns: QubitOperator

    """
    prod = 2.0
    for var in term:
        if var[1]!='W':
            raise ValueError('dissolve works on symbols W')
        prod*=(QO((),0.5) - QO('Z'+str(var[0]),0.5))
    return QO((),1.) - prod


def make_parity_list(code):
    if not isinstance(code, BinaryCode):
        raise TypeError('argument is not BinaryCode')
    parity_binaries=[SymbolicBinary()]
    tmp_term=[SymbolicBinary()]
    for index in np.arange(code.orbitals-1):
        tmp_term += code.dec[index]
        parity_binaries+=[tmp_term]
    return parity_binaries
        
        
def code_transform(hamiltonian, code):
    """
    transforms a Hamiltonian of Fermions into a Hamiltonian of qubits, via a binary code
    
    Args:
        hamiltonian: (FermionOperator) the fermionic Hamiltonian
        code: (BinaryCode) the binary code to translate the Hamiltonian
    
    Returns: QubitOperator
        
    
    """
    new_hamiltonian=QO()
    for term in hamiltonian.terms:
        weight=len(term)

            
        #the updated parity and occupation account for sign changes due to changed occupations mid-way in the term
        updated_parity=0
        updated_occupation=np.ones(weight,dtype=int)
            
            
            
            
        for main_index in np.arange(1,weight+1):
            tmp_occupation=0
            for runner_index in np.arange(1,main_index-1):
                if term[- runner_index][0]==term[- main_index][0]: tmp_occupation+=1
                if term[- runner_index][0]<term[- main_index][0]: updated_parity+=1
            updated_occupation[- main_index] = (-1)**(tmp_occupation)
                
        #making the parity and projection operator(s)
        transformed_term= QO(())*(-1)**(updated_parity)
        parity_list=make_parity_list(code)
        parity_term=SymbolicBinary()
            
        for index in np.arange(weight):

            transformed_term *= .5* (QO(())- updated_occupation[index]*(-1)**(term[index][1])*extractor(code.dec[term[index][0]]))
            parity_term += parity_list[term[index][0]]
        transformed_term *= extractor(parity_term)
            
        #making the update operator
        update_operator=QO(())
        changed_occupation_vector=np.zeros(code.orbitals, dtype=int)
            
        for op in term: changed_occupation_vector[op[0]]+=1
        changed_qubit_vector = np.dot(code.enc,np.array(changed_occupation_vector)) % 2
            
        for index in np.arange(code.qubits):
            if changed_qubit_vector[index]: update_operator *= QO('X'+str(index)) # I'll make it start from zero for now.
            
        transformed_term = hamiltonian.terms[term]*update_operator*transformed_term
        new_hamiltonian += transformed_term
    return new_hamiltonian
            
if __name__=='__main__':
    code1=BinaryCode(np.array([[1,0],[0,1]]),[SymbolicBinary('w0'),SymbolicBinary('w1')])
    hamil1=FO('1')
    print code_transform(hamil1,code1)


            
                
                
                
            
                
                
                

            
        
        
