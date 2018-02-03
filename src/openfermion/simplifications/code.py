from _binary_operator import SymbolicBinary
import numpy as np
import copy


def _shift_decoder(decode,runner): # this function was a bit hacky - please test it to death
    """
    Shifts the indices of a decoder by a constant.
    
    Args:
        decode: a symbolicBinary object
        runner: the qubit index that corresponds to the offset

    Returns: a shifted decoder
    """
    decode_shifted = []
    for entry in decode:
        tmp_string=''
        for element in np.arange(len(entry.terms)):
            for var in entry.terms[element]:
                qidx, op_name = var
                qubit_mapping = qidx + runner
                if(op_name=='W'): 
                    tmp_string+= ' w'+str(qubit_mapping) 
                else: 
                    tmp_string +='  1 '
            if(element<len(entry.terms)-1): tmp_string+= ' +'
        decode_shifted += [SymbolicBinary(tmp_string)]
        #decode_shifted += [SymbolicBinary(list_ops)]
    return np.array(decode_shifted)

def _double_decoding(dec1, dec2):
    """
    
    Args:
        dec1 (numpy array of SymbolicBinary) decoding of the outer code layer 
        dec2 (numpy array of SymbolicBinary) decoding of the inner code layer 
    
    Returns: the decoding difined by  w -> dec1 ( dec2 (w) )
    """
    doubled_decoder=[]
    for entry in dec1:
        tmp_sum=0
        for summand in entry.terms:
            tmp_term=SymbolicBinary('1')
            for factor in summand:
                if(factor[1]=='W'):
                    tmp_term*=dec2[factor[0]-1]
            tmp_sum = tmp_term + tmp_sum
        doubled_decoder +=[tmp_sum]
    return np.array(doubled_decoder)
            
            
class BinaryCodeError(Exception):
    pass



class BinaryCode(object):

    def __init__(self, encoding, decoding):

        if not isinstance(encoding, (list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        #transform the encoding into a numpy array extracting the total number qubits and orbitals
        self.enc=np.array(map(np.array,encoding))
        self.qubits,self.orbitals = np.shape(np.array(encoding))
        # np.reshape(self.enc,(self.qubits,self.orbitals)) casting it to np.array should be enough assuming its a 2D array
        # since we extract the number of orbitals/qubits from this. Also shouldn't we enforce that the user gives
        # qubits x orbitals shaped encoder?

        if(self.orbitals!=len(decoding)):
            raise BinaryCodeError('size mismatch, decoder and encoder should have the same first dimension')

        decoder_qubits = set()
        self.dec = []
        
        # build the decoder - question: what if its given as a matrix... we should be able to handle that
        for symbolic_binary in np.array(decoding):
            # ok there is an issue : do we want to allow for the stuff to be given as [((1,'W'),(2,'W'), ....] or not ?

            # how about this: we only allow components that are acceptable for SymbolicBinary. that way if the user
            # gives a list/text as a component, first we convert to symbolicBinary and do operations later :
            if isinstance(symbolic_binary, (tuple, list, str)):
                symbolic_binary = SymbolicBinary(symbolic_binary)
            if isinstance(symbolic_binary,SymbolicBinary):
                self.dec.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(symbolic_binary.count_qubits())
            else:
                raise TypeError('decoder component provided is not a suitable for SymbolicBinary',symbolic_binary)
        self.decoder_qubits = decoder_qubits

    def __iadd__(self, appendix):
        """
        Appends a binary code to the present code via += . 
        
        Args:
            appendix (BinaryCode): The code to append to the present one. 

        Returns:
            global code (BinaryCode)       
        """
        if not isinstance(appendix,BinaryCode):
            raise TypeError('argument must be a BinaryCode.')
        
        self.dec=np.append(self.dec, _shift_decoder(appendix.dec,self.qubits))
        
        tmp_mtx=np.zeros((self.qubits+appendix.qubits,self.orbitals+appendix.orbitals),int)
        tmp_mtx[:self.qubits,:self.orbitals]=self.enc
        tmp_mtx[self.qubits:,self.orbitals:]=appendix.enc
        
        self.enc=tmp_mtx
        self.qubits+=appendix.qubits
        self.orbitals+=appendix.orbitals
        return self
    
    def __add__(self, appendix):
        """
        Appends two binary codes via addition + . 
        
        Args:
            appendix (BinaryCode): The code to append to the present one. 

        Returns:
            global code (BinaryCode)       
        """
        twin = copy.deepcopy(self)
        twin += appendix
        return twin
    
    def __imul__(self,inner_code):
        """
        
        """
        
        if not isinstance(inner_code,BinaryCode):
            raise TypeError('argument must be a BinaryCode')
        if(self.qubits != inner_code.orbitals or self.orbitals != inner_code.qubits):
            raise BinaryCodeError('size mismatch between inner and outer code layer')
        
        self.dec= _double_decoding(self.dec,inner_code.dec)
        self.enc= np.dot(inner_code.enc,self.enc)
        self.qubits=inner_code.qubits
        return self 
    
    def __mul__(self,inner_code):
        """
        """
        twin=copy.deepcopy(self)
        twin *= inner_code
        return twin

if __name__ == '__main__':
    a=BinaryCode([[0,1],[1,0]],[SymbolicBinary(' w2 + w1 '),SymbolicBinary('w1 + 1')])
    b=a*a
    c=a+a
    #print b.enc
    #print b.dec
    print b.dec[0].terms
    print b.dec[0]
    #print c.enc
    #print c.dec

                    
        
                    
                       
            
                
        
        
