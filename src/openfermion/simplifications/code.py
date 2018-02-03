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
        tmp_entry = copy.deepcopy(entry)
        tmp_entry._shift(runner)
        decode_shifted.append(tmp_entry)
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
                    tmp_term*=dec2[factor[0]-1]  # are we going to count qubits from 0 or 1? - I think openfermion convention is 0, we should check!
            tmp_sum = tmp_term + tmp_sum
        doubled_decoder +=[tmp_sum]
    return np.array(doubled_decoder)
            
            
class BinaryCodeError(Exception):
    pass

def linearize_decoder(mtx):
    """
    outputs a linear decoding function when given a matrix
    Args:
        mtx: matrix to derive the decoding function from

    Returns: a list of symbolicBinaries

    """
    mtx = np.array(map(np.array,mtx))
    system_dim,code_dim = np.shape(mtx)
    res = []*system_dim
    for a in range(system_dim):
        dec_str = ''
        for b in range(code_dim):
            if mtx[a,b]==1:
                dec_str+='W'+str(b)+' + '
        dec_str = dec_str.rstrip(' + ')
        res.append(SymbolicBinary(dec_str))
    return res

class BinaryCode(object):

    def __init__(self, encoding, decoding):
        # only accepts linear decoders?

        if not isinstance(encoding, (list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        #transform the encoding into a numpy array extracting the total number qubits and orbitals
        self.enc=np.array(map(np.array,encoding))
        self.qubits,self.orbitals = np.shape(np.array(encoding))

        if(self.orbitals!=len(decoding)):
            raise BinaryCodeError('size mismatch, decoder and encoder should have the same first dimension')

        decoder_qubits = set()
        self.dec = []
        
        # build the decoder - question: what if its given as a matrix... we should be able to handle that

        for symbolic_binary in np.array(decoding):

            if isinstance(symbolic_binary, (tuple, list, str)):
                symbolic_binary = SymbolicBinary(symbolic_binary)
            if isinstance(symbolic_binary,SymbolicBinary):
                self.dec.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(symbolic_binary.count_qubits())
            else:
                raise TypeError('decoder component provided is not a suitable for SymbolicBinary',symbolic_binary)

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
    d = BinaryCode([[0,1],[1,0]],[SymbolicBinary(' w0 '),SymbolicBinary('w3 w4')])
    sum = a+d
    print sum.dec
    b=a*a
    c=a+a
    #print b.enc
    #print b.dec
    print b.dec[0].terms
    print b.dec[0]
    #print c.enc
    #print c.dec

                    
        
                    
                       
            
                
        
        
