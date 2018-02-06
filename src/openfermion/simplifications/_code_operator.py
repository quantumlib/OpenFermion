from _binary_operator import SymbolicBinary
import numpy as np
import copy

# TODO: sparse encoders, tests, dissolve toggle: may get too big

def _shift_decoder(decode,runner):
    """
    Shifts the indices of a decoder by a constant.
    
    Args:
        decode: a SymbolicBinary object
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
    
    Returns: the decoding defined by  w -> dec1 ( dec2 (w) )
    """
    doubled_decoder=[]
    for entry in dec1:
        tmp_sum=0
        for summand in entry.terms:
            tmp_term=SymbolicBinary('1')
            for factor in summand:
                if(factor[1]=='W'):
                    tmp_term*=dec2[factor[0]]  # here I made the qubits (and orbitals) start from index 0 instead of 1
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
    for a in np.arange(system_dim):
        dec_str = ''
        for b in np.arange(code_dim):
            if mtx[a,b]==1:
                dec_str+='W'+str(b)+' + '
        dec_str = dec_str.rstrip(' + ')
        res.append(SymbolicBinary(dec_str))
    return res

class BinaryCode(object):

    def __init__(self, encoding, decoding):

        # TODO: decoding accepts a matrix?

        if not isinstance(encoding, (np.ndarray,list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (np.ndarray,list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        self.enc=np.array(map(np.array,encoding))
        self.qubits,self.orbitals = np.shape(np.array(encoding))

        if(self.orbitals!=len(decoding)):
            raise BinaryCodeError('size mismatch, decoder and encoder should have the same first dimension')

        decoder_qubits = set()
        self.dec = []

        for symbolic_binary in np.array(decoding):

            if isinstance(symbolic_binary, (tuple, list, str)):
                symbolic_binary = SymbolicBinary(symbolic_binary)
            if isinstance(symbolic_binary,SymbolicBinary):
                self.dec.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(symbolic_binary.count_qubits())
            else:
                raise TypeError('decoder component provided is not a suitable for SymbolicBinary',symbolic_binary)

        if len(decoder_qubits)!=max(decoder_qubits)+1:
            Warning('the number of qubits and the max qubit value of the decoder does not match.\n, you '
                          'have an all zero decoder column')

        if len(decoder_qubits)!=self.qubits:
            raise ValueError('decoder and encoder provided has different number of qubits')
        
        self.dec=np.array(self.dec)

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
    
    def __imul__(self,factor):
        """
        Multiplier *= with BinaryCode yields code concatenation. If the factor is integer, it appends the code multiple times.  
        """
        
        if not isinstance(factor,(BinaryCode,int)):
            raise TypeError('argument must be a BinaryCode or integer')
        if isinstance(factor, BinaryCode):
            if(self.qubits != factor.orbitals or self.orbitals != factor.qubits):
                raise BinaryCodeError('size mismatch between inner and outer code layer')
            
            self.dec= _double_decoding(self.dec,factor.dec)
            self.enc= np.dot(factor.enc,self.enc)
            self.qubits=factor.qubits
            return self
        elif isinstance(factor, int):
            if factor<1:
                raise ValueError('integer factor has to be positive, non-zero ')
            self.enc=np.kron(np.identity(factor, dtype=int),self.enc)
            tmp_dec=self.dec
            for index in np.arange(1,factor):
                self.dec=np.append( self.dec, _shift_decoder(tmp_dec,index*self.qubits)) 
            self.qubits *= factor
            return self
            
            
        
    
    def __mul__(self,factor):
        """
        
        """
        twin=copy.deepcopy(self)
        twin *= factor
        return twin

    def __rmul__(self,right_factor):
        """
        
        """
        if isinstance(right_factor,int):
            return self * right_factor
            
            
        
if __name__ == '__main__':
    from decoder_encoder_functions import * 
    a=BinaryCode([[0,1],[1,0]],[SymbolicBinary(' w1 + w0 '),SymbolicBinary('w0 + 1')])
    d = BinaryCode([[0,1],[1,0]],[SymbolicBinary(' w0 '),SymbolicBinary('w0 w1')])
    sum = a+d
    #print '\n',sum.dec
    b=a*a
    c=a+a
    #print b.enc
    #print b.dec
    #print '\n',b.dec
    #print c.enc
    #print '\n',c.dec
    #a = SymbolicBinary('w1')
    #b = SymbolicBinary('w2')
    #print a*b
    #print SymbolicBinary('w1')*SymbolicBinary('w2')
    print type(_shift_decoder((JW_code(2)).dec,1)) 
    print (JW_code(3)*3).dec

                    
        
                    
                       
            
                
        
        
