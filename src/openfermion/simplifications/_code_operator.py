from _binary_operator import SymbolicBinary
import numpy 
import copy
import warnings

# TODO: sparse encoders, tests, dissolve toggle: may get too big

def _shift_decoder(decode,runner):
    """
    Shifts the indices of a decoder by a constant.
    
    Args:
        decode:  (array of SymbolicBinary) a decoder 
        runner: (int) the qubit index that corresponds to the offset.

    Returns: (array of SymbolicBinary) shifted decoder
    """
    decode_shifted = []
    for entry in decode:
        tmp_entry = copy.deepcopy(entry)
        tmp_entry._shift(runner)
        decode_shifted.append(tmp_entry)
    return numpy.array(decode_shifted)

def _double_decoding(dec1, dec2):
    """
    Concatenates two decodings     
    
    Args:
        dec1 (numpy array of SymbolicBinary) decoding of the outer code layer 
        dec2 (numpy array of SymbolicBinary) decoding of the inner code layer 
    
    Returns: the decoding (numpy array of SymbolicBinary) defined by
    w -> dec1(dec2(w))
    """
    doubled_decoder=[]
    for entry in dec1:
        tmp_sum=0
        for summand in entry.terms:
            tmp_term=SymbolicBinary('1')
            for factor in summand:
                if(factor[1]=='W'):
                    tmp_term*=dec2[factor[0]]  
            tmp_sum = tmp_term + tmp_sum
        doubled_decoder +=[tmp_sum]
    return numpy.array(doubled_decoder)
            
            
class BinaryCodeError(Exception):
    pass

def linearize_decoder(mtx):
    """
    Outputs a linear decoding function from a matrix
    
    Args:
        mtx: (array, tuple or list) to derive the decoding function from

    Returns: array of SymbolicBinary

    """
    mtx = numpy.array(list(map(numpy.array,mtx)))
    system_dim,code_dim = numpy.shape(mtx)
    res = []*system_dim
    for a in numpy.arange(system_dim):
        dec_str = ''
        for b in numpy.arange(code_dim):
            if mtx[a,b]==1:
                dec_str+='W'+str(b)+' + '
        dec_str = dec_str.rstrip(' + ')
        res.append(SymbolicBinary(dec_str))
    return numpy.array(res)

class BinaryCode(object):
    """
    The BinaryCode class provides a representation of an encoding-decoding pair
    for binary vectors of different lengths, where the decoding is allowed to 
    be non-linear. 
    
    As the occupation number of fermionic orbital is effectively binary,
    a length-N vector (v) of binary number can be utilized to describe 
    a configuration of a many-body fermionic state on N orbitals. 
    An n-qubit product state configuration |w0> |w1> |w2> ... |wn-1>,
    on the other hand is described by a length-n binary vector 
    w=(w0, w1, ..., wn-1). To map a subset of N-Orbital Fermion states 
    to n-qubit states we define a binary code, which consists of a 
    (here: linear) encoding (e) and a (non-linear) decoding (d), such 
    that for every v from that subset, w = e(v) is a length-n binary 
    vector with d(w) = v.  This can be used to save qubits given a 
    Hamiltonian that dictates such a subset, otherwise n=N. 
    
    Two binary codes (e,d) and (e',d') can construct a third code (e",d")
    by two possible operations:
    
    Concatenation: (e",d") = (e,d) * (e',d')
    which means e": v" -> e'( e(v") ) and d": w" -> d( d'(w") )
    where n" = n' and N" = N, with n = N' as necessary condition. 
    
    Appendage: (e",d") = (e,d) + (e',d')
    which means e": (v + v') -> e(v) + e'(v') and d": (w + w') -> d(w) + d'(w') 
    where the addition is to be understood as appending two vectors together,
    so N" = N' + N and n" = n + n'.
    
    Appending codes is particularily useful when considering segment codes or
    segmented transforms. 
    
    A BinaryCode-instance is initialized by BinaryCode(A,d),
    given the encoding (e) as n x N array or matrix-like nested lists A, 
    such that e(v) = (A v) mod 2. The decoding d is an array or a list-like 
    input of length N, which has entries either of type SymbolicBinary, or of 
    valid type for an input of the SymbolicBinary-constructor. 
    
    The signs + and *, += and *= are overloaded to implement concatination
    and appendage on BinaryCode-objects.  Note that multiplication of a 
    BinaryCode with an integer yields a multiple appending of the same 
    code, the multiplication with another BinaryCode their concatenation. 

    """

    def __init__(self, encoding, decoding):
        """
        Initialization of a binary code.
        Args: 
            encoding: (array or list) nested lists or binary 2D-array  
            decoding: (array or list) list of SymbolicBinary, list-like or str
            
        
        Raises: 
            TypeError
            ValueError
        
        """

        if not isinstance(encoding, (numpy.ndarray,list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (numpy.ndarray,list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        self.enc=numpy.array(list(map(numpy.array,encoding)))
        self.qubits,self.orbitals = numpy.shape(numpy.array(encoding))

        if(self.orbitals!=len(decoding)):
            raise BinaryCodeError('size mismatch, decoder and encoder should have the same first dimension')

        decoder_qubits = set()
        self.dec = []

        for symbolic_binary in numpy.array(decoding):

            if isinstance(symbolic_binary, (tuple, list, str)):
                symbolic_binary = SymbolicBinary(symbolic_binary)
            if isinstance(symbolic_binary,SymbolicBinary):
                self.dec.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(symbolic_binary.enumerate_qubits())
            else:
                raise TypeError('decoder component provided is not a suitable for SymbolicBinary',symbolic_binary)

        if max(decoder_qubits)+1>self.qubits:
            raise ValueError('decoder is indexing more qubits than encoder')
        
        if len(decoder_qubits)!=self.qubits:
            raise ValueError('decoder and encoder provided has different number of qubits')
        
        self.dec=numpy.array(self.dec)

    def __iadd__(self, appendix):
        """
        In-place appending a binary code with +=.  
        
        Args:
            appendix (BinaryCode): The code to append to the present one. 

        Returns:
            global code (BinaryCode)       
        """
        if not isinstance(appendix,BinaryCode):
            raise TypeError('argument must be a BinaryCode.')
        
        self.dec=numpy.append(self.dec, _shift_decoder(appendix.dec,self.qubits)) 
        
        tmp_mtx=numpy.zeros((self.qubits+appendix.qubits,self.orbitals+appendix.orbitals),int)
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
        In-place code concatenation or appendage via *= . 
        Multiplication with integer will yield appendage, otherwise 
        concatenation. 
        
        Args:
            factor: (int or array of SymbolicBinary) 
            
        Returns: segmented or concatenated code (BinaryCode) 
        """
        
        if not isinstance(factor,(BinaryCode,int)):
            raise TypeError('argument must be a BinaryCode or integer')
        if isinstance(factor, BinaryCode):
            if self.qubits != factor.orbitals:
                raise BinaryCodeError('size mismatch between inner and outer code layer')
            
            self.dec= _double_decoding(self.dec,factor.dec)
            self.enc= numpy.dot(factor.enc,self.enc)
            self.qubits=factor.qubits
            return self
        elif isinstance(factor, int):
            if factor<1:
                raise ValueError('integer factor has to be positive, non-zero ')
            self.enc=numpy.kron(numpy.identity(factor, dtype=int),self.enc)
            tmp_dec=self.dec
            for index in numpy.arange(1,factor):
                self.dec=numpy.append( self.dec, _shift_decoder(tmp_dec,index*self.qubits)) 
            self.qubits *= factor
            return self
            
            
        
    
    def __mul__(self,factor):
        """
        Concatenation of two codes or appendage of the same code over and over,  
        via * .  Multiplication with an integer will yield appendage, otherwise 
        concatenation. 
        
        Args:
            factor: (int or array of SymbolicBinary) 
            
        Returns: segmented or concatenated code (BinaryCode) 
        
        """
        twin=copy.deepcopy(self)
        twin *= factor
        return twin

    def __rmul__(self,right_factor):
        """
        Appending the same code multiple times via (right) multiplication.
        Args:
            factor: (int or array of SymbolicBinary) 
            
        Returns: (BinaryCode) segmented code 
        
        """
        if isinstance(right_factor,int):
            return self * right_factor
            
            
        
if __name__ == '__main__':
    from decoder_encoder_functions import * 
    a=BinaryCode([[0,1],[1,0]],[SymbolicBinary(' w1 + w0 '),SymbolicBinary('w0 + 1')])
    print (a.enc)
    print (a.dec[1].terms)
    d = BinaryCode([[0,1],[1,0]],[SymbolicBinary(' w0 '),SymbolicBinary('w0 w2')])
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
    print (type(_shift_decoder((JW_code(2)).dec,1)) )
    print ((JW_code(3)*3).dec)

                    
        
                    
                       
            
                
        
        
