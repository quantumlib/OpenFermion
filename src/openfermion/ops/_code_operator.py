import copy

import numpy
import scipy
import scipy.sparse

from openfermion.ops import SymbolicBinary


def _shift_decoder(decode, shift_constant):
    """ Shifts the indices of a decoder by a constant.
    
    Args:
        decode (list of SymbolicBinary): a decoder
        shift_constant (int): the qubit index that corresponds to the offset.

    Returns (list of SymbolicBinary):  shifted decoder
    """
    decode_shifted = []
    for entry in decode:
        tmp_entry = copy.deepcopy(entry)
        tmp_entry._shift(shift_constant)
        decode_shifted.append(tmp_entry)
    return numpy.array(decode_shifted)


def _double_decoding(dec1, dec2):
    """ Concatenates two decodings
    
    Args:
        dec1 (list of SymbolicBinary): decoding of the outer code layer
        dec2 (list of SymbolicBinary): decoding of the inner code layer
    
    Returns (list of SymbolicBinary): the decoding defined by
    w -> dec1(dec2(w))
    """
    doubled_decoder = []
    for entry in dec1:
        tmp_sum = 0
        for summand in entry.terms:
            tmp_term = SymbolicBinary('1')
            for factor in summand:
                if factor[1] == 'W':
                    tmp_term *= dec2[factor[0]]
            tmp_sum = tmp_term + tmp_sum
        doubled_decoder += [tmp_sum]
    return numpy.array(doubled_decoder)


def linearize_decoder(mtx):
    """ Outputs row_idx linear decoding function from row_idx matrix

    Args:
        mtx (array or list): row_idx 2D list of lists or row_idx numpy array
         to derive the decoding function from

    Returns: list of SymbolicBinary
    """
    mtx = numpy.array(list(map(numpy.array, mtx)))
    system_dim, code_dim = numpy.shape(mtx)
    res = [] * system_dim
    for row_idx in numpy.arange(system_dim):
        dec_str = ''
        for col_idx in numpy.arange(code_dim):
            if mtx[row_idx, col_idx] == 1:
                dec_str += 'W' + str(col_idx) + ' + '
        dec_str = dec_str.rstrip(' + ')
        res.append(SymbolicBinary(dec_str))
    return numpy.array(res)


class BinaryCodeError(Exception):
    pass


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
    
    Appending codes is particularly useful when considering segment codes or
    segmented transforms. 
    
    A BinaryCode-instance is initialized by BinaryCode(A,d),
    given the encoding (e) as n x N array or matrix-like nested lists A, 
    such that e(v) = (A v) mod 2. The decoding d is an array or a list-like 
    input of length N, which has entries either of type SymbolicBinary, or of 
    valid type for an input of the SymbolicBinary-constructor. 
    
    The signs + and *, += and *= are overloaded to implement concatenation
    and appendage on BinaryCode-objects.

    NOTE: that multiplication of a BinaryCode with an integer yields a
    multiple appending of the same code, the multiplication with another
    BinaryCode their concatenation.
    """

    def __init__(self, encoding, decoding):
        """ Initialization of a binary code.

        Args: 
            encoding (array or list): nested lists or binary 2D-array
            decoding (array or list): list of SymbolicBinary, list-like or str

        Raises: 
            TypeError: non-list, array like encoding or decoding, unsuitable
            symbolicBinary generators,
            BinaryCodeError: in case of decoder/encoder size mismatch or decoder
            size, qubits indexed mismatch
        """

        if not isinstance(encoding, (numpy.ndarray, list, tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (numpy.ndarray, list, tuple)):
            raise TypeError('decoding must be a list, array or tuple .')

        self.enc = scipy.sparse.csc_matrix(encoding)
        self.qubits, self.orbitals = numpy.shape(encoding)

        if (self.orbitals != len(decoding)):
            raise BinaryCodeError(
                'size mismatch, decoder and encoder should have the same'
                ' first dimension')

        decoder_qubits = set()
        self.dec = []

        for symbolic_binary in numpy.array(decoding):

            if isinstance(symbolic_binary, (tuple, list, str)):
                symbolic_binary = SymbolicBinary(symbolic_binary)
            if isinstance(symbolic_binary, SymbolicBinary):
                self.dec.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(
                    symbolic_binary.enumerate_qubits())
            else:
                raise TypeError(
                    'decoder component provided '
                    'is not a suitable for SymbolicBinary',
                    symbolic_binary)

        if len(decoder_qubits) != self.qubits:
            raise BinaryCodeError(
                'decoder and encoder provided has different number of qubits')

        if max(decoder_qubits) + 1 > self.qubits:
            raise BinaryCodeError('decoder is not indexing some qubits. Qubits'
                                  'indexed are: {}'.format(decoder_qubits))

        self.dec = numpy.array(self.dec)

    def __iadd__(self, appendix):
        """ In-place appending a binary code with +=.
        
        Args:
            appendix (BinaryCode): The code to append to the present one. 

        Returns (BinaryCode): a global binary code with size n1+n2, N1+N2

        Raises:
            TypeError: appendix must be a BinaryCode
        """
        if not isinstance(appendix, BinaryCode):
            raise TypeError('argument must be a BinaryCode.')

        self.dec = numpy.append(self.dec,
                                _shift_decoder(appendix.dec, self.qubits))
        self.enc = scipy.sparse.bmat([[self.enc, None], [None, appendix.enc]])
        self.qubits, self.orbitals = numpy.shape(self.enc)
        return self

    def __add__(self, appendix):
        """Appends two binary codes via addition +.
        
        Args:
            appendix (BinaryCode): The code to append to the present one. 

        Returns (BinaryCode): global binary code
        """
        twin = copy.deepcopy(self)
        twin += appendix
        return twin

    def __imul__(self, factor):
        """In-place code concatenation or appendage via *= .
        Multiplication with integer will yield appendage, otherwise 
        concatenation. 
        
        Args:
            factor (int or BinaryCode): the BinaryCode to concatenate. In case
             of int, it will append the code to itself factor times.
            
        Returns (BinaryCode): segmented or concatenated code

        Raises:
            TypeError: factor must be an integer or a BinaryCode
            BinaryCodeError: size mismatch between self and factor
            ValueError: in case of an integer factor that is < 1
        """

        if not isinstance(factor, (BinaryCode, int)):
            raise TypeError('argument must be a BinaryCode or integer')

        if isinstance(factor, BinaryCode):
            if self.qubits != factor.orbitals:
                raise BinaryCodeError(
                    'size mismatch between inner and outer code layer')

            self.dec = _double_decoding(self.dec, factor.dec)
            self.enc = factor.enc.dot(self.enc)
            self.qubits, self.orbitals = numpy.shape(self.enc)
            return self

        elif isinstance(factor, int):
            if factor < 1:
                raise ValueError('integer factor has to be positive, non-zero ')

            self.enc = scipy.sparse.kron(
                scipy.sparse.identity(factor, format='csc', dtype=int),
                self.enc, 'csc')
            tmp_dec = self.dec
            for index in numpy.arange(1, factor):
                self.dec = numpy.append(self.dec,
                                        _shift_decoder(tmp_dec,
                                                       index * self.qubits))
            self.qubits *= factor
            return self

    def __mul__(self, factor):
        """ Concatenation of two codes or appendage the same code factor times
        in case of integer factor.
        
        Args:
            factor (int or BinaryCode): the BinaryCode to concatenate. In case
             of int, it will append the code to itself factor times.

        Returns (BinaryCode): segmented or concatenated code
        """
        twin = copy.deepcopy(self)
        twin *= factor
        return twin

    def __rmul__(self, factor):
        """ Appending the same code factor times.
        Args:
            factor (int): integer defining number of appendages.
            
        Returns (BinaryCode): segmented code

        Raises:
            TypeError: right_factor must be an integer
        """
        if isinstance(factor, int):
            return self * factor
        else:
            raise TypeError('the left multiplier must be an integer to a'
                            'BinaryCode. Was given {} of '
                            'type {}'.format(factor, type(factor)))

    def __str__(self):
        """ Return an easy-to-read string representation."""
        string_return = [list(map(list, self.enc.toarray()))]

        dec_str = '['
        for term in self.dec:
            dec_str += term.__str__() + ','
        dec_str = dec_str[:-1]
        string_return.append(dec_str + ']')
        return str(string_return)

    def __repr__(self):
        return str(self)
