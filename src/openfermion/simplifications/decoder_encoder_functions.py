import numpy
from openfermion.ops import SymbolicBinary
from _code_operator import BinaryCode, linearize_decoder 

# TODO: BK, JW these are already implemented. ideally we would not have additional implementations here.?

def encoder_bk(d):
    """
    outputs the binary tree matrix used for the encoding
    Args:
        d: dimension

    Returns: 

    """
    reps = int(numpy.ceil(numpy.log2(d)))
    mtx = numpy.array([[1,0],[1,1]])
    for a in numpy.arange(1,reps+1):
        mtx = numpy.kron(numpy.eye(2,dtype=int),mtx)
        for b in numpy.arange(0,2**a+1):
            mtx[2**a,b]=1
    return mtx[0:d,0:d]

def decoder_bk(d):
    """
    outputs the inverse of the binary tree matrix utilized for decoding
    Args:
        d: dimension

    Returns: decoder matrix of Bravyi Kitaev

    """
    reps = int(numpy.ceil(numpy.log2(d)))
    mtx = numpy.array([[1, 0], [1, 1]])
    for a in numpy.arange(1,reps+1):
        mtx = numpy.kron(numpy.eye(2),mtx)
        mtx[2**a,2**(a-1)]=1
    return mtx[0:d,0:d]


def encoder_checksum(sites):
    """
    matrix for checksum codes
    Args:
        sites: matrix dimension is sites - 1 x sites

    Returns: encoder matrix

    """
    enc = numpy.zeros(shape=(sites-1,sites),dtype=int)
    for i in range(sites - 1): enc[i,i] = 1
    return enc

def decoder_checksum(sites,odd):
    """
    Decoding function for checksum codes.
    Args:
        sites:N, number of orbitals
        odd: either 1 or 0, if 1 we encode all states with odd Hamming weight

    Returns: a SymbolicBinary list

    """
    if odd==1: all_in = SymbolicBinary('1')
    else: all_in = SymbolicBinary()

    for a in range(sites-1):
        all_in += SymbolicBinary('w'+str(a))

    djw = linearize_decoder(numpy.identity(sites-1,dtype=int))
    djw = numpy.append( djw,[all_in])
    return djw

def checksum_code(sites,odd):
    """
        Checksum code for either even or odd Hamming-weight
        
        Args:
            sites:N, number of orbitals
            odd: either 1 or 0, if 1 we encode all states with odd Hamming - weight

        Returns: (BinaryCode)
    """
    return BinaryCode(encoder_checksum(sites),decoder_checksum(sites, odd))

def JW_code(sites):
    """
        The Jordan-Wigner transform as binary code. 
        
        Args:
            sites: (int) N, number of orbitals

        Returns: (BinaryCode)
    """
    return BinaryCode(numpy.identity(sites,dtype=int),linearize_decoder(numpy.identity(sites,dtype=int)))

def BK_code(sites):
    """
        The Bravyi-Kitaev transform as binary code. 
        
        Args:
            sites: (int) N, number of orbitals

        Returns: (BinaryCode)
    """
    return BinaryCode(encoder_bk(sites),linearize_decoder(decoder_bk(sites)))

def parity_code(sites):
    """
        The parity transform as binary code. 
        
        Args:
            sites: (int) N, number of orbitals

        Returns: (BinaryCode)
    """
    dec_mtx=numpy.reshape(([1]+[0]*(sites-1))+([1,1]+(sites-1)*[0])*(sites-2)+[1,1],(sites,sites))
    enc_mtx=numpy.tril(numpy.ones((sites,sites),dtype=int))
    
    return BinaryCode(enc_mtx,linearize_decoder(dec_mtx))

def binary_address(digits, address):
    """
    Helper function to fill in an encoding column /  decoding component of a certain number.
    Args:
        digits: (int) number of digits, which is the qubit number
        address: column index, decoding component
    Returns: (tuple) encoding column, 
    """
    _binary_expression=SymbolicBinary('1')
    
    #isolate the binary number and fill up the mismatching digits
    address=bin(address)[2:]
    address=('0'*(digits-len(address)))+address   
    for index in numpy.arange(digits):
        _binary_expression *= SymbolicBinary('w'+str(index)+' + 1 '+address[index])
    
    return list(map(int, list(address))), _binary_expression 
    
    

def weight_one_binary_addressing_code(exponent):
    """
        Weight-1 binary addressing code. 
        (Use with care !)
        
        
        Args: 
            exp: (int) exponent for the number of orbitals N = 2 ^ exponent
            
        Returns: (BinaryCode)

    """
    enc, dec = numpy.zeros((exponent,2**exponent),dtype=int), numpy.zeros(2**exponent,dtype=SymbolicBinary)
    for counter in numpy.arange(2**exponent):
        enc[:,counter], dec[counter] = binary_address(exponent, counter)
        #print(binary_address(exponent, counter))
    return BinaryCode(enc,dec)

def weight_one_segment_code():
    """
    Weight-1 segment code.
    (Use with care!)
    Returns: (BinaryCode)
    """
    
    return BinaryCode( [[1,0,1],[0,1,1]], ['w0 w1 + w1', 'w0 w1 + w0',' w0 w1'])

def weight_two_segment_code():
    """
    Weight-2 segment code.
    (Use with care!)
    Returns: (BinaryCode)
    """
    switch='w0 w1 w2 + w0 w1 w3 + w0 w2 w3 + w1 w2 w3 + w0 w1 w2 + w0 w1 w2 w3'
    
    return BinaryCode( [[1,0,0,0,1],[0,1,0,0,1],[0,0,1,0,1],[0,0,0,1,1]],['w0 + '+switch,'w1 + '+switch,'w2 + '+switch,'w3 + '+switch, switch ])



# if __name__=='__main__':



