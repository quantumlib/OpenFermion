import numpy as np
from _binary_operator import SymbolicBinary
from _code_operator import BinaryCode, linearize_decoder as linear

# TODO: BK, JW these are already implemented. ideally we would not have additional implementations here.?

def encoder_bk(d):
    """
    outputs the binary tree matrix used for the encoding
    Args:
        d: dimension

    Returns:

    """
    reps = int(np.ceil(np.log2(d)))
    mtx = np.array([[1,0],[1,1]])
    for a in np.arange(1,reps+1):
        mtx = np.kron(np.eye(2,dtype=int),mtx)
        for b in np.arange(0,2**a+1):
            mtx[2**a,b]=1
    return mtx[0:d,0:d]

def decoder_bk(d):
    """
    outputs the inverse of the binary tree matrix utilized for decoding
    Args:
        d: dimension

    Returns: decoder matrix of Bravyi Kitaev

    """
    reps = int(np.ceil(np.log2(d)))
    mtx = np.array([[1, 0], [1, 1]])
    for a in np.arange(1,reps+1):
        mtx = np.kron(np.eye(2),mtx)
        mtx[2**a,2**(a-1)]=1
    return mtx[0:d,0:d]


def encoder_checksum(sites):
    """
    matrix for checksum codes
    Args:
        sites: matrix dimension is sites - 1 x sites

    Returns: encoder matrix

    """
    enc = np.zeros(shape=(sites-1,sites),dtype=int)
    for i in range(sites - 1): enc[i,i] = 1
    return enc

def decoder_checksum(sites,odd):
    """
    decoding function for checksum codes
    Args:
        sites:N, number of orbitals
        odd: either 1 or 0, if 1 we encode all states with odd Hamming - weight

    Returns: a SymbolicBinary list

    """
    if odd==1: all_in = SymbolicBinary('1')
    else: all_in = SymbolicBinary()

    for a in range(sites-1):
        all_in += SymbolicBinary('w'+str(a))

    djw = linear(np.identity(sites-1,dtype=int))
    djw += [all_in]
    return djw

def checksum_code(sites,odd):
    """
        Checksum code for either even or odd Hamming-weight
        
        Args:
            sites:N, number of orbitals
            odd: either 1 or 0, if 1 we encode all states with odd Hamming - weight

        Returns: BinaryCode
    """
    return BinaryCode(encoder_checksum(sites),decoder_checksum(sites, odd))

def JW_code(sites):
    """
        Jordan-Wigner as binary code f
        
        Args:
            sites:N, number of orbitals

        Returns: BinaryCode
    """
    return BinaryCode(np.identity(sites,dtype=int),linear(np.identity(sites,dtype=int)))

def BK_code(sites):
    """
        Jordan-Wigner as binary code 
        
        Args:
            sites:N, number of orbitals

        Returns: BinaryCode
    """
    return BinaryCode(encoder_bk(sites),linear(decoder_bk(sites)))



# if __name__=='__main__':



