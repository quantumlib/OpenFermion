import numpy as np
from _binary_operator import SymbolicBinary

# TODO: BK, JW these are already implemented. ideally we would not have additional implementations here.

def encoder_bk(d):
    """
    outputs the binary tree matrix used for the encoding
    Args:
        d: dimension

    Returns:

    """
    reps = np.ceil(np.log2(d))
    mtx = [[1,0],[1,1]]
    for a in np.arange(1,reps+1):
        mtx = np.kron(np.eye(2),mtx)
        for b in np.arange(0,2**a+1):
            mtx[int(2**a),b]=1
    return mtx[0:d,0:d]

def decoder_bk(d):
    """
    outputs the inverse of the binary tree matrix utilized for the \
    decoding
    Args:
        d: dimension

    Returns: decoder matrix of Bravyi Kitaev

    """
    reps = np.ceil(np.log2(d))
    mtx = [[1, 0], [1, 1]]
    for a in np.arange(1,reps+1):
        mtx = np.kron(np.eye(2),mtx)
        mtx[int(2**a),int(2**(a-1))]=1
    return mtx[0:d,0:d]

def decoder_jw(d):
    """
    decoding function for Jordan - Wigner transforms
    Args:
        d: number of orbitals

    Returns: a list of SymbolicBinaries

    """
    return [SymbolicBinary('w'+str(i)) for i in range(d)]

def encoder_checksum(sites):
    """
    matrix for checksum codes
    Args:
        sites: matrix dimension is sites - 1 x sites

    Returns: encoder matrix

    """
    enc = np.zeros(shape=(sites-1,sites))
    for i in range(sites - 1): enc[i,i] = 1
    return enc

def decoder_checksum(sites,odd):
    """
    decoding function for checksum codes
    Args:
        sites:N, number of orbitals
        odd ... either 1 or 0, if 1 we encode all states with odd Hamming - weight

    Returns: a SymbolicBinary list

    """
    if odd==1: all_in = SymbolicBinary()
    else: all_in = SymbolicBinary(())

    for a in range(sites-1):
        all_in += SymbolicBinary('w'+str(a))

    djw = decoder_jw(sites-1)
    djw.append(all_in)
    return djw

if __name__=='__main__':
    print [x.terms for x in decoder_checksum(4,1)]


