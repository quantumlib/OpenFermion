#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
""" Pre-existing codes for Fermion-qubit mappings
    based on (arXiv:1712.07067) """

import numpy

from openfermion.ops import BinaryCode, SymbolicBinary, linearize_decoder


def _encoder_bk(n_modes):
    """  Helper function for bravyi_kitaev_code that outputs the binary-tree
    (dimension x dimension)-matrix used for the encoder in the
    Bravyi-Kitaev transform.

    Args:
        n_modes (int): length of the matrix, the dimension x dimension

    Returns (numpy.ndarray): encoder matrix
    """
    reps = int(numpy.ceil(numpy.log2(n_modes)))
    mtx = numpy.array([[1, 0], [1, 1]])
    for repetition in numpy.arange(1, reps + 1):
        mtx = numpy.kron(numpy.eye(2, dtype=int), mtx)
        for column in numpy.arange(0, 2 ** repetition):
            mtx[2 ** (repetition + 1) - 1, column] = 1
    return mtx[0:n_modes, 0:n_modes]


def _decoder_bk(n_modes):
    """ Helper function for bravyi_kitaev_code that outputs the inverse of the
    binary tree matrix utilized for decoder in the Bravyi-Kitaev transform.

    Args:
        n_modes (int): size of the matrix is modes x modes

    Returns (numpy.ndarray): decoder matrix
    """
    reps = int(numpy.ceil(numpy.log2(n_modes)))
    mtx = numpy.array([[1, 0], [1, 1]])
    for repetition in numpy.arange(1, reps + 1):
        mtx = numpy.kron(numpy.eye(2, dtype=int), mtx)
        mtx[2 ** (repetition + 1) - 1, 2 ** repetition - 1] = 1
    return mtx[0:n_modes, 0:n_modes]


def _encoder_checksum(modes):
    """ Helper function for checksum_code that outputs the encoder matrix.

    Args:
        modes (int):  matrix size is (modes - 1) x modes

    Returns (numpy.ndarray): encoder matrix
    """
    enc = numpy.zeros(shape=(modes - 1, modes), dtype=int)
    for i in range(modes - 1):
        enc[i, i] = 1
    return enc


def _decoder_checksum(modes, odd):
    """ Helper function for checksum_code that outputs the decoder.

    Args:
        modes (int):  number of modes
        odd (int or bool): 1 (True) or 0 (False), if odd,
            we encode all states with odd Hamming weight

    Returns (list): list of SymbolicBinary
    """
    if odd:
        all_in = SymbolicBinary('1')
    else:
        all_in = SymbolicBinary()

    for mode in range(modes - 1):
        all_in += SymbolicBinary('w' + str(mode))

    djw = linearize_decoder(numpy.identity(modes - 1, dtype=int))
    djw.append(all_in)
    return djw


def _binary_address(digits, address):
    """ Helper function to fill in an encoder column/decoder component of a
    certain number.

    Args:
        digits (int): number of digits, which is the qubit number
        address (int): column index, decoder component

    Returns (tuple): encoder column, decoder component
    """
    binary_expression = SymbolicBinary('1')

    # isolate the binary number and fill up the mismatching digits
    address = bin(address)[2:]
    address = ('0' * (digits - len(address))) + address
    for index in numpy.arange(digits):
        binary_expression *= SymbolicBinary(
            'w' + str(index) + ' + 1 + ' + address[index])

    return list(map(int, list(address))), binary_expression


def checksum_code(n_modes, odd):
    """ Checksum code for either even or odd Hamming weight. The Hamming weight
    is defined such that it yields the total occupation number for a given basis
    state. A Checksum code with odd weight will encode all states with odd
    occupation number. This code saves one qubit: n_qubits = n_modes - 1.
        
    Args:
        n_modes (int): number of modes
        odd (int or bool): 1 (True) or 0 (False), if odd,
            we encode all states with odd Hamming weight

    Returns (BinaryCode): The checksum BinaryCode
    """
    return BinaryCode(_encoder_checksum(n_modes),
                      _decoder_checksum(n_modes, odd))


def jordan_wigner_code(n_modes):
    """ The Jordan-Wigner transform as binary code.
        
    Args:
        n_modes (int): number of modes

    Returns (BinaryCode): The Jordan-Wigner BinaryCode
    """
    return BinaryCode(numpy.identity(n_modes, dtype=int), linearize_decoder(
        numpy.identity(n_modes, dtype=int)))


def bravyi_kitaev_code(n_modes):
    """ The Bravyi-Kitaev transform as binary code. The implementation
    follows arXiv:1208.5986.
        
    Args:
        n_modes (int): number of modes

    Returns (BinaryCode): The Bravyi-Kitaev BinaryCode
    """
    return BinaryCode(_encoder_bk(n_modes),
                      linearize_decoder(_decoder_bk(n_modes)))


def parity_code(n_modes):
    """ The parity transform (arXiv:1208.5986) as binary code. This code is
    very similar to the Jordan-Wigner transform, but with long update strings
    instead of parity strings. It does not save qubits: n_qubits = n_modes.
        
    Args:
        n_modes (int): number of modes

    Returns (BinaryCode): The parity transform BinaryCode
    """
    dec_mtx = numpy.reshape(([1] + [0] * (n_modes - 1)) +
                            ([1, 1] + (n_modes - 1) * [0]) * (n_modes - 2) +
                            [1, 1], (n_modes, n_modes))
    enc_mtx = numpy.tril(numpy.ones((n_modes, n_modes), dtype=int))

    return BinaryCode(enc_mtx, linearize_decoder(dec_mtx))


def weight_one_binary_addressing_code(exponent):
    """ Weight-1 binary addressing code (arXiv:1712.07067). This highly
    non-linear code works for a number of modes that is an integer power
    of two. It encodes all possible vectors with Hamming weight 1, which
    corresponds to all states with total occupation one. The amount of
    qubits saved here is maximal: for a given argument 'exponent', we find
    n_modes = 2 ^ exponent, n_qubits = exponent. 

    Note:
        This code is highly non-linear and might produce a lot of terms.

    Args:
        exponent (int): exponent for the number of modes n_modes = 2 ^ exponent

    Returns (BinaryCode): the weight one binary addressing BinaryCode
    """
    encoder = numpy.zeros((exponent, 2 ** exponent), dtype=int)
    decoder = [0] * (2 ** exponent)
    for counter in numpy.arange(2 ** exponent):
        encoder[:, counter], decoder[counter] = \
            _binary_address(exponent, counter)
    return BinaryCode(encoder, decoder)


def weight_one_segment_code():
    """ Weight-1 segment code (arXiv:1712.07067). Outputs a 3-mode, 2-qubit
    code, which encodes all the vectors (states) with Hamming weight
    (occupation) 0 and 1. n_qubits = 2, n_modes = 3.
    A linear amount of qubits can be saved  appending several instances of this
    code.

    Note:
        This code is highly non-linear and might produce a lot of terms.

    Returns (BinaryCode): weight one segment code
    """
    return BinaryCode([[1, 0, 1], [0, 1, 1]],
                      ['w0 w1 + w0', 'w0 w1 + w1', ' w0 w1'])


def weight_two_segment_code():
    """ Weight-2 segment code (arXiv:1712.07067). Outputs a 5-mode, 4-qubit
    code, which encodes all the vectors (states) with Hamming weight
    (occupation) 2 and 1. n_qubits = 4, n_modes = 5.
    A linear amount of qubits can be saved  appending several instances of this
    code.

    Note:
        This code is highly non-linear and might produce a lot of terms.

    Returns (BinaryCode): weight-2 segment code
    """
    switch = ('w0 w1 w2 + w0 w1 w3 + w0 w2 w3 + w1 w2 w3 + w0 w1 w2 +'
              ' w0 w1 w2 w3')

    return BinaryCode([[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1]], ['w0 + ' + switch, 'w1 + ' + switch,
                                          'w2 + ' + switch, 'w3 + ' + switch,
                                          switch])


def interleaved_code(modes):
    """ Linear code that reorders orbitals from even-odd to up-then-down.
    In up-then-down convention, one can append two instances of the same
    code 'c' in order to have two symmetric subcodes that are symmetric for 
    spin-up and -down modes: ' c + c '.
    In even-odd, one can concatenate with the interleaved_code 
    to have the same result:' interleaved_code * (c + c)'.
    This code changes the order of modes from (0, 1 , 2, ... , modes-1 )
    to (0, modes/2, 1 modes/2+1, ... , modes-1, modes/2 - 1).  
    n_qubits = n_modes. 
    
    Args: modes (int): number of modes, must be even 
    
    Returns (BinaryCode): code that interleaves orbitals
    """
    if modes % 2 == 1:
        raise ValueError('number of modes must be even')
    else:
        mtx = numpy.zeros((modes, modes), dtype=int)
        for index in numpy.arange(modes // 2, dtype=int):
            mtx[index, 2 * index] = 1
            mtx[modes // 2 + index, 2 * index + 1] = 1
        return BinaryCode(mtx, linearize_decoder(mtx.transpose()))
