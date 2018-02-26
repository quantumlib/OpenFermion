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


def _encoder_bk(dimension):
    """ Outputs the binary-tree (dimension x dimension)-matrix
    used for the encoder in the Bravyi-Kitaev transform.

    Args:
        dimension (int): length of the matrix, the dimension x dimension

    Returns (numpy.ndarray): encoder matrix
    """
    reps = int(numpy.ceil(numpy.log2(dimension)))
    mtx = numpy.array([[1, 0], [1, 1]])
    for repetition in numpy.arange(1, reps + 1):
        mtx = numpy.kron(numpy.eye(2, dtype=int), mtx)
        for column in numpy.arange(0, 2 ** repetition):
            mtx[2 ** (repetition + 1) - 1, column] = 1
    return mtx[0:dimension, 0:dimension]


def _decoder_bk(modes):
    """ Outputs the inverse of the binary tree matrix utilized for decoder
    in the Bravyi-Kitaev transform.

    Args:
        modes (int): size of the matrix is modes x modes

    Returns (numpy.ndarray): decoder matrix
    """
    reps = int(numpy.ceil(numpy.log2(modes)))
    mtx = numpy.array([[1, 0], [1, 1]])
    for repetition in numpy.arange(1, reps + 1):
        mtx = numpy.kron(numpy.eye(2, dtype=int), mtx)
        mtx[2 ** (repetition + 1) - 1, 2 ** repetition - 1] = 1
    return mtx[0:modes, 0:modes]


def _encoder_checksum(modes):
    """ Outputs the encoder matrix for checksum codes.

    Args:
        modes (int):  matrix size is (modes - 1) x modes

    Returns (numpy.ndarray): encoder matrix
    """
    enc = numpy.zeros(shape=(modes - 1, modes), dtype=int)
    for i in range(modes - 1):
        enc[i, i] = 1
    return enc


def _decoder_checksum(modes, odd):
    """ Outputs the decoder for checksum codes.

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


def checksum_code(modes, odd):
    """ Checksum code for either even or odd Hamming-weight
        
    Args:
        modes (int): number of modes
        odd (int or bool): 1 (True) or 0 (False), if odd,
            we encode all states with odd Hamming weight

    Returns (BinaryCode): The checksum BinaryCode
    """
    return BinaryCode(_encoder_checksum(modes), _decoder_checksum(modes, odd))


def jordan_wigner_code(modes):
    """ The Jordan-Wigner transform as binary code.
        
    Args:
        modes (int): number of modes

    Returns (BinaryCode): The Jordan-Wigner BinaryCode
    """
    return BinaryCode(numpy.identity(modes, dtype=int), linearize_decoder(
        numpy.identity(modes, dtype=int)))


def bravyi_kitaev_code(modes):
    """ The Bravyi-Kitaev transform as binary code.
        
    Args:
        modes (int): number of modes

    Returns (BinaryCode): The Bravyi-Kitaev BinaryCode
    """
    return BinaryCode(_encoder_bk(modes), 
                      linearize_decoder(_decoder_bk(modes)))


def parity_code(modes):
    """ The parity transform as binary code.
        
    Args:
        modes (int): number of modes

    Returns (BinaryCode): The parity transform BinaryCode
    """
    dec_mtx = numpy.reshape(([1] + [0] * (modes - 1)) +
                            ([1, 1] + (modes - 1) * [0]) * (modes - 2) +
                            [1, 1], (modes, modes))
    enc_mtx = numpy.tril(numpy.ones((modes, modes), dtype=int))

    return BinaryCode(enc_mtx, linearize_decoder(dec_mtx))


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


def weight_one_binary_addressing_code(exponent):
    """ Weight-1 binary addressing code.

    Note:
        This code is highly non-linear and might produce a lot of terms.

    Args:
        exponent (int):exponent for the number of modes N = 2 ^ exponent

    Returns (BinaryCode): the weight one binary addressing BinaryCode
    """
    encoder = numpy.zeros((exponent, 2 ** exponent), dtype=int)
    decoder = [0] * (2 ** exponent)
    for counter in numpy.arange(2 ** exponent):
        encoder[:, counter], decoder[counter] = \
            _binary_address(exponent, counter)
    return BinaryCode(encoder, decoder)


def weight_one_segment_code():
    """ Weight-1 segment code.

    Note:
        This code is highly non-linear and might produce a lot of terms.

    Returns (BinaryCode): weight one segment code
    """
    return BinaryCode([[1, 0, 1], [0, 1, 1]],
                      ['w0 w1 + w0', 'w0 w1 + w1', ' w0 w1'])


def weight_two_segment_code():
    """ Weight-2 segment code.

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
    """ Linear code that reorders orbitals to deal with an order of the modes 
    that is not 'half up', as it is referred to in arXiv:1403.1539.
    In 'half up', the first half of the modes corresponds to spin-up orbitals, 
    the second to spin-down which is a derivation from the convention in 
    OpenFermion, where spin-up and -down modes of the same spatial orbital are
    consecutive. In 'half up', one can append two instances of the same 
    code 'c' in order to have two symmetric subcodes that are symmetric for 
    spin-up and -down modes: ' c + c '.
    In the OpenFermion standard ordering, this requires the concatenation with
    the interleaved_code to have the same result:' interleaved_code * (c + c)'.
    This code changes the order of modes from (0, 1 , 2, ... , modes-1 )
    to (0, modes/2, 1 modes/2+1, ... , modes-1, modes/2 - 1).
    
    
    Args: modes (int): number of modes, must be even 
    
    Returns (BinaryCode): code that interleaves orbitals
    """
    if modes % 2 == 1:
        raise ValueError('number of modes must be even')
    else:
        mtx = numpy.zeros((modes, modes), dtype=int)
        for index in numpy.arange(modes//2, dtype=int):
            mtx[index, 2*index] = 1
            mtx[modes//2+index, 2*index+1] = 1
        return BinaryCode(mtx, linearize_decoder(mtx.transpose()))   


def half_up_order(fermion_operator, num_qubits=None, reverse=False):
    from openfermion.ops import FermionOperator, normal_ordered

    """ changes the fermionic order of the Hamiltonian from 0,1,2,...N to
    0,2,4,...,1,3,5,...
    Args:
        fermion_operator: The fermion_operator that is to be re-ordered
    Returns: Fermion operator with all-even all-odd ordering
    """
    if num_qubits is None:
        max_qubit = 0
        for term in fermion_operator.terms:
            for operator in term:
                if operator[0] > max_qubit:
                    max_qubit = operator[0]

        num_qubits = max_qubit + 1
    qubit_map = {2 * qidx + odd: qidx + odd * num_qubits for qidx in
                 range(num_qubits) for odd in [0, 1]}
    if reverse:
        qubit_map = {val: key for key, val in qubit_map.items()}

    rotated_hamiltonian = FermionOperator()
    for term, value in fermion_operator.terms.items():
        new_term = tuple([(qubit_map[op[0]], op[1]) for op in term])
        rotated_hamiltonian += FermionOperator(new_term, value)
    rotated_hamiltonian = normal_ordered(rotated_hamiltonian)
    return rotated_hamiltonian
    
