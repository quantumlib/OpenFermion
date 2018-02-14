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
    return BinaryCode(_encoder_bk(modes), linearize_decoder(_decoder_bk(modes)))


def parity_code(modes):
    """ The parity transform as binary code.
        
    Args:
        modes (int): number of modes

    Returns (BinaryCode): The parity transform BinaryCode
    """
    dec_mtx = numpy.reshape(([1] + [0] * (modes - 1)) +
                            ([1, 1] + (modes - 1) * [0]) * (modes - 2) + [1, 1],
                            (modes, modes))
    enc_mtx = numpy.tril(numpy.ones((modes, modes), dtype=int))

    return BinaryCode(enc_mtx, linearize_decoder(dec_mtx))


def binary_address(digits, address):
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
            'w' + str(index) + ' + 1 ' + address[index])

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
            binary_address(exponent, counter)
    return BinaryCode(encoder, decoder)


def weight_one_segment_code():
    """ Weight-1 segment code.

    Note:
        This code is highly non-linear and might produce a lot of terms.

    Returns (BinaryCode): weight one segment code
    """
    return BinaryCode([[1, 0, 1], [0, 1, 1]],
                      ['w0 w1 + w1', 'w0 w1 + w0', ' w0 w1'])


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


if __name__ == '__main__':
    from numpy.linalg import inv

    a = _decoder_checksum(5, 1)
    print type(a), a
    a = _decoder_checksum(5, True)
    print type(a), a

    # code1 = BinaryCode(numpy.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),
    #                    [SymbolicBinary('w0'), SymbolicBinary('w0 + w1 + 1'),
    #                     SymbolicBinary('w0 w1 w2')])
    #
    # from openfermion.hamiltonians import MolecularData
    # from openfermion.transforms import get_fermion_operator
    #
    # diatomic_bond_length = 1.45
    # geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
    # basis = 'sto-3g'
    # multiplicity = 1
    # active_space_start = 1
    # active_space_stop = 3
    # molecule = MolecularData(geometry, basis, multiplicity, 
    #                            description="1.45")
    # molecule.load()
    # molecular_hamiltonian = molecule.get_molecular_hamiltonian(
    #     occupied_indices=range(active_space_start),
    #     active_indices=range(active_space_start, active_space_stop))
    # hamil1 = get_fermion_operator(molecular_hamiltonian)
    #
    # code1 = BK_code(10)
    # print 'fermionic',eigenspectrum(hamil1)
    # print 'openfermion bk',eigenspectrum(bravyi_kitaev(hamil1))
    # st = time.time()
    # a = code_transform(hamil1, code1)
    # print 'original:', time.time() - st
    #
    # st = time.time()
    # print 'code2', time.time() - st
    #
    # print 'transform:', eigenspectrum(a)
    # print '\n'
    #
    # print ('\n______________\n')
    # st = time.time()
    # a = code_transform(hamil1, JW_code(4))
    # print 'original:', time.time() - st
    #
    # st = time.time()
    # print 'code2', time.time() - st
    #
    # print 'transform:', eigenspectrum(a)
    # print '\n'
