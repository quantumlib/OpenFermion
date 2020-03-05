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

""" Binary code class for Fermion-qubit mappings (arXiv:1712.07067) """

import copy

import numpy
import scipy
import scipy.sparse

from openfermion.ops import BinaryPolynomial


def shift_decoder(decoder, shift_constant):
    """ Shifts the indices of a decoder by a constant.

    Args:
        decoder (iterable): list of BinaryPolynomial; the decoder
        shift_constant (int): the qubit index that corresponds to the offset.

    Returns (list): list of BinaryPolynomial shifted decoder
    """
    decode_shifted = []
    if not isinstance(shift_constant, (numpy.int64, numpy.int32, int)):
        raise TypeError('the shift to the decoder must be integer. got {}'
                        'of type {}'.format(shift_constant,
                                            type(shift_constant)))
    for entry in decoder:
        tmp_entry = copy.deepcopy(entry)
        tmp_entry.shift(shift_constant)
        decode_shifted.append(tmp_entry)
    return decode_shifted


def double_decoding(decoder_1, decoder_2):
    """ Concatenates two decodings

    Args:
        decoder_1 (iterable): list of BinaryPolynomial
            decoding of the outer code layer
        decoder_2 (iterable): list of BinaryPolynomial
            decoding of the inner code layer

    Returns (list): list of BinaryPolynomial the decoding defined by
        w -> decoder_1( decoder_2(w) )
    """
    doubled_decoder = []
    for entry in decoder_1:
        tmp_sum = 0
        for summand in entry.terms:
            tmp_term = BinaryPolynomial('1')
            for factor in summand:
                if isinstance(factor, (numpy.int32, numpy.int64, int)):
                    tmp_term *= decoder_2[factor]
            tmp_sum = tmp_term + tmp_sum
        doubled_decoder += [tmp_sum]
    return doubled_decoder


class BinaryCodeError(Exception):
    pass


class BinaryCode(object):
    r"""The BinaryCode class provides a representation of an encoding-decoding
    pair for binary vectors of different lengths, where the decoding is allowed
    to be non-linear.

    As the occupation number of fermionic mode is effectively binary,
    a length-N vector (v) of binary number can be utilized to describe
    a configuration of a many-body fermionic state on N modes.
    An n-qubit product state configuration \|w0> \|w1> \|w2> ... \|wn-1>,
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
    which means e": (v + v') -> e(v) + e'(v') and d": (w + w') -> d(w) + d'(
    w')
    where the addition is to be understood as appending two vectors together,
    so N" = N' + N and n" = n + n'.

    Appending codes is particularly useful when considering segment codes or
    segmented transforms.

    A BinaryCode-instance is initialized by BinaryCode(A,d),
    given the encoding (e) as n x N array or matrix-like nested lists A,
    such that e(v) = (A v) mod 2. The decoding d is an array or a list
    input of length N, which has entries either of type BinaryPolynomial, or of
    valid type for an input of the BinaryPolynomial-constructor.

    The signs + and \*, += and \*= are overloaded to implement concatenation
    and appendage on BinaryCode-objects.

    NOTE: multiplication of a BinaryCode with an integer yields a
        multiple appending of the same code, the multiplication with another
        BinaryCode their concatenation.

    Attributes:
        decoder (list):  list of BinaryPolynomial: Outputs the decoding
            functions as components.
        encoder (scipy.sparse.csc_matrix): Outputs A, the linear matrix that
            implements the encoding function.
        n_modes (int): Outputs the number of modes.
        n_qubits (int): Outputs the number of qubits.
    """

    def __init__(self, encoding, decoding):
        """ Initialization of a binary code.

        Args:
            encoding (np.ndarray or list): nested lists or binary 2D-array
            decoding (array or list): list of BinaryPolynomial (list or str).

        Raises:
            TypeError: non-list, array like encoding or decoding, unsuitable
                BinaryPolynomial generators,
            BinaryCodeError: in case of decoder/encoder size mismatch or
                decoder size, qubits indexed mismatch
        """
        if not isinstance(encoding, (numpy.ndarray, list)):
            raise TypeError('encoding must be a list or array.')

        if not isinstance(decoding, (numpy.ndarray, list)):
            raise TypeError('decoding must be a list or array.')

        self.encoder = scipy.sparse.csc_matrix(encoding)
        self.n_qubits, self.n_modes = numpy.shape(encoding)

        if self.n_modes != len(decoding):
            raise BinaryCodeError(
                'size mismatch, decoder and encoder should have the same'
                ' first dimension')

        decoder_qubits = set()
        self.decoder = []

        for symbolic_binary in decoding:
            if isinstance(symbolic_binary, (tuple, list, str, int,
                                            numpy.int32, numpy.int64)):
                symbolic_binary = BinaryPolynomial(symbolic_binary)
            if isinstance(symbolic_binary, BinaryPolynomial):
                self.decoder.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(
                    symbolic_binary.enumerate_qubits())
            else:
                raise TypeError(
                    'decoder component provided '
                    'is not a suitable for BinaryPolynomial',
                    symbolic_binary)

        if len(decoder_qubits) != self.n_qubits:
            raise BinaryCodeError(
                'decoder and encoder provided has different number of qubits')

        if max(decoder_qubits) + 1 > self.n_qubits:
            raise BinaryCodeError('decoder is not indexing some qubits. Qubits'
                                  'indexed are: {}'.format(decoder_qubits))

    def __iadd__(self, appendix):
        """ In-place appending a binary code with +=.

        Args:
            appendix (BinaryCode): The code to append to the present one.

        Returns (BinaryCode): A global binary code with size
            (n_modes1 + n_modes2), (n_qubits1,n_qubits2)

        Raises:
            TypeError: Appendix must be a BinaryCode.
        """
        if not isinstance(appendix, BinaryCode):
            raise TypeError('argument must be a BinaryCode.')

        self.decoder = numpy.append(self.decoder,
                                    shift_decoder(appendix.decoder,
                                                  self.n_qubits)).tolist()
        self.encoder = scipy.sparse.bmat([[self.encoder, None],
                                          [None, appendix.encoder]])
        self.n_qubits, self.n_modes = numpy.shape(self.encoder)
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
        if not isinstance(factor, (BinaryCode, numpy.int32, numpy.int64, int)):
            raise TypeError('argument must be a BinaryCode or integer')

        if isinstance(factor, BinaryCode):
            if self.n_qubits != factor.n_modes:
                raise BinaryCodeError(
                    'size mismatch between inner and outer code layer')

            self.decoder = double_decoding(self.decoder, factor.decoder)
            self.encoder = factor.encoder.dot(self.encoder)
            self.n_qubits, self.n_modes = numpy.shape(self.encoder)
            return self

        elif isinstance(factor, (numpy.int32, numpy.int64, int)):
            if factor < 1:
                raise ValueError('integer factor has to be positive, '
                                 'non-zero ')

            self.encoder = scipy.sparse.kron(
                scipy.sparse.identity(factor, format='csc', dtype=int),
                self.encoder, 'csc')
            tmp_decoder = self.decoder
            for index in numpy.arange(1, factor):
                self.decoder = numpy.append(self.decoder,
                                            shift_decoder(tmp_decoder,
                                                          index *
                                                          self.n_qubits))
            self.n_qubits *= factor
            self.n_modes *= factor
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

        Returns (BinaryCode): Segmented code.

        Raises:
            TypeError: factor must be an integer
        """
        if isinstance(factor, (numpy.int32, numpy.int64, int)):
            return self * factor
        else:
            raise TypeError('the left multiplier must be an integer to a'
                            'BinaryCode. Was given {} of '
                            'type {}'.format(factor, type(factor)))

    def __str__(self):
        """ Return an easy-to-read string representation."""
        string_return = [list(map(list, self.encoder.toarray()))]

        dec_str = '['
        for term in self.decoder:
            dec_str += term.__str__() + ','
        dec_str = dec_str[:-1]
        string_return.append(dec_str + ']')
        return str(string_return)

    def __repr__(self):
        return str(self)
