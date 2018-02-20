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

""" The transform function that does Fermion-qubit mappings
    based on a BinaryCode (arXiv:1712.07067) """

import numpy

from openfermion.ops import (BinaryCode,
                             FermionOperator,
                             QubitOperator,
                             SymbolicBinary)


def extractor(binary_op):
    """ Applies the extraction superoperator to a binary expression
     to obtain the corresponding qubit operators.

    Args:
        binary_op (SymbolicBinary): the binary term

    Returns (QubitOperator): the qubit operator corresponding to the
        binary terms
    """
    return_fn = 1
    for term in binary_op.terms:
        multiplier = 1
        if len(term) == 1:
            term = term[0]
            if isinstance(term, (numpy.int32, numpy.int64, int)):
                multiplier = QubitOperator('Z' + str(term))
            else:
                multiplier = -1
        elif len(term) > 1:
            multiplier = dissolve(term)

        return_fn *= multiplier
    return return_fn


def dissolve(term):
    """Decomposition helper. Takes a product of binary variables
    and outputs the Pauli-string sum that corresponds to the
    decomposed multi-qubit operator.

    Args:
        term (tuple): product of binary variables, i.e.: 'w0 w2 w3'

    Returns (QubitOperator): superposition of Pauli-strings

    Raises:
        ValueError: if the variable in term is not integer
    """
    prod = 2.0
    for var in term:
        if not isinstance(var, (numpy.int32, numpy.int64, int)):
            raise ValueError('dissolve only works on integers')
        prod *= (QubitOperator((), 0.5) - QubitOperator(
            'Z' + str(var), 0.5))
    return QubitOperator((), 1.0) - prod


def make_parity_list(code):
    """Create the parity list from the decoder of the input code.
    The output parity list has a similar structure as code.decoder. 

    Args:
        code (BinaryCode): the code to extract the parity list from.

    Returns (list): list of SymbolicBinary, the parity list

    Raises:
        TypeError: argument is not BinaryCode
    """
    if not isinstance(code, BinaryCode):
        raise TypeError('argument is not BinaryCode')
    parity_binaries = [SymbolicBinary()]
    for index in numpy.arange(code.n_modes - 1):
        parity_binaries += [parity_binaries[-1] + code.decoder[index]]
    return parity_binaries


def binary_code_transform(hamiltonian, code):
    """ Transforms a Hamiltonian written in fermionic basis into a Hamiltonian
    written in qubit basis, via a binary code.

    The role of the binary code is to relate the occupation vectors (v0 v1 v2
    ... vN-1) that span the fermionic basis, to the qubit basis, spanned by
    binary vectors (w0, w1, w2, ..., wn-1).

    The binary code has to provide an analytic relation between the binary
    vectors (v0, v1, ..., vN-1) and (w0, w1, ..., wn-1), and possibly has the
    property N>n, when the Fermion basis is smaller than the fermionic Fock
    space. The binary_code_transform function can transform Fermion operators
    to qubit operators for custom- and qubit-saving mappings.
    
    Note:
        Logic multi-qubit operators are decomposed into Pauli-strings (e.g.
        CPhase(1,2) = 0.5 * (1 + Z1 + Z2 - Z1 Z2 ) ), which might increase
        the number of Hamiltonian terms drastically.

    Args:
        hamiltonian (FermionOperator): the fermionic Hamiltonian
        code (BinaryCode): the binary code to transform the Hamiltonian

    Returns (QubitOperator): the transformed Hamiltonian

    Raises:
        TypeError: if the hamiltonian is not a FermionOperator or code is not
        a BinaryCode
    """
    if not isinstance(hamiltonian, FermionOperator):
        raise TypeError('hamiltonian provided must be a FermionOperator'
                        'received {}'.format(type(hamiltonian)))

    if not isinstance(code, BinaryCode):
        raise TypeError('code provided must be a BinaryCode'
                        'received {}'.format(type(code)))

    new_hamiltonian = QubitOperator()
    parity_list = make_parity_list(code)

    # for each term in hamiltonian
    for term, term_coefficient in hamiltonian.terms.items():

        """ the updated parity and occupation account for sign changes due
        changed occupations mid-way in the term """
        updated_parity = 0  # parity sign exponent
        parity_term = SymbolicBinary()
        changed_occupation_vector = [0] * code.n_modes
        transformed_term = QubitOperator(())

        # keep track of indices appeared before
        fermionic_indices = numpy.array([])

        # for each multiplier
        for op_idx, op_tuple in enumerate(reversed(term)):
            # get count exponent, parity exponent addition
            fermionic_indices = numpy.append(fermionic_indices, op_tuple[0])
            count = numpy.count_nonzero(
                fermionic_indices[:op_idx] == op_tuple[0])
            updated_parity += numpy.count_nonzero(
                fermionic_indices[:op_idx] < op_tuple[0])

            # update term
            extracted = extractor(code.decoder[op_tuple[0]])
            extracted *= (((-1) ** count) * ((-1) ** (op_tuple[1])) * 0.5)
            transformed_term *= QubitOperator((), 0.5) - extracted

            # update parity term and occupation vector
            changed_occupation_vector[op_tuple[0]] += 1
            parity_term += parity_list[op_tuple[0]]

        # the parity sign and parity term
        transformed_term *= QubitOperator((), (-1) ** updated_parity)
        transformed_term *= extractor(parity_term)

        # the update operator
        changed_qubit_vector = numpy.mod(code.encoder.dot(
            changed_occupation_vector), 2)
        for index, q_vec in enumerate(changed_qubit_vector):
            if q_vec:
                transformed_term *= QubitOperator('X' + str(index))

        # append new term to new hamiltonian
        new_hamiltonian += term_coefficient * transformed_term

    new_hamiltonian.compress()
    return new_hamiltonian
