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

from openfermion.ops import BinaryCode, QubitOperator, SymbolicBinary


def _extract(term):
    """Extractor superoperator helper function.

    Args:
        term (SymbolicBinary): a single summand from a binary expression

    Returns (QubitOperator or int): the qubit operator that corresponds
        to the binary expression or -1
    """
    if len(term) == 1:
        term = term[0]
        if term[1] == 'W':
            return QubitOperator('Z' + str(term[0]))
        elif term[1] == '1':
            return -1
    if len(term) > 1:
        return dissolve(term)


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
        return_fn *= _extract(term)
    return return_fn


def dissolve(term):
    """Decomposition helper. Takes a product of binary variables
    and outputs the Pauli-string sum that corresponds to the 
    decomposed multi-qubit operator.

.    Args:
        term (SymbolicBinary): product of binary variables, i.e.: 'w0 w2 w3'

    Returns (QubitOperator): superposition of Pauli-strings

    Raises:
        ValueError: if the action is not 'W'
    """
    prod = 2.0
    for var in term:
        if var[1] != 'W':
            raise ValueError('dissolve only works on action W')
        prod *= (QubitOperator((), 0.5) - QubitOperator(
            'Z' + str(var[0]), 0.5))
    return QubitOperator((), 1.0) - prod


def make_parity_list(code):
    """Create the parity list from the decoder of the input code.
    The output parity list has a similar structure as code.decoder. 

    Args:
        code (BinaryCode): the code to extract the parity list from.

    Returns (list): list of SymbolicBinary the parity list

    Raises:
        TypeError: argument is not BinaryCode
    """
    if not isinstance(code, BinaryCode):
        raise TypeError('argument is not BinaryCode')
    parity_binaries = [SymbolicBinary()]
    for index in numpy.arange(code.n_modes - 1):
        parity_binaries += [parity_binaries[-1] + code.decoder[index]]
    return parity_binaries


def code_transform(hamiltonian, code):
    """Transforms a Hamiltonian of Fermions into a Hamiltonian of qubits,
    via a binary code. 
    The role of the binary code is to relate the 
    occupation vectors (v0 v1 v2 ... vN-1) that span the fermionic
    basis, where e.g. the vector (1,0,1,1,0,0) yields the state
    
    a_0^\dagger  a_2^\dagger  a_3^\dagger |vac>,
    
    to the qubit basis, spanned by binary vectors
    (w0, w1, w2, ..., wn-1) determining a product state 
    |w0> |w1> |w2> ... |wn-1>. 
    
    The binary code has to provide an analytic relation between 
    between the binary vectors (v0, v1, ..., vN-1) and 
    (w0, w1, ..., wn-1), and possibly has the property 
    N>n, when the Fermion basis is smaller than the fermionic Fock space.
    In this way, the code_transform function can transform Fermion operators
    to qubit operators for customized and qubit-saving mappings.
    NOTE: Logic multi-qubit operators are decomposed 
    (e.g. CPhase(1,2) = 0.5 * (1 + Z1 + Z2 - Z1 Z2 ) ), which might increase
    the number of Hamiltonian terms drastically. 
    
    Args:
        hamiltonian: (FermionOperator) the fermionic Hamiltonian
        code: (BinaryCode) the binary code to transform the Hamiltonian
    
    Returns: (QubitOperator) the transformed Hamiltonian
        
    Raises:
        TypeError: if the hamiltonian is not a FermionOperator or code is
            not a BinaryCode
    """
    if not isinstance(hamiltonian, FermionOperator):
        raise TypeError('hamiltonian provided must be a FermionOperator'
                        'received {}'.format(type(hamiltonian)))

    if not isinstance(code, BinaryCode):
        raise TypeError('code provided must be a BinaryCode'
                        'received {}'.format(type(code)))

    new_hamiltonian = QubitOperator()
    parity_list = make_parity_list(code)

    for term in hamiltonian.terms:
        num_operators = len(term)

        # the updated parity and occupation account for sign changes due
        # to changed occupations mid-way in the term
        updated_parity = 0
        updated_occupation = numpy.ones(num_operators, dtype=int)
        for main_index in numpy.arange(1, num_operators + 1):

            tmp_occupation = 0
            for runner_index in numpy.arange(1, main_index):
                if term[-runner_index][0] == term[-main_index][0]:
                    tmp_occupation += 1

                if term[-runner_index][0] < term[-main_index][
                    0]: updated_parity += 1
            updated_occupation[-main_index] = (-1) ** (tmp_occupation)

        # generating the parity and projection operator(s)
        transformed_term = QubitOperator((), (-1) ** (updated_parity))
        parity_term = SymbolicBinary()

        for index in numpy.arange(num_operators):
            transformed_term *= .5 * (
                QubitOperator(()) - updated_occupation[index] * (-1) ** (
                    term[index][1]) * extractor(code.decoder[term[index][0]]))
            parity_term += parity_list[term[index][0]]  # update
        transformed_term *= extractor(parity_term)
        transformed_term.compress()
        # generating the update operator
        update_operator = QubitOperator(())
        changed_occupation_vector = numpy.zeros(code.n_modes, dtype=int)

        for op in term:
            changed_occupation_vector[op[0]] += 1
        changed_qubit_vector = code.encoder.dot(
            numpy.array(changed_occupation_vector)) % 2

        for index in numpy.arange(code.n_qubits):
            if changed_qubit_vector[index]:
                update_operator *= QubitOperator('X' + str(index))

        transformed_term = hamiltonian.terms[
                               term] * update_operator * transformed_term
        new_hamiltonian += transformed_term
    return new_hamiltonian


# A BIT FASTER VERSION OF CODE_TRANSFORMS NEEDS TESTING AND TIMING

def code_transform2(hamiltonian, code):
    """Transforms a Hamiltonian of Fermions into a Hamiltonian of qubits,
    via a binary code.
    The role of the binary code is to relate the
    occupation vectors (v0 v1 v2 ... vN-1) that span the fermionic
    basis, where e.g. the vector (1,0,1,1,0,0) yields the state

    a_0^\dagger  a_2^\dagger  a_3^\dagger |vac>,

    to the qubit basis, spanned by binary vectors
    (w0, w1, w2, ..., wn-1) determining a product state
    |w0> |w1> |w2> ... |wn-1>.

    The binary code has to provide an analytic relation between
    between the binary vectors (v0, v1, ..., vN-1) and
    (w0, w1, ..., wn-1), and possibly has the property
    N>n, when the Fermion basis is smaller than the fermionic Fock space.
    In this way, the code_transform function can transform Fermion operators
    to qubit operators for customized and qubit-saving mappings.
    NOTE: Logic multi-qubit operators are decomposed
    (e.g. CPhase(1,2) = 0.5 * (1 + Z1 + Z2 - Z1 Z2 ) ), which might increase
    the number of Hamiltonian terms drastically.

    Args:
        hamiltonian: (FermionOperator) the fermionic Hamiltonian
        code: (BinaryCode) the binary code to transform the Hamiltonian

    Returns: (QubitOperator) the transformed Hamiltonian

    Raises:
        TypeError: if the hamiltonian is not a FermionOperator or code is
            not a BinaryCode
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
    for term, term_coeff in hamiltonian.terms.items():

        # the updated parity and occupation account for sign changes due
        # changed occupations mid-way in the term
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
        new_hamiltonian += term_coeff * transformed_term

    new_hamiltonian.compress()
    return new_hamiltonian


if __name__ == '__main__':
    from decoder_encoder_functions import BK_code, JW_code
    from openfermion.transforms import bravyi_kitaev
    from openfermion.ops._fermion_operator import FermionOperator
    from openfermion.utils import eigenspectrum
    import time

    code1 = BinaryCode(numpy.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),
                       [SymbolicBinary('w0'), SymbolicBinary('w0 + w1 + 1'),
                        SymbolicBinary('w0 w1 w2')])

    from openfermion.hamiltonians import MolecularData
    from openfermion.transforms import get_fermion_operator

    diatomic_bond_length = 1.45
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
    basis = 'sto-3g'
    multiplicity = 1
    active_space_start = 1
    active_space_stop = 3
    molecule = MolecularData(geometry, basis, multiplicity, description="1.45")
    molecule.load()
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=range(active_space_start),
        active_indices=range(active_space_start, active_space_stop))
    hamil1 = get_fermion_operator(molecular_hamiltonian)

    code1 = BK_code(10)
    print 'fermionic',eigenspectrum(hamil1)
    print 'openfermion bk',eigenspectrum(bravyi_kitaev(hamil1))
    st = time.time()
    a = code_transform(hamil1, code1)
    print 'original:', time.time() - st

    st = time.time()
    b = code_transform2(hamil1, code1)
    print 'code2', time.time() - st

    print 'transform:', eigenspectrum(a)
    print '\n'
    print 'code2:', eigenspectrum(b)

    print ('\n______________\n')
    st = time.time()
    a = code_transform(hamil1, JW_code(4))
    print 'original:', time.time() - st

    st = time.time()
    b = code_transform2(hamil1, JW_code(4))
    print 'code2', time.time() - st

    print 'transform:', eigenspectrum(a)
    print '\n'
    print 'code2', eigenspectrum(b)
