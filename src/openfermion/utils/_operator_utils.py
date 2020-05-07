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

"""This module provides generic tools for classes in ops/"""
from builtins import map, zip
import copy
import itertools
import marshal
import os

import numpy
from numpy.random import RandomState
from scipy.sparse import spmatrix

from openfermion.config import DATA_DIRECTORY, EQ_TOLERANCE
from openfermion.ops import (BosonOperator,
                             DiagonalCoulombHamiltonian,
                             FermionOperator,
                             InteractionOperator,
                             InteractionRDM,
                             MajoranaOperator,
                             QuadOperator,
                             QubitOperator,
                             PolynomialTensor)


class OperatorUtilsError(Exception):
    pass


class OperatorSpecificationError(Exception):
    pass


def chemist_ordered(fermion_operator):
    r"""Puts a two-body fermion operator in chemist ordering.

    The normal ordering convention for chemists is different.
    Rather than ordering the two-body term as physicists do, as
    :math:`a^\dagger a^\dagger a a`
    the chemist ordering of the two-body term is
    :math:`a^\dagger a a^\dagger a`

    TODO: This routine can be made more efficient.

    Args:
        fermion_operator (FermionOperator): a fermion operator guarenteed to
            have number conserving one- and two-body fermion terms only.
    Returns:
        chemist_ordered_operator (FermionOperator): the input operator
            ordered in the chemistry convention.
    Raises:
        OperatorSpecificationError: Operator is not two-body number conserving.
    """
    # Make sure we're dealing with a fermion operator from a molecule.
    if not fermion_operator.is_two_body_number_conserving():
        raise OperatorSpecificationError(
            'Operator is not two-body number conserving.')

    # Normal order and begin looping.
    normal_ordered_input = normal_ordered(fermion_operator)
    chemist_ordered_operator = FermionOperator()
    for term, coefficient in normal_ordered_input.terms.items():
        if len(term) == 2 or not len(term):
            chemist_ordered_operator += FermionOperator(term, coefficient)
        else:
            # Possibly add new one-body term.
            if term[1][0] == term[2][0]:
                new_one_body_term = (term[0], term[3])
                chemist_ordered_operator += FermionOperator(
                    new_one_body_term, coefficient)
            # Reorder two-body term.
            new_two_body_term = (term[0], term[2], term[1], term[3])
            chemist_ordered_operator += FermionOperator(
                new_two_body_term, -coefficient)
    return chemist_ordered_operator


def inline_sum(summands, seed):
    """Computes a sum, using the __iadd__ operator.
    Args:
        seed (T): The starting total. The zero value.
        summands (iterable[T]): Values to add (with +=) into the total.
    Returns:
        T: The result of adding all the factors into the zero value.
    """
    for r in summands:
        seed += r
    return seed


def freeze_orbitals(fermion_operator, occupied, unoccupied=None, prune=True):
    """Fix some orbitals to be occupied and others unoccupied.

    Removes all operators acting on the specified orbitals, and renumbers the
    remaining orbitals to eliminate unused indices. The sign of each term
    is modified according to the ladder uperator anti-commutation relations in
    order to preserve the expectation value of the operator.

    Args:
        occupied: A list containing the indices of the orbitals that are to be
            assumed to be occupied.
        unoccupied: A list containing the indices of the orbitals that are to
            be assumed to be unoccupied.
    """
    new_operator = fermion_operator
    frozen = [(index, 1) for index in occupied]
    if unoccupied is not None:
        frozen += [(index, 0) for index in unoccupied]

    # Loop over each orbital to be frozen. Within each term, move all
    # ops acting on that orbital to the right side of the term, keeping
    # track of sign flips that come from swapping operators.
    for item in frozen:
        tmp_operator = FermionOperator()
        for term in new_operator.terms:
            new_term = []
            new_coef = new_operator.terms[term]
            current_occupancy = item[1]
            n_ops = 0  # Number of operations on index that have been moved
            n_swaps = 0  # Number of swaps that have been done

            for op in enumerate(reversed(term)):
                if op[1][0] is item[0]:
                    n_ops += 1

                    # Determine number of swaps needed to bring the op in
                    # front of all ops acting on other indices
                    n_swaps += op[0] - n_ops

                    # Check if the op annihilates the current state
                    if current_occupancy == op[1][1]:
                        new_coef = 0

                    # Update current state
                    current_occupancy = (current_occupancy + 1) % 2
                else:
                    new_term.insert(0, op[1])
            if n_swaps % 2:
                new_coef *= -1
            if new_coef and current_occupancy == item[1]:
                tmp_operator += FermionOperator(tuple(new_term), new_coef)
        new_operator = tmp_operator

    # For occupied frozen orbitals, we must also bring together the creation
    # operator from the ket and the annihilation operator from the bra when
    # evaluating expectation values. This can result in an additional minus
    # sign.
    for term in new_operator.terms:
        for index in occupied:
            for op in term:
                if op[0] > index:
                    new_operator.terms[term] *= -1

    # Renumber indices to remove frozen orbitals
    new_operator = prune_unused_indices(new_operator)

    return new_operator


def prune_unused_indices(symbolic_operator):
    """
    Remove indices that do not appear in any terms.

    Indices will be renumbered such that if an index i does not appear in
    any terms, then the next largest index that appears in at least one
    term will be renumbered to i.
    """

    # Determine which indices appear in at least one term
    indices = []
    for term in symbolic_operator.terms:
        for op in term:
            if op[0] not in indices:
                indices.append(op[0])
    indices.sort()

    # Construct a dict that maps the old indices to new ones
    index_map = {}
    for index in enumerate(indices):
        index_map[index[1]] = index[0]

    new_operator = copy.deepcopy(symbolic_operator)
    new_operator.terms.clear()

    # Replace the indices in the terms with the new indices
    for term in symbolic_operator.terms:
        new_term = [(index_map[op[0]], op[1]) for op in term]
        new_operator.terms[tuple(new_term)] = symbolic_operator.terms[term]

    return new_operator


def hermitian_conjugated(operator):
    """Return Hermitian conjugate of operator."""
    # Handle FermionOperator
    if isinstance(operator, FermionOperator):
        conjugate_operator = FermionOperator()
        for term, coefficient in operator.terms.items():
            conjugate_term = tuple([(tensor_factor, 1 - action) for
                                    (tensor_factor, action) in reversed(term)])
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Handle BosonOperator
    elif isinstance(operator, BosonOperator):
        conjugate_operator = BosonOperator()
        for term, coefficient in operator.terms.items():
            conjugate_term = tuple([(tensor_factor, 1 - action) for
                                    (tensor_factor, action) in reversed(term)])
            # take into account that different indices commute
            conjugate_term = tuple(
                sorted(conjugate_term, key=lambda factor: factor[0]))
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Handle QubitOperator
    elif isinstance(operator, QubitOperator):
        conjugate_operator = QubitOperator()
        for term, coefficient in operator.terms.items():
            conjugate_operator.terms[term] = coefficient.conjugate()

    # Handle QuadOperator
    elif isinstance(operator, QuadOperator):
        conjugate_operator = QuadOperator()
        for term, coefficient in operator.terms.items():
            conjugate_term = reversed(term)
            # take into account that different indices commute
            conjugate_term = tuple(
                sorted(conjugate_term, key=lambda factor: factor[0]))
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Handle InteractionOperator
    elif isinstance(operator, InteractionOperator):
        conjugate_constant = operator.constant.conjugate()
        conjugate_one_body_tensor = hermitian_conjugated(
            operator.one_body_tensor)
        conjugate_two_body_tensor = hermitian_conjugated(
            operator.two_body_tensor)
        conjugate_operator = type(operator)(conjugate_constant,
            conjugate_one_body_tensor, conjugate_two_body_tensor)

    # Handle sparse matrix
    elif isinstance(operator, spmatrix):
        conjugate_operator = operator.getH()

    # Handle numpy array
    elif isinstance(operator, numpy.ndarray):
        conjugate_operator = operator.T.conj()

    # Unsupported type
    else:
        raise TypeError('Taking the hermitian conjugate of a {} is not '
                        'supported.'.format(type(operator).__name__))

    return conjugate_operator


def is_hermitian(operator):
    """Test if operator is Hermitian."""
    # Handle FermionOperator, BosonOperator, and InteractionOperator
    if isinstance(operator,
            (FermionOperator, BosonOperator, InteractionOperator)):
        return (normal_ordered(operator) ==
                normal_ordered(hermitian_conjugated(operator)))

    # Handle QubitOperator and QuadOperator
    if isinstance(operator, (QubitOperator, QuadOperator)):
        return operator == hermitian_conjugated(operator)

    # Handle sparse matrix
    elif isinstance(operator, spmatrix):
        difference = operator - hermitian_conjugated(operator)
        discrepancy = 0.
        if difference.nnz:
            discrepancy = max(abs(difference.data))
        return discrepancy < EQ_TOLERANCE

    # Handle numpy array
    elif isinstance(operator, numpy.ndarray):
        difference = operator - hermitian_conjugated(operator)
        discrepancy = numpy.amax(abs(difference))
        return discrepancy < EQ_TOLERANCE

    # Unsupported type
    else:
        raise TypeError('Checking whether a {} is hermitian is not '
                        'supported.'.format(type(operator).__name__))


def count_qubits(operator):
    """Compute the minimum number of qubits on which operator acts.

    Args:
        operator: FermionOperator, QubitOperator, DiagonalCoulombHamiltonian,
            or PolynomialTensor.

    Returns:
        num_qubits (int): The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.
    """
    # Handle FermionOperator.
    if isinstance(operator, FermionOperator):
        num_qubits = 0
        for term in operator.terms:
            for ladder_operator in term:
                if ladder_operator[0] + 1 > num_qubits:
                    num_qubits = ladder_operator[0] + 1
        return num_qubits

    # Handle QubitOperator.
    elif isinstance(operator, QubitOperator):
        num_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > num_qubits:
                    num_qubits = term[-1][0] + 1
        return num_qubits

    # Handle MajoranaOperator.
    if isinstance(operator, MajoranaOperator):
        num_qubits = 0
        for term in operator.terms:
            for majorana_index in term:
                if numpy.ceil((majorana_index+1) / 2) > num_qubits:
                    num_qubits = int(numpy.ceil((majorana_index+1) / 2))
        return num_qubits

    # Handle DiagonalCoulombHamiltonian
    elif isinstance(operator, DiagonalCoulombHamiltonian):
        return operator.one_body.shape[0]

    # Handle PolynomialTensor
    elif isinstance(operator, PolynomialTensor):
        return operator.n_qubits

    # Raise for other classes.
    else:
        raise TypeError('Operator of invalid type.')


def eigenspectrum(operator, n_qubits=None):
    """Compute the eigenspectrum of an operator.

    WARNING: This function has cubic runtime in dimension of
        Hilbert space operator, which might be exponential.

    NOTE: This function does not currently support
        QuadOperator and BosonOperator.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            PolynomialTensor, or InteractionRDM.
        n_qubits (int): number of qubits/modes in operator. if None, will
            be counted.

    Returns:
        spectrum: dense numpy array of floats giving eigenspectrum.
    """
    if isinstance(operator, (QuadOperator, BosonOperator)):
        raise TypeError('Operator of invalid type.')
    from openfermion.transforms import get_sparse_operator
    from openfermion.utils import sparse_eigenspectrum
    sparse_operator = get_sparse_operator(operator, n_qubits)
    spectrum = sparse_eigenspectrum(sparse_operator)
    return spectrum


def is_identity(operator):
    """Check whether QubitOperator of FermionOperator is identity.

    Args:
        operator: QubitOperator, FermionOperator,
            BosonOperator, or QuadOperator.

    Raises:
        TypeError: Operator of invalid type.
    """
    if isinstance(operator, (QubitOperator, FermionOperator,
                             BosonOperator, QuadOperator)):
        return list(operator.terms) == [()]
    raise TypeError('Operator of invalid type.')


def _fourier_transform_helper(hamiltonian,
                              grid,
                              spinless,
                              phase_factor,
                              vec_func_1,
                              vec_func_2):
    hamiltonian_t = FermionOperator.zero()
    normalize_factor = numpy.sqrt(1.0 / float(grid.num_points))

    for term in hamiltonian.terms:
        transformed_term = FermionOperator.identity()
        for ladder_op_mode, ladder_op_type in term:
            indices_1 = grid.grid_indices(ladder_op_mode, spinless)
            vec1 = vec_func_1(indices_1)
            new_basis = FermionOperator.zero()
            for indices_2 in grid.all_points_indices():
                vec2 = vec_func_2(indices_2)
                spin = None if spinless else ladder_op_mode % 2
                orbital = grid.orbital_id(indices_2, spin)
                exp_index = phase_factor * 1.0j * numpy.dot(vec1, vec2)
                if ladder_op_type == 1:
                    exp_index *= -1.0

                element = FermionOperator(((orbital, ladder_op_type),),
                                          numpy.exp(exp_index))
                new_basis += element

            new_basis *= normalize_factor
            transformed_term *= new_basis

        # Coefficient.
        transformed_term *= hamiltonian.terms[term]

        hamiltonian_t += transformed_term

    return hamiltonian_t


def fourier_transform(hamiltonian, grid, spinless):
    r"""Apply Fourier transform to change hamiltonian in plane wave basis.

    .. math::

        c^\dagger_v = \sqrt{1/N} \sum_m {a^\dagger_m \exp(-i k_v r_m)}
        c_v = \sqrt{1/N} \sum_m {a_m \exp(i k_v r_m)}

    Args:
        hamiltonian (FermionOperator): The hamiltonian in plane wave basis.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The fourier-transformed hamiltonian.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     grid=grid,
                                     spinless=spinless,
                                     phase_factor=+1,
                                     vec_func_1=grid.momentum_vector,
                                     vec_func_2=grid.position_vector)


def get_file_path(file_name, data_directory):
    """Compute file_path for the file that stores operator.

    Args:
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
                        directory specified in config file.

    Returns:
        file_path (string): File path.

    Raises:
        OperatorUtilsError: File name is not provided.
    """
    if file_name:
        if file_name[-5:] != '.data':
            file_name = file_name + ".data"
    else:
        raise OperatorUtilsError("File name is not provided.")

    if data_directory is None:
        file_path = DATA_DIRECTORY + '/' + file_name
    else:
        file_path = data_directory + '/' + file_name

    return file_path


def inverse_fourier_transform(hamiltonian, grid, spinless):
    r"""Apply inverse Fourier transform to change hamiltonian in
    plane wave dual basis.

    .. math::

        a^\dagger_v = \sqrt{1/N} \sum_m {c^\dagger_m \exp(i k_v r_m)}
        a_v = \sqrt{1/N} \sum_m {c_m \exp(-i k_v r_m)}

    Args:
        hamiltonian (FermionOperator):
            The hamiltonian in plane wave dual basis.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The inverse-fourier-transformed hamiltonian.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     grid=grid,
                                     spinless=spinless,
                                     phase_factor=-1,
                                     vec_func_1=grid.position_vector,
                                     vec_func_2=grid.momentum_vector)


def load_operator(file_name=None, data_directory=None, plain_text=False):
    """Load FermionOperator or QubitOperator from file.

    Args:
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
                        directory specified in config file.
        plain_text: Whether the input file is plain text

    Returns:
        operator: The stored FermionOperator, BosonOperator,
            QuadOperator, or QubitOperator

    Raises:
        TypeError: Operator of invalid type.
    """
    file_path = get_file_path(file_name, data_directory)

    if plain_text:
        with open(file_path, 'r') as f:
            data = f.read()
            operator_type, operator_terms = data.split(":\n")

        if operator_type == 'FermionOperator':
            operator = FermionOperator(operator_terms)
        elif operator_type == 'BosonOperator':
            operator = BosonOperator(operator_terms)
        elif operator_type == 'QubitOperator':
            operator = QubitOperator(operator_terms)
        elif operator_type == 'QuadOperator':
            operator = QuadOperator(operator_terms)
        else:
            raise TypeError('Operator of invalid type.')
    else:
        with open(file_path, 'rb') as f:
            data = marshal.load(f)
            operator_type = data[0]
            operator_terms = data[1]

        if operator_type == 'FermionOperator':
            operator = FermionOperator()
            for term in operator_terms:
                operator += FermionOperator(term, operator_terms[term])
        elif operator_type == 'BosonOperator':
            operator = BosonOperator()
            for term in operator_terms:
                operator += BosonOperator(term, operator_terms[term])
        elif operator_type == 'QubitOperator':
            operator = QubitOperator()
            for term in operator_terms:
                operator += QubitOperator(term, operator_terms[term])
        elif operator_type == 'QuadOperator':
            operator = QuadOperator()
            for term in operator_terms:
                operator += QuadOperator(term, operator_terms[term])
        else:
            raise TypeError('Operator of invalid type.')

    return operator


def save_operator(operator, file_name=None, data_directory=None,
                  allow_overwrite=False, plain_text=False):
    """Save FermionOperator or QubitOperator to file.

    Args:
        operator: An instance of FermionOperator, BosonOperator,
            or QubitOperator.
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
                        directory specified in config file.
        allow_overwrite: Whether to allow files to be overwritten.
        plain_text: Whether the operator should be saved to a
                        plain-text format for manual analysis

    Raises:
        OperatorUtilsError: Not saved, file already exists.
        TypeError: Operator of invalid type.
    """
    file_path = get_file_path(file_name, data_directory)

    if os.path.isfile(file_path) and not allow_overwrite:
        raise OperatorUtilsError("Not saved, file already exists.")

    if isinstance(operator, FermionOperator):
        operator_type = "FermionOperator"
    elif isinstance(operator, BosonOperator):
        operator_type = "BosonOperator"
    elif isinstance(operator, QubitOperator):
        operator_type = "QubitOperator"
    elif isinstance(operator, QuadOperator):
        operator_type = "QuadOperator"
    elif isinstance(operator, (InteractionOperator, InteractionRDM)):
        raise NotImplementedError('Not yet implemented for '
                                  'InteractionOperator or InteractionRDM.')
    else:
        raise TypeError('Operator of invalid type.')

    if plain_text:
        with open(file_path, 'w') as f:
            f.write(operator_type + ":\n" + str(operator))
    else:
        tm = operator.terms
        with open(file_path, 'wb') as f:
            marshal.dump((operator_type,
                          dict(zip(tm.keys(), map(complex, tm.values())))), f)


def reorder(operator, order_function, num_modes=None, reverse=False):
    """Changes the ladder operator order of the Hamiltonian based on the
    provided order_function per mode index.

    Args:
        operator (SymbolicOperator): the operator that will be reordered. must
            be a SymbolicOperator or any type of operator that inherits from
            SymbolicOperator.
        order_function (func): a function per mode that is used to map the
            indexing. must have arguments mode index and num_modes.
        num_modes (int): default None. User can provide the number of modes
            assumed for the system. if None, the number of modes will be
            calculated based on the Operator.
        reverse (bool): default False. if set to True, the mode mapping is
            reversed. reverse = True will not revert back to original if
            num_modes calculated differs from original and reverted.

    Note: Every order function must take in a mode_idx and num_modes.
    """

    if num_modes is None:
        num_modes = max(
            [factor[0] for term in operator.terms for factor in term]) + 1

    mode_map = {mode_idx: order_function(mode_idx, num_modes) for mode_idx in
                range(num_modes)}

    if reverse:
        mode_map = {val: key for key, val in mode_map.items()}

    rotated_hamiltonian = operator.__class__()
    for term, value in operator.terms.items():
        new_term = tuple([(mode_map[op[0]], op[1]) for op in term])
        rotated_hamiltonian += operator.__class__(new_term, value)
    return rotated_hamiltonian


def up_then_down(mode_idx, num_modes):
    """ up then down reordering, given the operator has the default even-odd
     ordering. Otherwise this function will reorder indices where all even
     indices now come before odd indices.

     Example:
         0,1,2,3,4,5 -> 0,2,4,1,3,5

    The function takes in the index of the mode that will be relabeled and
    the total number modes.

    Args:
        mode_idx (int): the mode index that is being reordered
        num_modes (int): the total number of modes of the operator.

    Returns (int): reordered index of the mode.
    """
    halfway = int(numpy.ceil(num_modes / 2.))

    if mode_idx % 2 == 0:
        return mode_idx // 2

    return mode_idx // 2 + halfway


def normal_ordered_ladder_term(term, coefficient, parity=-1):
    """Return a normal ordered FermionOperator or BosonOperator corresponding
    to single term.

    Args:
        term (list or tuple): A sequence of tuples. The first element of each
            tuple is an integer indicating the mode on which a fermion ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient(complex or float): The coefficient of the term.
        parity (int): parity=-1 corresponds to a Fermionic term that should be
            ordered based on the canonical anti-commutation relations.
            parity=1 corresponds to a Bosonic term that should be ordered based
            on the canonical commutation relations.

    Returns:
        ordered_term: a FermionOperator or BosonOperator instance.
            The normal ordered form of the input.
            Note that this might have more terms.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original term, the
        runtime of this method is exponential in the number of qubits.
    """
    # Iterate from left to right across operators and reorder to normal
    # form. Swap terms operators into correct position by moving from
    # left to right across ladder operators.
    term = list(term)

    if parity == -1:
        Op = FermionOperator
    elif parity == 1:
        Op = BosonOperator

    ordered_term = Op()

    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if raising on right and lowering on left.
            if right_operator[1] and not left_operator[1]:
                term[j - 1] = right_operator
                term[j] = left_operator
                coefficient *= parity

                # Replace a a^\dagger with 1 + parity*a^\dagger a
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1):]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_ladder_term(
                        tuple(new_term), parity*coefficient, parity)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:

                # If same two Fermionic operators are repeated,
                # evaluate to zero.
                if parity == -1 and right_operator[0] == left_operator[0]:
                    return ordered_term

                # Swap if same ladder type but lower index on left.
                elif right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator
                    coefficient *= parity

    # Add processed term and return.
    ordered_term += Op(tuple(term), coefficient)
    return ordered_term


def normal_ordered_quad_term(term, coefficient, hbar=1.):
    """Return a normal ordered QuadOperator corresponding to single term.

    Args:
        term: A tuple of tuples. The first element of each tuple is
            an integer indicating the mode on which a boson ladder
            operator acts, starting from zero. The second element of each
            tuple is an integer, either 1 or 0, indicating whether creation
            or annihilation acts on that mode.
        coefficient: The coefficient of the term.
        hbar (float): the value of hbar used in the definition of the
            commutator [q_i, p_j] = i hbar delta_ij. By default hbar=1.

    Returns:
        ordered_term (QuadOperator): The normal ordered form of the input.
            Note that this might have more terms.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, q operators come first.
    """
    # Iterate from left to right across operators and reorder to normal
    # form. Swap terms operators into correct position by moving from
    # left to right across ladder operators.
    term = list(term)
    ordered_term = QuadOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_operator = term[j]
            left_operator = term[j - 1]

            # Swap operators if q on right and p on left.
            # p q -> q p
            if right_operator[1] == 'q' and not left_operator[1] == 'q':
                term[j - 1] = right_operator
                term[j] = left_operator

                # Replace p q with i hbar + q p
                # if indices are the same.
                if right_operator[0] == left_operator[0]:
                    new_term = term[:(j - 1)] + term[(j + 1)::]

                    # Recursively add the processed new term.
                    ordered_term += normal_ordered_quad_term(
                        tuple(new_term), -coefficient*1j*hbar)

            # Handle case when operator type is the same.
            elif right_operator[1] == left_operator[1]:

                # Swap if same type but lower index on left.
                if right_operator[0] > left_operator[0]:
                    term[j - 1] = right_operator
                    term[j] = left_operator

    # Add processed term and return.
    ordered_term += QuadOperator(tuple(term), coefficient)
    return ordered_term


def normal_ordered(operator, hbar=1.):
    r"""Compute and return the normal ordered form of a FermionOperator,
    BosonOperator, QuadOperator, or InteractionOperator.

    Due to the canonical commutation/anticommutation relations satisfied
    by these operators, there are multiple forms that the same operator
    can take. Here, we define the normal ordered form of each operator,
    providing a distinct representation for distinct operators.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right). In
    addition:

    * FermionOperators: a^\dagger comes before a
    * BosonOperators: b^\dagger comes before b
    * QuadOperators: q operators come before p operators,

    Args:
        operator: an instance of the FermionOperator, BosonOperator,
            QuadOperator, or InteractionOperator classes.
        hbar (float): the value of hbar used in the definition of the
            commutator [q_i, p_j] = i hbar delta_ij. By default hbar=1.
            This argument only applies when normal ordering QuadOperators.
    """
    kwargs = {}

    if isinstance(operator, FermionOperator):
        ordered_operator = FermionOperator()
        order_fn = normal_ordered_ladder_term
        kwargs['parity'] = -1

    elif isinstance(operator, BosonOperator):
        ordered_operator = BosonOperator()
        order_fn = normal_ordered_ladder_term
        kwargs['parity'] = 1

    elif isinstance(operator, QuadOperator):
        ordered_operator = QuadOperator()
        order_fn = normal_ordered_quad_term
        kwargs['hbar'] = hbar

    elif isinstance(operator, InteractionOperator):
        constant = operator.constant
        n_modes = operator.n_qubits
        one_body_tensor = operator.one_body_tensor.copy()
        two_body_tensor = numpy.zeros_like(operator.two_body_tensor)
        quadratic_index_pairs = (
            (pq, pq) for pq in itertools.combinations(range(n_modes)[::-1], 2))
        cubic_index_pairs = (index_pair
            for p, q, r in itertools.combinations(range(n_modes)[::-1], 3)
            for index_pair in [
                ((p, q), (p, r)), ((p, r), (p, q)),
                ((p, q), (q, r)), ((q, r), (p, q)),
                ((p, r), (q, r)), ((q, r), (p, r))])
        quartic_index_pairs = (index_pair
            for p, q, r, s in itertools.combinations(range(n_modes)[::-1], 4)
            for index_pair in [
                ((p, q), (r, s)), ((r, s), (p, q)),
                ((p, r), (q, s)), ((q, s), (p, r)),
                ((p, s), (q, r)), ((q, r), (p, s))])
        index_pairs = itertools.chain(
            quadratic_index_pairs, cubic_index_pairs, quartic_index_pairs)
        for pq, rs in index_pairs:
            two_body_tensor[pq + rs] = sum(
                s * ss * operator.two_body_tensor[pq[::s] + rs[::ss]]
                for s, ss in itertools.product([-1, 1], repeat=2))
        return InteractionOperator(constant, one_body_tensor, two_body_tensor)

    else:
        raise TypeError('Can only normal order FermionOperator, '
                        'BosonOperator, QuadOperator, or InteractionOperator.')

    for term, coefficient in operator.terms.items():
        ordered_operator += order_fn(term, coefficient, **kwargs)

    return ordered_operator


def _find_compatible_basis(term, bases):
    for basis in bases:
        basis_qubits = {op[0] for op in basis}
        conflicts = ((i, P) for (i, P) in term
                     if i in basis_qubits and (i, P) not in basis)
        if any(conflicts):
            continue
        return basis
    return None


def group_into_tensor_product_basis_sets(operator, seed=None):
    """
    Split an operator (instance of QubitOperator) into `sub-operator`
    QubitOperators, where each sub-operator has terms that are diagonal
    in the same tensor product basis.

    Each `sub-operator` can be measured using the same qubit post-rotations
    in expectation estimation. Grouping into these tensor product basis
    sets has been found to improve the efficiency of expectation estimation
    significantly for some Hamiltonians in the context of
    VQE (see section V(A) in the supplementary material of
    https://arxiv.org/pdf/1704.05018v2.pdf). The more general problem
    of grouping operators into commutitative groups is discussed in
    section IV (B2) of https://arxiv.org/pdf/1509.04279v1.pdf. The
    original input operator is the union of all output sub-operators,
    and all sub-operators are disjoint (do not share any terms).

    Args:
        operator (QubitOperator): the operator that will be split into
            sub-operators (tensor product basis sets).
        seed (int): default None. Random seed used to initialize the
            numpy.RandomState pseudo-random number generator.

    Returns:
        sub_operators (dict): a dictionary where each key defines a
            tensor product basis, and each corresponding value is a
            QubitOperator with terms that are all diagonal in
            that basis.
            **key** (tuple of tuples): Each key is a term, which defines
                a tensor product basis. A term is a product of individual
                factors; each factor is represented by a tuple of the form
                (`index`, `action`), and these tuples are collected into a
                larger tuple which represents the term as the product of
                its factors. `action` is from the set {'X', 'Y', 'Z'} and
                `index` is a non-negative integer corresponding to the
                index of a qubit.
            **value** (QubitOperator): A QubitOperator with terms that are
                diagonal in the basis defined by the key it is stored in.

    Raises:
       TypeError: Operator of invalid type.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError('Can only split QubitOperator into tensor product'
                        ' basis sets. {} is not supported.'.format(
                            type(operator).__name__))

    sub_operators = {}
    r = RandomState(seed)
    for term, coefficient in operator.terms.items():
        bases = list(sub_operators.keys())
        r.shuffle(bases)
        basis = _find_compatible_basis(term, bases)
        if basis is None:
            sub_operators[term] = QubitOperator(term, coefficient)
        else:
            sub_operator = sub_operators.pop(basis)
            sub_operator += QubitOperator(term, coefficient)
            additions = tuple(op for op in term if op not in basis)
            basis = tuple(
                sorted(basis + additions, key=lambda factor: factor[0]))
            sub_operators[basis] = sub_operator

    return sub_operators


def _commutes(operator1,operator2):
    return operator1*operator2==operator2*operator1

def _non_fully_commuting_terms(hamiltonian):
    terms = list([QubitOperator(key) for key in hamiltonian.terms.keys()])
    T = []  # will contain the subset of terms that do not
    # commute universally in terms
    for i in range(len(terms)):
        if any(not _commutes(terms[i],terms[j]) for j in range(len(terms))):
            T.append(terms[i])
    return T

def is_contextual(hamiltonian):
    """
    Determine whether a hamiltonian (instance of QubitOperator) is contextual,
    in the sense of https://arxiv.org/abs/1904.02260.

    Args:
        hamiltonian (QubitOperator): the hamiltonian whose
            contextuality is to be evaluated.

    Returns:
        boolean indicating whether hamiltonian is contextual or not.
    """
    T = _non_fully_commuting_terms(hamiltonian)
    # Search in T for triples in which exactly one pair anticommutes;
    # if any exist, hamiltonian is contextual.
    for i in range(len(T)):  # WLOG, i indexes the operator that (putatively)
        # commutes with both others.
        for j in range(len(T)):
            for k in range(j + 1, len(T)):  # Ordering of j, k does not matter.
                if i!=j and i!=k and _commutes(T[i],T[j]) \
                    and _commutes(T[i],T[k]) and \
                        not _commutes(T[j],T[k]):
                    return True
    return False
