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
from __future__ import absolute_import

import marshal
import numpy
import os
import time

from openfermion.config import *
from openfermion.hamiltonians._jellium import (grid_indices,
                                               momentum_vector,
                                               orbital_id,
                                               position_vector)
from openfermion.ops import *
from future.builtins.iterators import map, zip


class OperatorUtilsError(Exception):
    pass


def count_qubits(operator):
    """Compute the minimum number of qubits on which operator acts.

    Args:
        operator: FermionOperator, QubitOperator, or PolynomialTensor.

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

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            PolynomialTensor, or InteractionRDM.

    Returns:
        eigenspectrum: dense numpy array of floats giving eigenspectrum.
    """
    from openfermion.transforms import get_sparse_operator
    from openfermion.utils import sparse_eigenspectrum
    sparse_operator = get_sparse_operator(operator, n_qubits)
    eigenspectrum = sparse_eigenspectrum(sparse_operator)
    return eigenspectrum


def is_identity(operator):
    """Check whether QubitOperator of FermionOperator is identity.

    Args:
        operator: QubitOperator or FermionOperator.

    Raises:
        TypeError: Operator of invalid type.
    """
    if isinstance(operator, (QubitOperator, FermionOperator)):
        return list(operator.terms) == [()]
    raise TypeError('Operator of invalid type.')


def _fourier_transform_helper(hamiltonian,
                              grid,
                              spinless,
                              phase_factor,
                              vec_func_1,
                              vec_func_2):
    hamiltonian_t = FermionOperator.zero()
    normalize_factor = numpy.sqrt(1.0 / float(grid.num_points()))

    for term in hamiltonian.terms:
        transformed_term = FermionOperator.identity()
        for ladder_op_mode, ladder_op_type in term:
            indices_1 = grid_indices(ladder_op_mode, grid, spinless)
            vec1 = vec_func_1(indices_1, grid)
            new_basis = FermionOperator.zero()
            for indices_2 in grid.all_points_indices():
                vec2 = vec_func_2(indices_2, grid)
                spin = None if spinless else ladder_op_mode % 2
                orbital = orbital_id(grid, indices_2, spin)
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
    """Apply Fourier transform to change hamiltonian in plane wave basis.

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
                                     vec_func_1=momentum_vector,
                                     vec_func_2=position_vector)


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
    """Apply inverse Fourier transform to change hamiltonian in plane wave dual
    basis.

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
                                     vec_func_1=position_vector,
                                     vec_func_2=momentum_vector)


def load_operator(file_name=None, data_directory=None):
    """Load FermionOperator or QubitOperator from file.

    Args:
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
                        directory specified in config file.

    Returns:
        operator: The stored FermionOperator or QubitOperator

    Raises:
        TypeError: Operator of invalid type.
    """
    file_path = get_file_path(file_name, data_directory)

    with open(file_path, 'rb') as f:
        data = marshal.load(f)
        operator_type = data[0]
        operator_terms = data[1]

    if operator_type == 'FermionOperator':
        operator = FermionOperator()
        for term in operator_terms:
            operator += FermionOperator(term, operator_terms[term])
    elif operator_type == 'QubitOperator':
        operator = QubitOperator()
        for term in operator_terms:
            operator += QubitOperator(term, operator_terms[term])
    else:
        raise TypeError('Operator of invalid type.')

    return operator


def save_operator(operator, file_name=None, data_directory=None,
                  allow_overwrite=False):
    """Save FermionOperator or QubitOperator to file.

    Args:
        operator: An instance of FermionOperator or QubitOperator.
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
                        directory specified in config file.
        allow_overwrite: Whether to allow files to be overwritten.

    Raises:
        OperatorUtilsError: Not saved, file already exists.
        TypeError: Operator of invalid type.
    """
    file_path = get_file_path(file_name, data_directory)

    if os.path.isfile(file_path) and not allow_overwrite:
        raise OperatorUtilsError("Not saved, file already exists.")

    if isinstance(operator, FermionOperator):
        operator_type = "FermionOperator"
    elif isinstance(operator, QubitOperator):
        operator_type = "QubitOperator"
    elif (isinstance(operator, InteractionOperator) or
          isinstance(operator, InteractionRDM)):
        raise NotImplementedError('Not yet implemented for InteractionOperator'
                                  ' or InteractionRDM.')
    else:
        raise TypeError('Operator of invalid type.')

    tm = operator.terms
    with open(file_path, 'wb') as f:
        marshal.dump((operator_type, dict(zip(tm.keys(),
                                          map(complex, tm.values())))), f)
