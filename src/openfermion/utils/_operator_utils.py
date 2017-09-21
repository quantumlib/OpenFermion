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
from openfermion.ops import *
from future.builtins.iterators import map, zip


class OperatorUtilsError(Exception):
    pass


def eigenspectrum(operator):
    """Compute the eigenspectrum of an operator.

    WARNING: This function has cubic runtime in dimension of
        Hilbert space operator, which might be exponential.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            InteractionTensor, or InteractionRDM.

    Returns:
        eigenspectrum: dense numpy array of floats giving eigenspectrum.
    """
    from openfermion.transforms import get_sparse_operator
    from openfermion.utils import sparse_eigenspectrum
    sparse_operator = get_sparse_operator(operator)
    eigenspectrum = sparse_eigenspectrum(sparse_operator)
    return eigenspectrum


def count_qubits(operator):
    """Compute the minimum number of qubits on which operator acts.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            InteractionTensor, or InteractionRDM.

    Returns:
        n_qubits (int): The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.
    """
    # Handle FermionOperator.
    if isinstance(operator, FermionOperator):
        n_qubits = 0
        for term in operator.terms:
            for ladder_operator in term:
                if ladder_operator[0] + 1 > n_qubits:
                    n_qubits = ladder_operator[0] + 1
        return n_qubits

    # Handle QubitOperator.
    elif isinstance(operator, QubitOperator):
        n_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > n_qubits:
                    n_qubits = term[-1][0] + 1
        return n_qubits

    # Handle InteractionOperator, InteractionRDM, InteractionTensor.
    elif isinstance(operator, (InteractionOperator,
                               InteractionRDM,
                               InteractionTensor)):
        return operator.n_qubits

    # Raise for other classes.
    else:
        raise TypeError('Operator of invalid type.')


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


def commutator(operator_a, operator_b):
    """Compute the commutator of two QubitOperators or FermionOperators.

    Args:
        operator_a, operator_b (FermionOperator): Operators in commutator.
    """
    if not ((isinstance(operator_a, FermionOperator) and
            isinstance(operator_b, FermionOperator)) or
            (isinstance(operator_a, QubitOperator) and
             isinstance(operator_b, QubitOperator))):
        raise TypeError('operator_a and operator_b must both be Fermion or'
                        ' QubitOperators.')
    result = operator_a * operator_b
    result -= operator_b * operator_a
    return result


def save_operator(operator, file_name=None, data_directory=None):
    """Save FermionOperator or QubitOperator to file.

    Args:
        operator: An instance of FermionOperator or QubitOperator.
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
                        directory specified in config file.

    Raises:
        OperatorUtilsError: Not saved, file already exists.
        TypeError: Operator of invalid type.
    """
    file_path = get_file_path(file_name, data_directory)

    if os.path.isfile(file_path):
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
