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
import marshal
import os

import numpy
import sympy
from scipy.sparse import spmatrix

from openfermion.config import DATA_DIRECTORY, EQ_TOLERANCE
from openfermion.ops.operators import (
    BosonOperator,
    FermionOperator,
    MajoranaOperator,
    QuadOperator,
    QubitOperator,
    IsingOperator,
)
from openfermion.ops.representations import (
    PolynomialTensor,
    DiagonalCoulombHamiltonian,
    InteractionOperator,
    InteractionRDM,
)
from openfermion.transforms.opconversions.term_reordering import normal_ordered


# Maximum size allowed for data files read by load_operator(). This is a (weak) safety
# measure against corrupted or insecure files.
_MAX_TEXT_OPERATOR_DATA = 5 * 1024 * 1024
_MAX_BINARY_OPERATOR_DATA = 1024 * 1024


class OperatorUtilsError(Exception):
    pass


class OperatorSpecificationError(Exception):
    pass


def hermitian_conjugated(operator):
    """Return Hermitian conjugate of operator."""
    # Handle FermionOperator
    if isinstance(operator, FermionOperator):
        conjugate_operator = FermionOperator()
        for term, coefficient in operator.terms.items():
            conjugate_term = tuple(
                [(tensor_factor, 1 - action) for (tensor_factor, action) in reversed(term)]
            )
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Handle BosonOperator
    elif isinstance(operator, BosonOperator):
        conjugate_operator = BosonOperator()
        for term, coefficient in operator.terms.items():
            conjugate_term = tuple(
                [(tensor_factor, 1 - action) for (tensor_factor, action) in reversed(term)]
            )
            # take into account that different indices commute
            conjugate_term = tuple(sorted(conjugate_term, key=lambda factor: factor[0]))
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
            conjugate_term = tuple(sorted(conjugate_term, key=lambda factor: factor[0]))
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Handle InteractionOperator
    elif isinstance(operator, InteractionOperator):
        conjugate_constant = operator.constant.conjugate()
        conjugate_one_body_tensor = hermitian_conjugated(operator.one_body_tensor)
        conjugate_two_body_tensor = hermitian_conjugated(operator.two_body_tensor)
        conjugate_operator = type(operator)(
            conjugate_constant, conjugate_one_body_tensor, conjugate_two_body_tensor
        )

    # Handle sparse matrix
    elif isinstance(operator, spmatrix):
        conjugate_operator = operator.getH()

    # Handle numpy array
    elif isinstance(operator, numpy.ndarray):
        conjugate_operator = operator.T.conj()

    # Unsupported type
    else:
        raise TypeError(
            'Taking the hermitian conjugate of a {} is not '
            'supported.'.format(type(operator).__name__)
        )

    return conjugate_operator


def is_hermitian(operator):
    """Test if operator is Hermitian."""
    # Handle FermionOperator, BosonOperator, and InteractionOperator
    if isinstance(operator, (FermionOperator, BosonOperator, InteractionOperator)):
        return normal_ordered(operator) == normal_ordered(hermitian_conjugated(operator))

    # Handle QubitOperator and QuadOperator
    if isinstance(operator, (QubitOperator, QuadOperator)):
        return operator == hermitian_conjugated(operator)

    # Handle sparse matrix
    elif isinstance(operator, spmatrix):
        difference = operator - hermitian_conjugated(operator)
        discrepancy = 0.0
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
        raise TypeError(
            'Checking whether a {} is hermitian is not '
            'supported.'.format(type(operator).__name__)
        )


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
                if numpy.ceil((majorana_index + 1) / 2) > num_qubits:
                    num_qubits = int(numpy.ceil((majorana_index + 1) / 2))
        return num_qubits

    # Handle DiagonalCoulombHamiltonian
    elif isinstance(operator, DiagonalCoulombHamiltonian):
        return operator.one_body.shape[0]

    # Handle PolynomialTensor
    elif isinstance(operator, PolynomialTensor):
        return operator.n_qubits

    # Handle IsingOperator
    elif isinstance(operator, IsingOperator):
        num_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > num_qubits:
                    num_qubits = term[-1][0] + 1
        return num_qubits

    # Raise for other classes.
    else:
        raise TypeError('Operator of invalid type.')


def is_identity(operator):
    """Check whether QubitOperator of FermionOperator is identity.

    Args:
        operator: QubitOperator, FermionOperator,
            BosonOperator, or QuadOperator.

    Raises:
        TypeError: Operator of invalid type.
    """
    if isinstance(operator, (QubitOperator, FermionOperator, BosonOperator, QuadOperator)):
        return list(operator.terms) == [()]
    raise TypeError('Operator of invalid type.')


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


def load_operator(file_name=None, data_directory=None, plain_text=False):
    """Load an operator (such as a FermionOperator) from a file.

    Args:
        file_name: The name of the data file to read.
        data_directory: Optional data directory to change from default data
            directory specified in config file.
        plain_text: Whether the input file is plain text

    Returns:
        operator: The stored FermionOperator, BosonOperator,
            QuadOperator, or QubitOperator.

    Raises:
        TypeError: Operator of invalid type.
        ValueError: If the file is larger than the maximum allowed.
        ValueError: If the file content is not as expected or loading fails.
        IOError: If the file cannot be opened.

    Warning:
        Loading from binary files (plain_text=False) uses the Python 'marshal'
        module, which is not secure against untrusted or maliciously crafted
        data. Only load binary operator files from sources that you trust.
        Prefer using the plain_text format for data from untrusted sources.
    """

    file_path = get_file_path(file_name, data_directory)

    operator_type = None
    operator_terms = None

    if plain_text:
        with open(file_path, 'r') as f:
            data = f.read(_MAX_TEXT_OPERATOR_DATA)
        try:
            operator_type, operator_terms = data.split(":\n", 1)
        except ValueError:
            raise ValueError(
                "Invalid format in plain-text data file {file_path}: " "expected 'TYPE:\\nTERMS'"
            )

        if operator_type == 'FermionOperator':
            operator = FermionOperator(operator_terms)
        elif operator_type == 'BosonOperator':
            operator = BosonOperator(operator_terms)
        elif operator_type == 'QubitOperator':
            operator = QubitOperator(operator_terms)
        elif operator_type == 'QuadOperator':
            operator = QuadOperator(operator_terms)
        else:
            raise TypeError(
                f"Invalid operator type '{operator_type}' encountered "
                f"found in plain-text data file '{file_path}'."
            )
    else:
        # marshal.load() doesn't have a size parameter, so we test it ourselves.
        if os.path.getsize(file_path) > _MAX_BINARY_OPERATOR_DATA:
            raise ValueError(
                f"Size of {file_path} exceeds maximum allowed "
                f"({_MAX_BINARY_OPERATOR_DATA} bytes)."
            )
        try:
            with open(file_path, 'rb') as f:
                raw_data = marshal.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load marshaled data from {file_path}: {e}")

        operator_type, operator_terms = _validate_operator_data(raw_data)

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


def save_operator(
    operator, file_name=None, data_directory=None, allow_overwrite=False, plain_text=False
):
    """Save an operator (such as a FermionOperator) to a file.

    Args:
        operator: An instance of FermionOperator, BosonOperator,
            or QubitOperator.
        file_name: The name of the saved file.
        data_directory: Optional data directory to change from default data
            directory specified in config file.
        allow_overwrite: Whether to allow files to be overwritten.
        plain_text: Whether the operator should be saved to a
            plain-text format for manual analysis.

    Raises:
        OperatorUtilsError: Not saved, file already exists.
        TypeError: Operator of invalid type.
        TypeError: Coefficients in Operator sympy expressions.
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
        raise NotImplementedError(
            'Not yet implemented for ' 'InteractionOperator or InteractionRDM.'
        )
    else:
        raise TypeError('Operator of invalid type.')

    for term in operator.terms:
        if isinstance(operator.terms[term], sympy.Expr):
            raise TypeError('Cannot save sympy expressions.')

    if plain_text:
        with open(file_path, 'w') as f:
            f.write(operator_type + ":\n" + str(operator))
    else:
        tm = operator.terms
        with open(file_path, 'wb') as f:
            marshal.dump((operator_type, dict(zip(tm.keys(), map(complex, tm.values())))), f)


def _validate_operator_data(raw_data):
    """Validates the structure and types of data loaded using marshal.

    The file is expected to contain a tuple of (type, data), where the
    "type" is one of the currently-supported operators, and "data" is a dict.

    Args:
        raw_data: text or binary data.

    Returns:
        tuple(str, dict) where the 0th element is the name of the operator
            type (e.g., 'FermionOperator') and the dict is the operator data.

    Raises:
        TypeError: raw_data did not contain a tuple of length 2.
        TypeError: the first element of the tuple is not a string.
        TypeError: the second element of the tuple is not a dict.
        TypeError: the given operator type is not supported.
    """

    if not isinstance(raw_data, tuple) or len(raw_data) != 2:
        raise TypeError(
            f"Invalid marshaled structure: Expected a tuple "
            "of length 2, but got {type(raw_data)} instead."
        )

    operator_type, operator_terms = raw_data

    if not isinstance(operator_type, str):
        raise TypeError(
            f"Invalid type for operator_type: Expected str but "
            "got type {type(operator_type)} instead."
        )

    allowed = {'FermionOperator', 'BosonOperator', 'QubitOperator', 'QuadOperator'}
    if operator_type not in allowed:
        raise TypeError(
            f"Operator type '{operator_type}' is not supported. "
            "The operator must be one of {allowed}."
        )

    if not isinstance(operator_terms, dict):
        raise TypeError(
            f"Invalid type for operator_terms: Expected dict "
            "but got type {type(operator_terms)} instead."
        )

    return operator_type, operator_terms
