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
from typing import Union
import itertools
import numpy
import sympy

from openfermion.ops.operators import (QuadOperator, BosonOperator,
                                       FermionOperator, MajoranaOperator)
from openfermion.ops.representations import (PolynomialTensor,
                                             DiagonalCoulombHamiltonian)

from openfermion.utils.operator_utils import count_qubits


def get_quad_operator(operator, hbar=1.):
    """Convert to QuadOperator.

    Args:
        operator: BosonOperator.
        hbar (float): the value of hbar used in the definition
            of the commutator [q_i, p_j] = i hbar delta_ij.
            By default hbar=1.

    Returns:
        quad_operator: An instance of the QuadOperator class.
    """
    quad_operator = QuadOperator()

    if isinstance(operator, BosonOperator):
        for term, coefficient in operator.terms.items():
            tmp = QuadOperator('', coefficient)
            for i, d in term:
                tmp *= (1./numpy.sqrt(2.*hbar)) \
                    * (QuadOperator(((i, 'q')))
                        + QuadOperator(((i, 'p')), 1j*(-1)**d))
            quad_operator += tmp

    else:
        raise TypeError("Only BosonOperator is currently "
                        "supported for get_quad_operator.")

    return quad_operator


def check_no_sympy(operator):
    """Checks whether a SymbolicOperator contains any
    sympy expressions, which will prevent it being converted
    to a PolynomialTensor or DiagonalCoulombHamiltonian

    Args:
        operator(SymbolicOperator): the operator to be tested
    """
    for key in operator.terms:
        if isinstance(operator.terms[key], sympy.Expr):
            raise TypeError('This conversion is currently not supported ' +
                            'for operators with sympy expressions ' +
                            'as coefficients')


def get_boson_operator(operator, hbar=1.):
    """Convert to BosonOperator.

    Args:
        operator: QuadOperator.
        hbar (float): the value of hbar used in the definition
            of the commutator [q_i, p_j] = i hbar delta_ij.
            By default hbar=1.

    Returns:
        boson_operator: An instance of the BosonOperator class.
    """
    boson_operator = BosonOperator()

    if isinstance(operator, QuadOperator):
        for term, coefficient in operator.terms.items():
            tmp = BosonOperator('', coefficient)
            for i, d in term:
                if d == 'q':
                    coeff = numpy.sqrt(hbar / 2)
                    sign = 1
                elif d == 'p':
                    coeff = -1j * numpy.sqrt(hbar / 2)
                    sign = -1

                tmp *= coeff * (BosonOperator(((i, 0))) + BosonOperator(
                    ((i, 1)), sign))
            boson_operator += tmp

    else:
        raise TypeError("Only QuadOperator is currently "
                        "supported for get_boson_operator.")

    return boson_operator


def get_fermion_operator(operator):
    """Convert to FermionOperator.

    Returns:
        fermion_operator: An instance of the FermionOperator class.
    """
    if isinstance(operator, PolynomialTensor):
        return _polynomial_tensor_to_fermion_operator(operator)
    elif isinstance(operator, DiagonalCoulombHamiltonian):
        return _diagonal_coulomb_hamiltonian_to_fermion_operator(operator)
    elif isinstance(operator, MajoranaOperator):
        return _majorana_operator_to_fermion_operator(operator)
    else:
        raise TypeError('{} cannot be converted to FermionOperator'.format(
            type(operator)))


def _polynomial_tensor_to_fermion_operator(operator):
    fermion_operator = FermionOperator()
    for term in operator:
        fermion_operator += FermionOperator(term, operator[term])
    return fermion_operator


def _diagonal_coulomb_hamiltonian_to_fermion_operator(operator):
    fermion_operator = FermionOperator()
    n_qubits = count_qubits(operator)
    fermion_operator += FermionOperator((), operator.constant)
    for p, q in itertools.product(range(n_qubits), repeat=2):
        fermion_operator += FermionOperator(((p, 1), (q, 0)),
                                            operator.one_body[p, q])
        fermion_operator += FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)),
                                            operator.two_body[p, q])
    return fermion_operator


def _majorana_operator_to_fermion_operator(majorana_operator):
    fermion_operator = FermionOperator()
    for term, coeff in majorana_operator.terms.items():
        converted_term = _majorana_term_to_fermion_operator(term)
        converted_term *= coeff
        fermion_operator += converted_term
    return fermion_operator


def _majorana_term_to_fermion_operator(term):
    converted_term = FermionOperator(())
    for index in term:
        j, b = divmod(index, 2)
        if b:
            converted_op = FermionOperator((j, 0), -1j)
            converted_op += FermionOperator((j, 1), 1j)
        else:
            converted_op = FermionOperator((j, 0))
            converted_op += FermionOperator((j, 1))
        converted_term *= converted_op
    return converted_term


def get_majorana_operator(
        operator: Union[PolynomialTensor, DiagonalCoulombHamiltonian,
                        FermionOperator]) -> MajoranaOperator:
    """
    Convert to MajoranaOperator.

    Uses the convention of even + odd indexing of Majorana modes derived from
    a fermionic mode:
        fermion annhil.  c_k  -> ( gamma_{2k} + 1.j * gamma_{2k+1} ) / 2
        fermion creation c^_k -> ( gamma_{2k} - 1.j * gamma_{2k+1} ) / 2

    Args:
        operator (PolynomialTensor,
            DiagonalCoulombHamiltonian or
            FermionOperator): Operator to write as Majorana Operator.

    Returns:
        majorana_operator: An instance of the MajoranaOperator class.

    Raises:
        TypeError: If operator is not of PolynomialTensor,
            DiagonalCoulombHamiltonian or FermionOperator.
    """
    if isinstance(operator, FermionOperator):
        return _fermion_operator_to_majorana_operator(operator)
    elif isinstance(operator, (PolynomialTensor, DiagonalCoulombHamiltonian)):
        return _fermion_operator_to_majorana_operator(
            get_fermion_operator(operator))
    raise TypeError('{} cannot be converted to MajoranaOperator'.format(
        type(operator)))


def _fermion_operator_to_majorana_operator(fermion_operator: FermionOperator
                                          ) -> MajoranaOperator:
    """
    Convert FermionOperator to MajoranaOperator.

    Auxiliar function of get_majorana_operator.

    Args:
        fermion_operator (FermionOperator): To convert to MajoranaOperator.

    Returns:
        majorana_operator object.

    Raises:
        TypeError: if input is not a FermionOperator.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError('Input a FermionOperator.')

    majorana_operator = MajoranaOperator()
    for term, coeff in fermion_operator.terms.items():
        converted_term = _fermion_term_to_majorana_operator(term)
        converted_term *= coeff
        majorana_operator += converted_term

    return majorana_operator


def _fermion_term_to_majorana_operator(term: tuple) -> MajoranaOperator:
    """
    Convert single terms of FermionOperator to Majorana.
    (Auxiliary function of get_majorana_operator.)

    Convention: even + odd indexing of Majorana modes derived from a
    fermionic mode:
        fermion annhil.  c_k  -> ( gamma_{2k} + 1.j * gamma_{2k+1} ) / 2
        fermion creation c^_k -> ( gamma_{2k} - 1.j * gamma_{2k+1} ) / 2

    Args:
        term (tuple): single FermionOperator term.

    Returns:
        converted_term: single MajoranaOperator term.

    Raises:
        TypeError: if term is a tuple.
    """
    if not isinstance(term, tuple):
        raise TypeError('Term does not have the correct Type.')

    converted_term = MajoranaOperator(())
    for index, action in term:
        converted_op = MajoranaOperator((2 * index,), 0.5)

        if action:
            converted_op += MajoranaOperator((2 * index + 1,), -0.5j)

        else:
            converted_op += MajoranaOperator((2 * index + 1,), 0.5j)

        converted_term *= converted_op

    return converted_term
