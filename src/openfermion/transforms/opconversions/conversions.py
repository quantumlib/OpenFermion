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

from openfermion.config import EQ_TOLERANCE
from openfermion.ops.operators import (QuadOperator, BosonOperator,
                                       FermionOperator, MajoranaOperator)
from openfermion.ops.representations import (PolynomialTensor,
                                             DiagonalCoulombHamiltonian,
                                             InteractionOperator,
                                             InteractionOperatorError)

# for breaking cyclic imports
import openfermion.utils.operator_utils as op_utils
import openfermion.ops.representations.quadratic_hamiltonian as quad
import openfermion.chem as chem


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


def get_interaction_operator(fermion_operator, n_qubits=None):
    r"""Convert a 2-body fermionic operator to InteractionOperator.

    This function should only be called on fermionic operators which
    consist of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s
    terms. The one-body terms are stored in a matrix, one_body[p, q], and
    the two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
       interaction_operator: An instance of the InteractionOperator class.

    Raises:
        TypeError: Input must be a FermionOperator.
        TypeError: FermionOperator does not map to InteractionOperator.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original operator, the
        runtime of this method is exponential in the number of qubits.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError('Input must be a FermionOperator.')

    check_no_sympy(fermion_operator)

    if n_qubits is None:
        n_qubits = op_utils.count_qubits(fermion_operator)
    if n_qubits < op_utils.count_qubits(fermion_operator):
        raise ValueError('Invalid number of qubits specified.')

    # Normal order the terms and initialize.
    fermion_operator = op_utils.normal_ordered(fermion_operator)
    constant = 0.
    one_body = numpy.zeros((n_qubits, n_qubits), complex)
    two_body = numpy.zeros((n_qubits, n_qubits, n_qubits, n_qubits), complex)

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        # Ignore this term if the coefficient is zero
        if abs(coefficient) < EQ_TOLERANCE:
            # not testable because normal_ordered kills
            # fermion terms lower than EQ_TOLERANCE
            continue  # pragma: no cover

        # Handle constant shift.
        if len(term) == 0:
            constant = coefficient

        elif len(term) == 2:
            # Handle one-body terms.
            if [operator[1] for operator in term] == [1, 0]:
                p, q = [operator[0] for operator in term]
                one_body[p, q] = coefficient
            else:
                raise InteractionOperatorError('FermionOperator does not map '
                                               'to InteractionOperator.')

        elif len(term) == 4:
            # Handle two-body terms.
            if [operator[1] for operator in term] == [1, 1, 0, 0]:
                p, q, r, s = [operator[0] for operator in term]
                two_body[p, q, r, s] = coefficient
            else:
                raise InteractionOperatorError('FermionOperator does not map '
                                               'to InteractionOperator.')

        else:
            # Handle non-molecular Hamiltonian.
            raise InteractionOperatorError('FermionOperator does not map '
                                           'to InteractionOperator.')

    # Form InteractionOperator and return.
    interaction_operator = InteractionOperator(constant, one_body, two_body)
    return interaction_operator


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
    n_qubits = op_utils.count_qubits(operator)
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


def get_quadratic_hamiltonian(fermion_operator,
                              chemical_potential=0.,
                              n_qubits=None,
                              ignore_incompatible_terms=False):
    r"""Convert a quadratic fermionic operator to QuadraticHamiltonian.

    Args:
        fermion_operator(FermionOperator): The operator to convert.
        chemical_potential(float): A chemical potential to include in
            the returned operator
        n_qubits(int): Optionally specify the total number of qubits in the
            system
        ignore_incompatible_terms(bool): This flag determines the behavior
            of this method when it encounters terms that are not quadratic
            that is, terms that are not of the form a^\dagger_p a_q.
            If set to True, this method will simply ignore those terms.
            If False, then this method will raise an error if it encounters
            such a term. The default setting is False.

    Returns:
       quadratic_hamiltonian: An instance of the QuadraticHamiltonian class.

    Raises:
        TypeError: Input must be a FermionOperator.
        TypeError: FermionOperator does not map to QuadraticHamiltonian.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original operator, the
        runtime of this method is exponential in the number of qubits.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError('Input must be a FermionOperator.')

    check_no_sympy(fermion_operator)

    if n_qubits is None:
        n_qubits = op_utils.count_qubits(fermion_operator)
    if n_qubits < op_utils.count_qubits(fermion_operator):
        raise ValueError('Invalid number of qubits specified.')

    # Normal order the terms and initialize.
    fermion_operator = op_utils.normal_ordered(fermion_operator)
    constant = 0.
    combined_hermitian_part = numpy.zeros((n_qubits, n_qubits), complex)
    antisymmetric_part = numpy.zeros((n_qubits, n_qubits), complex)

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        # Ignore this term if the coefficient is zero
        if abs(coefficient) < EQ_TOLERANCE:
            # not testable because normal_ordered kills
            # fermion terms lower than EQ_TOLERANCE
            continue  # pragma: no cover

        if len(term) == 0:
            # Constant term
            constant = coefficient
        elif len(term) == 2:
            ladder_type = [operator[1] for operator in term]
            p, q = [operator[0] for operator in term]

            if ladder_type == [1, 0]:
                combined_hermitian_part[p, q] = coefficient
            elif ladder_type == [1, 1]:
                # Need to check that the corresponding [0, 0] term is present
                conjugate_term = ((p, 0), (q, 0))
                if conjugate_term not in fermion_operator.terms:
                    raise quad.QuadraticHamiltonianError(
                        'FermionOperator does not map '
                        'to QuadraticHamiltonian (not Hermitian).')
                else:
                    matching_coefficient = -fermion_operator.terms[
                        conjugate_term].conjugate()
                    discrepancy = abs(coefficient - matching_coefficient)
                    if discrepancy > EQ_TOLERANCE:
                        raise quad.QuadraticHamiltonianError(
                            'FermionOperator does not map '
                            'to QuadraticHamiltonian (not Hermitian).')
                antisymmetric_part[p, q] += .5 * coefficient
                antisymmetric_part[q, p] -= .5 * coefficient
            else:
                # ladder_type == [0, 0]
                # Need to check that the corresponding [1, 1] term is present
                conjugate_term = ((p, 1), (q, 1))
                if conjugate_term not in fermion_operator.terms:
                    raise quad.QuadraticHamiltonianError(
                        'FermionOperator does not map '
                        'to QuadraticHamiltonian (not Hermitian).')
                else:
                    matching_coefficient = -fermion_operator.terms[
                        conjugate_term].conjugate()
                    discrepancy = abs(coefficient - matching_coefficient)
                    if discrepancy > EQ_TOLERANCE:
                        raise quad.QuadraticHamiltonianError(
                            'FermionOperator does not map '
                            'to QuadraticHamiltonian (not Hermitian).')
                antisymmetric_part[p, q] -= .5 * coefficient.conjugate()
                antisymmetric_part[q, p] += .5 * coefficient.conjugate()
        elif not ignore_incompatible_terms:
            # Operator contains non-quadratic terms
            raise quad.QuadraticHamiltonianError(
                'FermionOperator does not map '
                'to QuadraticHamiltonian '
                '(contains non-quadratic terms).')

    # Compute Hermitian part
    hermitian_part = (combined_hermitian_part +
                      chemical_potential * numpy.eye(n_qubits))

    # Check that the operator is Hermitian
    if not op_utils.is_hermitian(hermitian_part):
        raise quad.QuadraticHamiltonianError(
            'FermionOperator does not map '
            'to QuadraticHamiltonian (not Hermitian).')

    # Form QuadraticHamiltonian and return.
    discrepancy = numpy.max(numpy.abs(antisymmetric_part))
    if discrepancy < EQ_TOLERANCE:
        # Hamiltonian conserves particle number
        quadratic_hamiltonian = quad.QuadraticHamiltonian(
            hermitian_part,
            constant=constant,
            chemical_potential=chemical_potential)
    else:
        # Hamiltonian does not conserve particle number
        quadratic_hamiltonian = quad.QuadraticHamiltonian(
            hermitian_part, antisymmetric_part, constant, chemical_potential)

    return quadratic_hamiltonian


def get_diagonal_coulomb_hamiltonian(fermion_operator,
                                     n_qubits=None,
                                     ignore_incompatible_terms=False):
    r"""Convert a FermionOperator to a DiagonalCoulombHamiltonian.

    Args:
        fermion_operator(FermionOperator): The operator to convert.
        n_qubits(int): Optionally specify the total number of qubits in the
            system
        ignore_incompatible_terms(bool): This flag determines the behavior
            of this method when it encounters terms that are not represented
            by the DiagonalCoulombHamiltonian class, namely, terms that are
            not quadratic and not quartic of the form
            a^\dagger_p a_p a^\dagger_q a_q. If set to True, this method will
            simply ignore those terms. If False, then this method will raise
            an error if it encounters such a term. The default setting is False.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError('Input must be a FermionOperator.')

    check_no_sympy(fermion_operator)

    if n_qubits is None:
        n_qubits = op_utils.count_qubits(fermion_operator)
    if n_qubits < op_utils.count_qubits(fermion_operator):
        raise ValueError('Invalid number of qubits specified.')

    fermion_operator = op_utils.normal_ordered(fermion_operator)
    constant = 0.
    one_body = numpy.zeros((n_qubits, n_qubits), complex)
    two_body = numpy.zeros((n_qubits, n_qubits), float)

    for term, coefficient in fermion_operator.terms.items():
        # Ignore this term if the coefficient is zero
        if abs(coefficient) < EQ_TOLERANCE:
            # not testable because normal_ordered kills
            # fermion terms lower than EQ_TOLERANCE
            continue  # pragma: no cover

        if len(term) == 0:
            constant = coefficient
        else:
            actions = [operator[1] for operator in term]
            if actions == [1, 0]:
                p, q = [operator[0] for operator in term]
                one_body[p, q] = coefficient
            elif actions == [1, 1, 0, 0]:
                p, q, r, s = [operator[0] for operator in term]
                if p == r and q == s:
                    if abs(numpy.imag(coefficient)) > EQ_TOLERANCE:
                        raise ValueError(
                            'FermionOperator does not map to '
                            'DiagonalCoulombHamiltonian (not Hermitian).')
                    coefficient = numpy.real(coefficient)
                    two_body[p, q] = -.5 * coefficient
                    two_body[q, p] = -.5 * coefficient
                elif not ignore_incompatible_terms:
                    raise ValueError('FermionOperator does not map to '
                                     'DiagonalCoulombHamiltonian '
                                     '(contains terms with indices '
                                     '{}).'.format((p, q, r, s)))
            elif not ignore_incompatible_terms:
                raise ValueError('FermionOperator does not map to '
                                 'DiagonalCoulombHamiltonian (contains terms '
                                 'with action {}.'.format(tuple(actions)))

    # Check that the operator is Hermitian
    if not op_utils.is_hermitian(one_body):
        raise ValueError(
            'FermionOperator does not map to DiagonalCoulombHamiltonian '
            '(not Hermitian).')

    return DiagonalCoulombHamiltonian(one_body, two_body, constant)


def get_molecular_data(interaction_operator,
                       geometry=None,
                       basis=None,
                       multiplicity=None,
                       n_electrons=None,
                       reduce_spin=True,
                       data_directory=None):
    """Output a MolecularData object generated from an InteractionOperator

    Args:
        interaction_operator(InteractionOperator): two-body interaction
            operator defining the "molecular interaction" to be simulated.
        geometry(string or list of atoms):
        basis(string):  String denoting the basis set used to discretize the
            system.
        multiplicity(int): Spin multiplicity desired in the system.
        n_electrons(int): Number of electrons in the system
        reduce_spin(bool): True if one wishes to perform spin reduction on
            integrals that are given in interaction operator.  Assumes
            spatial (x) spin structure generically.

    Returns:
        molecule(MolecularData):
            Instance that captures the
            interaction_operator converted into the format that would come
            from an electronic structure package adorned with some meta-data
            that may be useful.
    """

    n_spin_orbitals = interaction_operator.n_qubits

    # Introduce bare molecular operator to fill
    molecule = chem.MolecularData(geometry=geometry,
                                  basis=basis,
                                  multiplicity=multiplicity,
                                  data_directory=data_directory)

    molecule.nuclear_repulsion = interaction_operator.constant

    # Remove spin from integrals and put into molecular operator
    if reduce_spin:
        reduction_indices = list(range(0, n_spin_orbitals, 2))
    else:
        reduction_indices = list(range(n_spin_orbitals))

    molecule.n_orbitals = len(reduction_indices)

    molecule.one_body_integrals = interaction_operator.one_body_tensor[
        numpy.ix_(reduction_indices, reduction_indices)]
    molecule.two_body_integrals = interaction_operator.two_body_tensor[
        numpy.ix_(reduction_indices, reduction_indices, reduction_indices,
                  reduction_indices)]

    # Fill in other metadata
    molecule.overlap_integrals = numpy.eye(molecule.n_orbitals)
    molecule.n_qubits = n_spin_orbitals
    molecule.n_electrons = n_electrons
    molecule.multiplicity = multiplicity

    return molecule
