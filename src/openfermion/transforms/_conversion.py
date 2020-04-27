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

"""Transformations acting on operators and RDMs."""

import itertools
from typing import Union

import numpy
import scipy

from openfermion.config import EQ_TOLERANCE
from openfermion.hamiltonians import MolecularData
from openfermion.ops import (DiagonalCoulombHamiltonian,
                             FermionOperator,
                             InteractionOperator,
                             InteractionRDM,
                             MajoranaOperator,
                             PolynomialTensor,
                             QuadraticHamiltonian,
                             QubitOperator,
                             BosonOperator,
                             QuadOperator)
from openfermion.ops._interaction_operator import InteractionOperatorError
from openfermion.ops._quadratic_hamiltonian import QuadraticHamiltonianError
from openfermion.utils import (boson_operator_sparse,
                               count_qubits,
                               is_hermitian,
                               jordan_wigner_sparse,
                               normal_ordered,
                               qubit_operator_sparse)


def get_sparse_operator(operator, n_qubits=None, trunc=None, hbar=1.):
    r"""Map an operator to a sparse matrix.

    If the input is not a QubitOperator, the Jordan-Wigner Transform is used.

    Args:
        operator: Currently supported operators include:
            FermionOperator, QubitOperator, DiagonalCoulombHamiltonian,
            PolynomialTensor, BosonOperator, QuadOperator.
        n_qubits(int): Number qubits in the system Hilbert space.
            Applicable only to fermionic systems.
        trunc (int): The size at which the Fock space should be truncated.
            Applicable only to bosonic systems.
        hbar (float): the value of hbar to use in the definition of the
            canonical commutation relation [q_i, p_j] = \delta_{ij} i hbar.
            Applicable only to the QuadOperator.
    """
    if isinstance(operator, (DiagonalCoulombHamiltonian, PolynomialTensor)):
        return jordan_wigner_sparse(get_fermion_operator(operator))
    elif isinstance(operator, FermionOperator):
        return jordan_wigner_sparse(operator, n_qubits)
    elif isinstance(operator, QubitOperator):
        return qubit_operator_sparse(operator, n_qubits)
    elif isinstance(operator, (BosonOperator, QuadOperator)):
        return boson_operator_sparse(operator, trunc, hbar)
    else:
        raise TypeError('Failed to convert a {} to a sparse matrix.'.format(
            type(operator).__name__))


def get_interaction_rdm(qubit_operator, n_qubits=None):
    """Build an InteractionRDM from measured qubit operators.

    Returns: An InteractionRDM object.
    """
    # Avoid circular import.
    from openfermion.transforms import jordan_wigner
    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    one_rdm = numpy.zeros((n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(n_qubits), repeat=2):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 0))))
        for term, coefficient in transformed_operator.terms.items():
            if term in qubit_operator.terms:
                one_rdm[i, j] += coefficient * qubit_operator.terms[term]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 1),
                                                              (k, 0), (l, 0))))
        for term, coefficient in transformed_operator.terms.items():
            if term in qubit_operator.terms:
                two_rdm[i, j, k, l] += coefficient * qubit_operator.terms[term]

    return InteractionRDM(one_rdm, two_rdm)


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

    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError('Invalid number of qubits specified.')

    # Normal order the terms and initialize.
    fermion_operator = normal_ordered(fermion_operator)
    constant = 0.
    one_body = numpy.zeros((n_qubits, n_qubits), complex)
    two_body = numpy.zeros((n_qubits, n_qubits,
                            n_qubits, n_qubits), complex)

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        # Ignore this term if the coefficient is zero
        if abs(coefficient) < EQ_TOLERANCE:
            continue

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

    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError('Invalid number of qubits specified.')

    # Normal order the terms and initialize.
    fermion_operator = normal_ordered(fermion_operator)
    constant = 0.
    combined_hermitian_part = numpy.zeros((n_qubits, n_qubits), complex)
    antisymmetric_part = numpy.zeros((n_qubits, n_qubits), complex)

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        # Ignore this term if the coefficient is zero
        if abs(coefficient) < EQ_TOLERANCE:
            continue

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
                    raise QuadraticHamiltonianError(
                        'FermionOperator does not map '
                        'to QuadraticHamiltonian (not Hermitian).')
                else:
                    matching_coefficient = -fermion_operator.terms[
                        conjugate_term].conjugate()
                    discrepancy = abs(coefficient - matching_coefficient)
                    if discrepancy > EQ_TOLERANCE:
                        raise QuadraticHamiltonianError(
                            'FermionOperator does not map '
                            'to QuadraticHamiltonian (not Hermitian).')
                antisymmetric_part[p, q] += .5 * coefficient
                antisymmetric_part[q, p] -= .5 * coefficient
            else:
                # ladder_type == [0, 0]
                # Need to check that the corresponding [1, 1] term is present
                conjugate_term = ((p, 1), (q, 1))
                if conjugate_term not in fermion_operator.terms:
                    raise QuadraticHamiltonianError(
                        'FermionOperator does not map '
                        'to QuadraticHamiltonian (not Hermitian).')
                else:
                    matching_coefficient = -fermion_operator.terms[
                        conjugate_term].conjugate()
                    discrepancy = abs(coefficient - matching_coefficient)
                    if discrepancy > EQ_TOLERANCE:
                        raise QuadraticHamiltonianError(
                            'FermionOperator does not map '
                            'to QuadraticHamiltonian (not Hermitian).')
                antisymmetric_part[p, q] -= .5 * coefficient.conjugate()
                antisymmetric_part[q, p] += .5 * coefficient.conjugate()
        elif not ignore_incompatible_terms:
            # Operator contains non-quadratic terms
            raise QuadraticHamiltonianError('FermionOperator does not map '
                                            'to QuadraticHamiltonian '
                                            '(contains non-quadratic terms).')

    # Compute Hermitian part
    hermitian_part = (combined_hermitian_part +
                      chemical_potential * numpy.eye(n_qubits))

    # Check that the operator is Hermitian
    if not is_hermitian(hermitian_part):
        raise QuadraticHamiltonianError(
            'FermionOperator does not map '
            'to QuadraticHamiltonian (not Hermitian).')

    # Form QuadraticHamiltonian and return.
    discrepancy = numpy.max(numpy.abs(antisymmetric_part))
    if discrepancy < EQ_TOLERANCE:
        # Hamiltonian conserves particle number
        quadratic_hamiltonian = QuadraticHamiltonian(
            hermitian_part, constant=constant,
            chemical_potential=chemical_potential)
    else:
        # Hamiltonian does not conserve particle number
        quadratic_hamiltonian = QuadraticHamiltonian(
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

    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError('Invalid number of qubits specified.')

    fermion_operator = normal_ordered(fermion_operator)
    constant = 0.
    one_body = numpy.zeros((n_qubits, n_qubits), complex)
    two_body = numpy.zeros((n_qubits, n_qubits), float)

    for term, coefficient in fermion_operator.terms.items():
        # Ignore this term if the coefficient is zero
        if abs(coefficient) < EQ_TOLERANCE:
            continue

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
    if not is_hermitian(one_body):
        raise ValueError(
            'FermionOperator does not map to DiagonalCoulombHamiltonian '
            '(not Hermitian).')

    return DiagonalCoulombHamiltonian(one_body, two_body, constant)


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
        fermion_operator += FermionOperator(
            ((p, 1), (q, 0)),
            operator.one_body[p, q])
        fermion_operator += FermionOperator(
            ((p, 1), (p, 0), (q, 1), (q, 0)),
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


def get_molecular_data(interaction_operator,
                       geometry=None, basis=None, multiplicity=None,
                       n_electrons=None, reduce_spin=True,
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
    molecule = MolecularData(geometry=geometry,
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
        numpy.ix_(reduction_indices, reduction_indices,
                  reduction_indices, reduction_indices)]

    # Fill in other metadata
    molecule.overlap_integrals = numpy.eye(molecule.n_orbitals)
    molecule.n_qubits = n_spin_orbitals
    molecule.n_electrons = n_electrons
    molecule.multiplicity = multiplicity

    return molecule


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
                    coeff = numpy.sqrt(hbar/2)
                    sign = 1
                elif d == 'p':
                    coeff = -1j*numpy.sqrt(hbar/2)
                    sign = -1

                tmp *= coeff*(BosonOperator(((i, 0)))
                              + BosonOperator(((i, 1)), sign))
            boson_operator += tmp

    else:
        raise TypeError("Only QuadOperator is currently "
                        "supported for get_boson_operator.")

    return boson_operator


def get_number_preserving_sparse_operator(
        fermion_op,
        num_qubits,
        num_electrons,
        spin_preserving=False,
        reference_determinant=None,
        excitation_level=None):
    """Initialize a Scipy sparse matrix in a specific symmetry sector.

    This method initializes a Scipy sparse matrix from a FermionOperator,
    explicitly working in a particular particle number sector. Optionally, it
    can also restrict the space to contain only states with a particular Sz.

    Finally, the Hilbert space can also be restricted to only those states
    which are reachable by excitations up to a fixed rank from an initial
    reference determinant.

    Args:
        fermion_op(FermionOperator): An instance of the FermionOperator class.
            It should not contain terms which do not preserve particle number.
            If spin_preserving is set to True it should also not contain terms
            which do not preserve the Sz (it is assumed that the ordering of
            the indices goes alpha, beta, alpha, beta, ...).
        num_qubits(int): The total number of qubits / spin-orbitals in the
            system.
        num_electrons(int): The number of particles in the desired Hilbert
            space.
        spin_preserving(bool): Whether or not the constructed operator should
            be defined in a space which has support only on states with the
            same Sz value as the reference_determinant.
        reference_determinant(list(bool)): A list, whose length is equal to
            num_qubits, which specifies which orbitals should be occupied in
            the reference state. If spin_preserving is set to True then the Sz
            value of this reference state determines the Sz value of the
            symmetry sector in which the generated operator acts. If a value
            for excitation_level is provided then the excitations are generated
            with respect to the reference state. In any case, the ordering of
            the states in the matrix representation of the operator depends on
            reference_determinant and the state corresponding to
            reference_determinant is the vector [1.0, 0.0, 0.0 ... 0.0]. Can be
            set to None in order to take the first num_electrons orbitals to be
            the occupied orbitals.
        excitation_level(int): The number of excitations from the reference
            state which should be included in the generated operator's matrix
            representation. Can be set to None to include all levels of
            excitation.

    Returns:
        sparse_op(scipy.sparse.csc_matrix): A sparse matrix representation of
            fermion_op in the basis set by the arguments.
    """

    # We use the Hartree-Fock determinant as a reference if none is provided.
    if reference_determinant is None:
        reference_determinant = numpy.array([i < num_electrons for i in
                                             range(num_qubits)])
    else:
        reference_determinant = numpy.asarray(reference_determinant)

    if excitation_level is None:
        excitation_level = num_electrons

    state_array = numpy.asarray(list(_iterate_basis_(
        reference_determinant, excitation_level, spin_preserving)))
    # Create a 1d array with each determinant encoded
    # as an integer for sorting purposes.
    int_state_array = state_array.dot(
        1 << numpy.arange(state_array.shape[1])[::-1])
    sorting_indices = numpy.argsort(int_state_array)

    space_size = state_array.shape[0]

    fermion_op = normal_ordered(fermion_op)

    sparse_op = scipy.sparse.csc_matrix((space_size, space_size), dtype=float)

    for term, coefficient in fermion_op.terms.items():
        if len(term) == 0:
            constant = coefficient * scipy.sparse.identity(
                space_size, dtype=float, format='csc')

            sparse_op += constant

        else:
            term_op = _build_term_op_(term, state_array, int_state_array,
                                      sorting_indices)

            sparse_op += coefficient * term_op

    return sparse_op


def _iterate_basis_(reference_determinant, excitation_level, spin_preserving):
    """A helper method which iterates over the specified basis states.

    Note that this method always yields the states in order of their excitation
    rank from the reference_determinant.

    Args:
        reference_determinant(list(bool)): A list of bools which indicates
            which orbitals are occupied and which are unoccupied in the
            reference state.
        excitation_level(int): The maximum excitation rank to iterate over.
        spin_preserving(bool): A bool which, if set to True, constrains the
            method to iterate over only those states which have the same Sz as
            reference_determinant.

    Yields:
        Lists of bools which indicate which orbitals are occupied and which are
            unoccupied in the current determinant.
    """
    if not spin_preserving:
        for order in range(excitation_level + 1):
            for determinant in _iterate_basis_order_(reference_determinant,
                                                     order):
                yield determinant

    else:
        alpha_excitation_level = min((numpy.sum(reference_determinant[::2]),
                                      excitation_level))
        beta_excitation_level = min((numpy.sum(reference_determinant[1::2]),
                                     excitation_level))

        for order in range(excitation_level + 1):
            for alpha_order in range(alpha_excitation_level + 1):
                beta_order = order - alpha_order
                if (beta_order < 0 or beta_order > beta_excitation_level):
                    continue

                for determinant in _iterate_basis_spin_order_(
                        reference_determinant, alpha_order, beta_order):
                    yield determinant


def _iterate_basis_order_(reference_determinant, order):
    """A helper for iterating over determinants of a fixed excitation rank.

    Args:
        reference_determinant(list(bool)): The reference state with respect to
            which we are iterating over excited determinants.
        order(int): The number of excitations from the modes which are occupied
            in the reference_determinant.

    Yields:
        Lists of bools which indicate which orbitals are occupied and which are
            unoccupied in the current determinant.
        """
    occupied_indices = numpy.where(reference_determinant)[0]
    unoccupied_indices = numpy.where(numpy.invert(reference_determinant))[0]

    for occ_ind, unocc_ind in itertools.product(
            itertools.combinations(occupied_indices, order),
            itertools.combinations(unoccupied_indices, order)):
        basis_state = reference_determinant.copy()

        occ_ind = list(occ_ind)
        unocc_ind = list(unocc_ind)

        basis_state[occ_ind] = False
        basis_state[unocc_ind] = True

        yield basis_state


def _iterate_basis_spin_order_(reference_determinant, alpha_order, beta_order):
    """Iterates over states with a fixed excitation rank for each spin sector.

    This helper method assumes that the two spin sectors are interleaved:
    [1_alpha, 1_beta, 2_alpha, 2_beta, ...].

    Args:
        reference_determinant(list(bool)): The reference state with respect to
            which we are iterating over excited determinants.
        alpha_order(int): The number of excitations from the alpha spin sector
            of the reference_determinant.
        beta_order(int): The number of excitations from the beta spin sector of
            the reference_determinant.

    Yields:
        Lists of bools which indicate which orbitals are occupied and which are
            unoccupied in the current determinant.
        """
    occupied_alpha_indices = numpy.where(
        reference_determinant[::2])[0] * 2
    unoccupied_alpha_indices = numpy.where(
        numpy.invert(reference_determinant[::2]))[0] * 2
    occupied_beta_indices = numpy.where(
        reference_determinant[1::2])[0] * 2 + 1
    unoccupied_beta_indices = numpy.where(
        numpy.invert(reference_determinant[1::2]))[0] * 2 + 1

    for (alpha_occ_ind,
            alpha_unocc_ind,
            beta_occ_ind,
            beta_unocc_ind) in itertools.product(
            itertools.combinations(occupied_alpha_indices, alpha_order),
            itertools.combinations(unoccupied_alpha_indices, alpha_order),
            itertools.combinations(occupied_beta_indices, beta_order),
            itertools.combinations(unoccupied_beta_indices, beta_order)):
        basis_state = reference_determinant.copy()

        alpha_occ_ind = list(alpha_occ_ind)
        alpha_unocc_ind = list(alpha_unocc_ind)
        beta_occ_ind = list(beta_occ_ind)
        beta_unocc_ind = list(beta_unocc_ind)

        basis_state[alpha_occ_ind] = False
        basis_state[alpha_unocc_ind] = True
        basis_state[beta_occ_ind] = False
        basis_state[beta_unocc_ind] = True

        yield basis_state


def _build_term_op_(term, state_array, int_state_array, sorting_indices):
    """Builds a scipy sparse representation of a term from a FermionOperator.

    Args:
        term(tuple of tuple(int, int)s): The argument is a tuple of tuples
            representing a product of normal ordered fermionic creation and
            annihilation operators, each of which is of the form (int, int)
            where the first int indicates which site the operator acts on and
            the second int indicates whether the operator is a creation
            operator (1) or an annihilation operator (0). See the
            implementation of FermionOperator for more details.
        state_array(ndarray(bool)): A Numpy array which encodes each of the
            determinants in the space we are working in a bools which indicate
            the occupation of each mode. See the implementation of
            get_number_preserving_sparse_operator for more details.
        int_state_array(ndarray(int)): A one dimensional Numpy array which
            encodes the integer representation of the binary number
            corresponding to each determinant in state_array.
        sorting_indices(ndarray.view): A Numpy view which sorts
            int_state_array. This, together with int_state_array, allows for a
            quick lookup of the position of a particular determinant in
            state_array by converting it to its integer representation and
            searching through the sorted int_state_array.

    Raises:
        ValueError: If term does not represent a particle number conserving
            operator.

    Returns:
        A scipy.sparse.csc_matrix which corresponds to the operator specified
            by term expressed in the basis corresponding to the other arguments
            of the method."""

    space_size = state_array.shape[0]

    needs_to_be_occupied = []
    needs_to_be_unoccupied = []

    # We keep track of the number of creation and annihilation operators and
    # ensure that there are an equal number of them in order to help detect
    # invalid inputs.
    delta = 0
    for index, op_type in reversed(term):
        if op_type == 0:
            needs_to_be_occupied.append(index)
            delta -= 1
        else:
            if index not in needs_to_be_occupied:
                needs_to_be_unoccupied.append(index)
            delta += 1

    if delta != 0:
        raise ValueError(
            "The supplied operator doesn't preserve particle number")

    # We search for every state which has the necessary orbitals occupied and
    # unoccupied in order to not be immediately zeroed out based on the
    # creation and annihilation operators specified in term.
    maybe_valid_states = numpy.where(
        numpy.logical_and(
            numpy.all(state_array[:, needs_to_be_occupied], axis=1),
            numpy.logical_not(
                numpy.any(state_array[:, needs_to_be_unoccupied], axis=1))))[0]

    data = []
    row_ind = []
    col_ind = []
    shape = (space_size, space_size)

    # For each state that is not immediately zeroed out by the action of our
    # operator we check to see if the determinant which this state gets mapped
    # to is in the space we are considering.
    # Note that a failure to find any state does not necessarily indicate that
    # term specifies an invalid operator. For example, if we are restricting
    # ourselves to double excitations from a fixed reference state then the
    # action of term on some of our basis states may lead to determinants with
    # more than two excitations from the reference. These more than double
    # excited determinants are not included in the matrix representation (and
    # hence, will not be present in state_array).
    for _, state in enumerate(maybe_valid_states):
        determinant = state_array[state, :]
        target_determinant = determinant.copy()

        parity = 1
        for i, _ in reversed(term):
            area_to_check = target_determinant[0:i]
            parity *= (-1) ** numpy.sum(area_to_check)

            target_determinant[i] = not target_determinant[i]

        int_encoding = target_determinant.dot(
            1 << numpy.arange(target_determinant.size)[::-1])

        target_state_index_sorted = numpy.searchsorted(int_state_array,
                                                       int_encoding,
                                                       sorter=sorting_indices)

        target_state = sorting_indices[target_state_index_sorted]

        if int_state_array[target_state] == int_encoding:
            # Then target state is in the space considered:
            data.append(parity)
            row_ind.append(target_state)
            col_ind.append(state)

    data = numpy.asarray(data)
    row_ind = numpy.asarray(row_ind)
    col_ind = numpy.asarray(col_ind)

    term_op = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)

    return term_op
