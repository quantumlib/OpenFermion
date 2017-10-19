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
from __future__ import absolute_import

import itertools

import numpy
from future.utils import iteritems

from openfermion.hamiltonians import MolecularData
from openfermion.ops import (FermionOperator,
                             normal_ordered,
                             InteractionOperator,
                             InteractionRDM,
                             QubitOperator)
from openfermion.ops._interaction_operator import InteractionOperatorError
from openfermion.utils import (count_qubits,
                               jordan_wigner_sparse,
                               qubit_operator_sparse)


def get_sparse_operator(operator, n_qubits=None):
    """Map a Fermion, Qubit, or InteractionOperator to a SparseOperator."""
    if isinstance(operator, InteractionOperator):
        return get_sparse_interaction_operator(operator)
    elif isinstance(operator, FermionOperator):
        return jordan_wigner_sparse(operator, n_qubits)
    elif isinstance(operator, QubitOperator):
        if n_qubits is None:
            n_qubits = count_qubits(operator)
        return qubit_operator_sparse(operator, n_qubits)


def get_sparse_interaction_operator(interaction_operator):
    fermion_operator = get_fermion_operator(interaction_operator)
    sparse_operator = jordan_wigner_sparse(fermion_operator)
    return sparse_operator


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
        for term, coefficient in iteritems(transformed_operator.terms):
            if term in qubit_operator.terms:
                one_rdm[i, j] += coefficient * qubit_operator.terms[term]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 1),
                                                              (k, 0), (l, 0))))
        for term, coefficient in iteritems(transformed_operator.terms):
            if term in qubit_operator.terms:
                two_rdm[i, j, k, l] += coefficient * qubit_operator.terms[term]

    return InteractionRDM(one_rdm, two_rdm)


def get_interaction_operator(fermion_operator, n_qubits=None):
    """Convert a 2-body fermionic operator to InteractionOperator.

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


def get_fermion_operator(interaction_operator):
    """Output InteractionOperator as instance of FermionOperator class.

    Returns:
        fermion_operator: An instance of the FermionOperator class.
    """
    # Initialize with identity term.
    fermion_operator = FermionOperator((), interaction_operator.constant)
    n_qubits = count_qubits(interaction_operator)

    # Add one-body terms.
    for p, q in itertools.product(range(n_qubits), repeat=2):
        fermion_operator += FermionOperator(
            ((p, 1), (q, 0)), coefficient=interaction_operator[p, q])

    # Add two-body terms.
    for p, q, r, s in itertools.product(range(n_qubits), repeat=4):
        coefficient = interaction_operator[p, q, r, s]
        fermion_operator += FermionOperator(((p, 1), (q, 1),
                                             (r, 0), (s, 0)),
                                            coefficient)

    return fermion_operator


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
        molecule(MolecularData): Instance that captures the
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
