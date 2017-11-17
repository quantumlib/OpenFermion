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

"""This module provides functions to interface with scipy.sparse."""
from __future__ import absolute_import

from functools import reduce
from future.utils import iteritems

import itertools
import numpy
import numpy.linalg
import scipy
import scipy.sparse
import scipy.sparse.linalg
import warnings

from openfermion.config import *
from openfermion.ops import (FermionOperator, hermitian_conjugated,
                             normal_ordered, number_operator,
                             QuadraticHamiltonian, QubitOperator)
from openfermion.utils import (commutator, fourier_transform,
                               gaussian_state_preparation_circuit, Grid)
from openfermion.hamiltonians._jellium import (momentum_vector,
                                               position_vector,
                                               grid_indices)


# Make global definitions.
identity_csc = scipy.sparse.identity(2, format='csc', dtype=complex)
pauli_x_csc = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
pauli_y_csc = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
pauli_z_csc = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
q_raise_csc = (pauli_x_csc - 1.j * pauli_y_csc) / 2.
q_lower_csc = (pauli_x_csc + 1.j * pauli_y_csc) / 2.
pauli_matrix_map = {'I': identity_csc, 'X': pauli_x_csc,
                    'Y': pauli_y_csc, 'Z': pauli_z_csc}


def wrapped_kronecker(operator_1, operator_2):
    """Return the Kronecker product of two sparse.csc_matrix operators."""
    return scipy.sparse.kron(operator_1, operator_2, 'csc')


def kronecker_operators(*args):
    """Return the Kronecker product of multiple sparse.csc_matrix operators."""
    return reduce(wrapped_kronecker, *args)


def jordan_wigner_ladder_sparse(n_qubits, tensor_factor, ladder_type):
    """Make a matrix representation of a fermion ladder operator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Args:
        index: This is a nonzero integer. The integer indicates the tensor
            factor and the sign indicates raising or lowering.
        n_qubits(int): Number qubits in the system Hilbert space.

    Returns:
        The corresponding SparseOperator.
    """
    parities = tensor_factor * [pauli_z_csc]
    identities = [scipy.sparse.identity(
        2 ** (n_qubits - tensor_factor - 1), dtype=complex, format='csc')]
    if ladder_type:
        operator = kronecker_operators(parities + [q_raise_csc] + identities)
    else:
        operator = kronecker_operators(parities + [q_lower_csc] + identities)
    return operator


def jordan_wigner_sparse(fermion_operator, n_qubits=None):
    """Initialize a SparseOperator from a FermionOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Args:
        fermion_operator(FermionOperator): instance of the FermionOperator
            class.
        n_qubits(int): Number of qubits.

    Returns:
        The corresponding SparseOperator.
    """
    if n_qubits is None:
        from openfermion.utils import count_qubits
        n_qubits = count_qubits(fermion_operator)

    # Create a list of raising and lowering operators for each orbital.
    jw_operators = []
    for tensor_factor in range(n_qubits):
        jw_operators += [(jordan_wigner_ladder_sparse(n_qubits,
                                                      tensor_factor,
                                                      0),
                          jordan_wigner_ladder_sparse(n_qubits,
                                                      tensor_factor,
                                                      1))]

    # Construct the SparseOperator.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        sparse_matrix = coefficient * scipy.sparse.identity(
            2 ** n_qubits, dtype=complex, format='csc')
        for ladder_operator in term:
            sparse_matrix = sparse_matrix * jw_operators[
                ladder_operator[0]][ladder_operator[1]]

        if coefficient:
            # Extract triplets from sparse_term.
            sparse_matrix = sparse_matrix.tocoo(copy=False)
            values_list.append(sparse_matrix.data)
            (row, column) = sparse_matrix.nonzero()
            row_list.append(row)
            column_list.append(column)

    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix((
        values_list, (row_list, column_list)),
        shape=(n_hilbert, n_hilbert)).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def qubit_operator_sparse(qubit_operator, n_qubits=None):
    """Initialize a SparseOperator from a QubitOperator.

    Args:
        qubit_operator(QubitOperator): instance of the QubitOperator class.
        n_qubits (int): Number of qubits.

    Returns:
        The corresponding SparseOperator.
    """
    from openfermion.utils import count_qubits
    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits < count_qubits(qubit_operator):
        raise ValueError('Invalid number of qubits specified.')

    # Construct the SparseOperator.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]

    # Loop through the terms.
    for qubit_term in qubit_operator.terms:
        tensor_factor = 0
        coefficient = qubit_operator.terms[qubit_term]
        sparse_operators = [coefficient]
        for pauli_operator in qubit_term:

            # Grow space for missing identity operators.
            if pauli_operator[0] > tensor_factor:
                identity_qubits = pauli_operator[0] - tensor_factor
                identity = scipy.sparse.identity(
                    2 ** identity_qubits, dtype=complex, format='csc')
                sparse_operators += [identity]

            # Add actual operator to the list.
            sparse_operators += [pauli_matrix_map[pauli_operator[1]]]
            tensor_factor = pauli_operator[0] + 1

        # Grow space at end of string unless operator acted on final qubit.
        if tensor_factor < n_qubits or not qubit_term:
            identity_qubits = n_qubits - tensor_factor
            identity = scipy.sparse.identity(
                2 ** identity_qubits, dtype=complex, format='csc')
            sparse_operators += [identity]

        # Extract triplets from sparse_term.
        sparse_matrix = kronecker_operators(sparse_operators)
        values_list.append(sparse_matrix.tocoo(copy=False).data)
        (column, row) = sparse_matrix.nonzero()
        column_list.append(column)
        row_list.append(row)

    # Create sparse operator.
    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix((
        values_list, (row_list, column_list)),
        shape=(n_hilbert, n_hilbert)).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def jw_slater_determinant(occupied_orbitals, n_orbitals):
    """Function to produce a Slater determinant in JW representation.

    Args:
        occupied_orbitals(list): A list of integers representing the indices
            of the occupied orbitals in the desired Slater determinant
        n_orbitals(int): The total number of orbitals

    Returns:
        slater_determinant(sparse): The JW-encoded Slater determinant as a
            sparse matrix
    """
    one_index = sum([2 ** (n_orbitals - 1 - i) for i in occupied_orbitals])
    slater_determinant = scipy.sparse.csc_matrix(([1.], ([one_index], [0])),
                                                 shape=(2 ** n_orbitals, 1),
                                                 dtype=float)
    return slater_determinant


def jw_hartree_fock_state(n_electrons, n_orbitals):
    """Function to produce Hartree-Fock state in JW representation."""
    hartree_fock_state = jw_slater_determinant(range(n_electrons), n_orbitals)
    return hartree_fock_state


def jw_number_indices(n_electrons, n_qubits):
    """Return the indices for n_electrons in n_qubits under JW encoding

    Calculates the indices for all possible arrangements of n-electrons
        within n-qubit orbitals when a Jordan-Wigner encoding is used.
        Useful for restricting generic operators or vectors to a particular
        particle number space when desired

    Args:
        n_electrons(int): Number of particles to restrict the operator to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        indices(list): List of indices in a 2^n length array that indicate
            the indices of constant particle number within n_qubits
            in a Jordan-Wigner encoding.

    """
    occupations = itertools.combinations(range(n_qubits), n_electrons)
    indices = [sum([2**n for n in occupation])
               for occupation in occupations]
    return indices


def jw_number_restrict_operator(operator, n_electrons, n_qubits=None):
    """Restrict a Jordan-Wigner encoded operator to a given particle number

    Args:
        sparse_operator(ndarray or sparse): Numpy operator acting on
            the space of n_qubits.
        n_electrons(int): Number of particles to restrict the operator to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        new_operator(ndarray or sparse): Numpy operator restricted to
            acting on states with the same particle number.
    """
    if n_qubits is None:
        n_qubits = int(numpy.log2(operator.shape[0]))

    select_indices = jw_number_indices(n_electrons, n_qubits)
    return operator[numpy.ix_(select_indices, select_indices)]


def jw_get_ground_states_by_particle_number(sparse_operator, particle_number,
                                            sparse=True, num_eigs=3):
    """For a Jordan-Wigner encoded Hermitian operator, compute the lowest
    eigenvalue and eigenstates at a particular particle number. The operator
    must conserve particle number.

    Args:
        sparse_operator(sparse): A Jordan-Wigner encoded sparse operator.
        particle_number(int): The particle number at which to compute
            ground states.
        sparse(boolean, optional): Whether to use sparse eigensolver.
            Default is True.
        num_eigs(int, optional): The number of eigenvalues to request from the
            sparse eigensolver. Needs to be at least as large as the degeneracy
            of the ground energy in order to obtain all ground states.
            Only used if `sparse=True`. Default is 3.

    Returns:
        ground_energy(float): The lowest eigenvalue of sparse_operator within
            the eigenspace of the number operator corresponding to
            particle_number.

        ground_states(list[ndarray]): A list of the corresponding eigenstates.

    Warning:
        The running time of this method is exponential in the number of qubits.
    """
    # Check if operator is Hermitian
    if not is_hermitian(sparse_operator):
        raise ValueError('sparse_operator must be Hermitian.')

    n_qubits = int(numpy.log2(sparse_operator.shape[0]))

    # Check if operator conserves particle number
    sparse_num_op = jordan_wigner_sparse(number_operator(n_qubits))
    com = commutator(sparse_num_op, sparse_operator)
    if com.nnz:
        maxval = max(map(abs, com.data))
        if maxval > EQ_TOLERANCE:
            raise ValueError('sparse_operator must conserve particle number.')

    # Get the operator restricted to the subspace of the desired
    # particle number
    restricted_operator = jw_number_restrict_operator(sparse_operator,
                                                      particle_number,
                                                      n_qubits)

    if sparse and num_eigs >= restricted_operator.shape[0] - 1:
        # Restricted operator too small for sparse eigensolver
        sparse = False

    # Compute eigenvalues and eigenvectors
    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
                                                     k=num_eigs,
                                                     which='SA')
        if abs(max(eigvals) - min(eigvals)) < EQ_TOLERANCE:
            warnings.warn('The lowest {} eigenvalues are degenerate. '
                          'There may be more ground states; increase '
                          'num_eigs or set sparse=False to get '
                          'them.'.format(num_eigs),
                          RuntimeWarning)
    else:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = numpy.linalg.eigh(dense_restricted_operator)

    # Get the ground energy
    if sparse:
        ground_energy = sorted(eigvals)[0]
    else:
        # No need to sort in the case of dense eigenvalue computation
        ground_energy = eigvals[0]

    # Get the indices of eigenvectors corresponding to the ground energy
    ground_state_indices = numpy.where(abs(eigvals - ground_energy) <
                                       EQ_TOLERANCE)

    ground_states = list()

    for i in ground_state_indices[0]:
        restricted_ground_state = eigvecs[:, i]
        # Expand this ground state to the whole vector space
        number_indices = jw_number_indices(particle_number, n_qubits)
        expanded_ground_state = scipy.sparse.csc_matrix(
                (restricted_ground_state.flatten(),
                 (number_indices, [0] * len(number_indices))),
                shape=(2 ** n_qubits, 1))
        # Add the expanded ground state to the list
        ground_states.append(expanded_ground_state)

    return ground_energy, ground_states


def jw_get_gaussian_state(quadratic_hamiltonian, occupied_orbitals=None):
    """Compute an eigenvalue and eigenstate of a quadratic Hamiltonian.

    Eigenstates of a quadratic Hamiltonian are also known as fermionic
    Gaussian states.

    Args:
        quadratic_hamiltonian(QuadraticHamiltonian):
            The Hamiltonian whose eigenstate is desired.
        occupied_orbitals(list):
            A list of integers representing the indices of the occupied
            orbitals in the desired Gaussian state. If this is None
            (the default), then it is assumed that the ground state is
            desired, i.e., the orbitals with negative energies are filled.

    Returns
    -------
        energy (float):
            The eigenvalue.
        state (sparse):
            The eigenstate in scipy.sparse csc format.
    """
    if not isinstance(quadratic_hamiltonian, QuadraticHamiltonian):
        raise ValueError('Input must be an instance of QuadraticHamiltonian.')

    n_qubits = quadratic_hamiltonian.n_qubits

    # Compute the energy
    orbital_energies, constant = quadratic_hamiltonian.orbital_energies()
    if occupied_orbitals is None:
        # The ground energy is desired
        if quadratic_hamiltonian.conserves_particle_number:
            num_negative_energies = numpy.count_nonzero(
                    orbital_energies < -EQ_TOLERANCE)
            occupied_orbitals = range(num_negative_energies)
        else:
            occupied_orbitals = []
    energy = numpy.sum(orbital_energies[occupied_orbitals]) + constant

    # Obtain the circuit that prepares the Gaussian state
    circuit_description, start_orbitals = gaussian_state_preparation_circuit(
            quadratic_hamiltonian, occupied_orbitals)

    # Initialize the starting state
    state = jw_slater_determinant(start_orbitals, n_qubits)

    # Apply the circuit
    particle_hole_transformation = (
            jw_sparse_particle_hole_transformation_last_mode(n_qubits))
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            if op == 'pht':
                state = particle_hole_transformation.dot(state)
            else:
                i, j, theta, phi = op
                state = jw_sparse_givens_rotation(
                            i, j, theta, phi, n_qubits).dot(state)

    return energy, state


def jw_sparse_givens_rotation(i, j, theta, phi, n_qubits):
    """Return the matrix (acting on a full wavefunction) that performs a
    Givens rotation of modes i and j in the Jordan-Wigner encoding."""
    if j != i + 1:
        raise ValueError('Only adjacent modes can be rotated.')
    if j > n_qubits - 1:
        raise ValueError('Too few qubits requested.')

    cosine = numpy.cos(theta)
    sine = numpy.sin(theta)
    phase = numpy.exp(1.j * phi)

    # Create the two-qubit rotation matrix
    rotation_matrix = scipy.sparse.csc_matrix(
            ([1., phase * cosine, -phase * sine, sine, cosine, phase],
             ((0, 1, 1, 2, 2, 3), (0, 1, 2, 1, 2, 3))),
            shape=(4, 4))

    # Initialize identity operators
    left_eye = scipy.sparse.eye(2 ** i, format='csc')
    right_eye = scipy.sparse.eye(2 ** (n_qubits - 1 - j), format='csc')

    # Construct the matrix and return
    givens_matrix = kronecker_operators([left_eye, rotation_matrix, right_eye])

    return givens_matrix


def jw_sparse_particle_hole_transformation_last_mode(n_qubits):
    """Return the matrix (acting on a full wavefunction) that performs a
    particle-hole transformation on the last mode in the Jordan-Wigner
    encoding.
    """
    left_eye = scipy.sparse.eye(2 ** (n_qubits - 1), format='csc')
    return kronecker_operators([left_eye, pauli_matrix_map['X']])


def get_density_matrix(states, probabilities):
    n_qubits = states[0].shape[0]
    density_matrix = scipy.sparse.csc_matrix(
        (n_qubits, n_qubits), dtype=complex)
    for state, probability in zip(states, probabilities):
        density_matrix = density_matrix + probability * state * state.getH()
    return density_matrix


def is_hermitian(sparse_operator):
    """Test if matrix is Hermitian."""
    difference = sparse_operator - sparse_operator.getH()
    if difference.nnz:
        discrepancy = max(map(abs, difference.data))
        if discrepancy > EQ_TOLERANCE:
            return False
    return True


def get_ground_state(sparse_operator):
    """Compute lowest eigenvalue and eigenstate.
    Returns:
        eigenvalue: The lowest eigenvalue, a float.
        eigenstate: The lowest eigenstate in scipy.sparse csc format.
    """
    if not is_hermitian(sparse_operator):
        raise ValueError('sparse_operator must be Hermitian.')

    values, vectors = scipy.sparse.linalg.eigsh(
        sparse_operator, 2, which='SA', maxiter=1e7)

    order = numpy.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eigenvalue = values[0]
    eigenstate = scipy.sparse.csc_matrix(vectors[:, 0])
    return eigenvalue, eigenstate.T


def sparse_eigenspectrum(sparse_operator):
    """Perform a dense diagonalization.

    Returns:
        eigenspectrum: The lowest eigenvalues in a numpy array.
    """
    dense_operator = sparse_operator.todense()
    if is_hermitian(sparse_operator):
        eigenspectrum = numpy.linalg.eigvalsh(dense_operator)
    else:
        eigenspectrum = numpy.linalg.eigvals(dense_operator)
    return numpy.sort(eigenspectrum)


def expectation(sparse_operator, state):
    """Compute expectation value of operator with a state.

    Args:
        state: scipy.sparse.csc vector representing a pure state,
            or, a scipy.sparse.csc matrix representing a density matrix.

    Returns:
        A real float giving expectation value.

    Raises:
        ValueError: Input state has invalid format.
    """
    # Handle density matrix.
    if state.shape == sparse_operator.shape:
        product = state * sparse_operator
        expectation = numpy.sum(product.diagonal())

    elif state.shape == (sparse_operator.shape[0], 1):
        # Handle state vector.
        expectation = state.getH() * sparse_operator * state
        expectation = expectation[0, 0]

    else:
        # Handle exception.
        raise ValueError('Input state has invalid format.')

    # Return.
    return expectation


def expectation_computational_basis_state(operator, computational_basis_state):
    """Compute expectation value of operator with a  state.

    Args:
        operator: Qubit or FermionOperator to evaluate expectation value of.
                  If operator is a FermionOperator, it must be normal-ordered.
        computational_basis_state (scipy.sparse vector / list): normalized
            computational basis state (if scipy.sparse vector), or list of
            occupied orbitals.

    Returns:
        A real float giving expectation value.

    Raises:
        TypeError: Incorrect operator or state type.
    """
    if isinstance(operator, QubitOperator):
        raise NotImplementedError('Not yet implemented for QubitOperators.')

    if not isinstance(operator, FermionOperator):
        raise TypeError('operator must be a FermionOperator.')

    occupied_orbitals = computational_basis_state

    if not isinstance(occupied_orbitals, list):
        computational_basis_state_index = (
            occupied_orbitals.nonzero()[0][0])

        occupied_orbitals = [digit == '1' for digit in
                             bin(computational_basis_state_index)[2:]][::-1]

    expectation_value = operator.terms.get((), 0.0)

    for i in range(len(occupied_orbitals)):
        if occupied_orbitals[i]:
            expectation_value += operator.terms.get(
                ((i, 1), (i, 0)), 0.0)

            for j in range(i + 1, len(occupied_orbitals)):
                expectation_value -= operator.terms.get(
                    ((j, 1), (i, 1), (j, 0), (i, 0)), 0.0)

    return expectation_value


def expectation_db_operator_with_pw_basis_state(
        operator, plane_wave_occ_orbitals, n_spatial_orbitals, grid,
        spinless):
    """Compute expectation value of a dual basis operator with a plane
    wave computational basis state.

    Args:
        operator: Dual-basis representation of FermionOperator to evaluate
                  expectation value of. Can have at most 3-body terms.
        plane_wave_occ_orbitals (list): list of occupied plane-wave orbitals.
        n_spatial_orbitals (int): Number of spatial orbitals.
        grid (openfermion.utils.Grid): The grid used for discretization.
        spinless (bool): Whether the system is spinless.

    Returns:
        A real float giving the expectation value.
    """
    expectation_value = operator.terms.get((), 0.0)

    for single_action, coefficient in iteritems(operator.terms):
        if len(single_action) == 2:
            expectation_value += coefficient * (
                expectation_one_body_db_operator_computational_basis_state(
                    single_action, plane_wave_occ_orbitals, grid, spinless) /
                n_spatial_orbitals)

        elif len(single_action) == 4:
            expectation_value += coefficient * (
                expectation_two_body_db_operator_computational_basis_state(
                    single_action, plane_wave_occ_orbitals, grid, spinless) /
                n_spatial_orbitals ** 2)

        elif len(single_action) == 6:
            expectation_value += coefficient * (
                expectation_three_body_db_operator_computational_basis_state(
                    single_action, plane_wave_occ_orbitals, grid, spinless) /
                n_spatial_orbitals ** 3)

    return expectation_value


def expectation_one_body_db_operator_computational_basis_state(
        dual_basis_action, plane_wave_occ_orbitals, grid, spinless):
    """Compute expectation value of a 1-body dual-basis operator with a
    plane wave computational basis state.

    Args:
        dual_basis_action: Dual-basis action of FermionOperator to
                           evaluate expectation value of.
        plane_wave_occ_orbitals (list): list of occupied plane-wave orbitals.
        grid (openfermion.utils.Grid): The grid used for discretization.
        spinless (bool): Whether the system is spinless.

    Returns:
        A real float giving the expectation value.
    """
    expectation_value = 0.0

    r_p = position_vector(grid_indices(dual_basis_action[0][0],
                                       grid, spinless), grid)
    r_q = position_vector(grid_indices(dual_basis_action[1][0],
                                       grid, spinless), grid)

    for orbital in plane_wave_occ_orbitals:
        # If there's spin, p and q have to have the same parity (spin),
        # and the new orbital has to have the same spin as these.
        k_orbital = momentum_vector(grid_indices(orbital,
                                                 grid, spinless), grid)
        # The Fourier transform is spin-conserving. This means that p, q,
        # and the new orbital all have to have the same spin (parity).
        if spinless or (dual_basis_action[0][0] % 2 ==
                        dual_basis_action[1][0] % 2 == orbital % 2):
            expectation_value += numpy.exp(-1j * k_orbital.dot(r_p - r_q))

    return expectation_value


def expectation_two_body_db_operator_computational_basis_state(
        dual_basis_action, plane_wave_occ_orbitals, grid, spinless):
    """Compute expectation value of a 2-body dual-basis operator with a
    plane wave computational basis state.

    Args:
        dual_basis_action: Dual-basis action of FermionOperator to
                           evaluate expectation value of.
        plane_wave_occ_orbitals (list): list of occupied plane-wave orbitals.
        grid (openfermion.utils.Grid): The grid used for discretization.
        spinless (bool): Whether the system is spinless.

    Returns:
        A float giving the expectation value.
    """
    expectation_value = 0.0

    r = {}
    for i in range(4):
        r[i] = position_vector(grid_indices(dual_basis_action[i][0], grid,
                                            spinless), grid)

    rr = {}
    k_map = {}
    for i in range(2):
        rr[i] = {}
        k_map[i] = {}
        for j in range(2, 4):
            rr[i][j] = r[i] - r[j]
            k_map[i][j] = {}

    # Pre-computations.
    for o in plane_wave_occ_orbitals:
        k = momentum_vector(grid_indices(o, grid, spinless), grid)
        for i in range(2):
            for j in range(2, 4):
                k_map[i][j][o] = k.dot(rr[i][j])

    for orbital1 in plane_wave_occ_orbitals:
        k1ac = k_map[0][2][orbital1]
        k1ad = k_map[0][3][orbital1]

        for orbital2 in plane_wave_occ_orbitals:
            if orbital1 != orbital2:
                k2bc = k_map[1][2][orbital2]
                k2bd = k_map[1][3][orbital2]

                # The Fourier transform is spin-conserving. This means that
                # the parity of the orbitals involved in the transition must
                # be the same.
                if spinless or (
                        (dual_basis_action[0][0] % 2 ==
                         dual_basis_action[3][0] % 2 == orbital1 % 2) and
                        (dual_basis_action[1][0] % 2 ==
                         dual_basis_action[2][0] % 2 == orbital2 % 2)):
                    value = numpy.exp(-1j * (k1ad + k2bc))

                    # Add because it came from two anti-commutations.
                    expectation_value += value

                # The Fourier transform is spin-conserving. This means that
                # the parity of the orbitals involved in the transition must
                # be the same.
                if spinless or (
                        (dual_basis_action[0][0] % 2 ==
                         dual_basis_action[2][0] % 2 == orbital1 % 2) and
                        (dual_basis_action[1][0] % 2 ==
                         dual_basis_action[3][0] % 2 == orbital2 % 2)):
                    value = numpy.exp(-1j * (k1ac + k2bd))

                    # Subtract because it came from a single anti-commutation.
                    expectation_value -= value

    return expectation_value


def expectation_three_body_db_operator_computational_basis_state(
        dual_basis_action, plane_wave_occ_orbitals, grid, spinless):
    """Compute expectation value of a 3-body dual-basis operator with a
    plane wave computational basis state.

    Args:
        dual_basis_action: Dual-basis action of FermionOperator to
                           evaluate expectation value of.
        plane_wave_occ_orbitals (list): list of occupied plane-wave orbitals.
        grid (openfermion.utils.Grid): The grid used for discretization.
        spinless (bool): Whether the system is spinless.

    Returns:
        A float giving the expectation value.
    """
    expectation_value = 0.0

    r = {}
    for i in range(6):
        r[i] = position_vector(grid_indices(dual_basis_action[i][0], grid,
                                            spinless), grid)

    rr = {}
    k_map = {}
    for i in range(3):
        rr[i] = {}
        k_map[i] = {}
        for j in range(3, 6):
            rr[i][j] = r[i] - r[j]
            k_map[i][j] = {}

    # Pre-computations.
    for o in plane_wave_occ_orbitals:
        k = momentum_vector(grid_indices(o, grid, spinless), grid)
        for i in range(3):
            for j in range(3, 6):
                k_map[i][j][o] = k.dot(rr[i][j])

    for orbital1 in plane_wave_occ_orbitals:
        k1ad = k_map[0][3][orbital1]
        k1ae = k_map[0][4][orbital1]
        k1af = k_map[0][5][orbital1]

        for orbital2 in plane_wave_occ_orbitals:
            if orbital1 != orbital2:
                k2bd = k_map[1][3][orbital2]
                k2be = k_map[1][4][orbital2]
                k2bf = k_map[1][5][orbital2]

                for orbital3 in plane_wave_occ_orbitals:
                    if orbital1 != orbital3 and orbital2 != orbital3:
                        k3cd = k_map[2][3][orbital3]
                        k3ce = k_map[2][4][orbital3]
                        k3cf = k_map[2][5][orbital3]

                        # Handle \delta_{ad} \delta_{bf} \delta_{ce} after FT.
                        # The Fourier transform is spin-conserving.
                        if spinless or (
                                (dual_basis_action[0][0] % 2 ==
                                 dual_basis_action[3][0] % 2 ==
                                 orbital1 % 2) and
                                (dual_basis_action[1][0] % 2 ==
                                 dual_basis_action[5][0] % 2 ==
                                 orbital2 % 2) and
                                (dual_basis_action[2][0] % 2 ==
                                 dual_basis_action[4][0] % 2 ==
                                 orbital3 % 2)):
                            expectation_value += numpy.exp(-1j * (
                                k1ad + k2bf + k3ce))

                        # Handle -\delta_{ad} \delta_{be} \delta_{cf} after FT.
                        # The Fourier transform is spin-conserving.
                        if spinless or (
                                (dual_basis_action[0][0] % 2 ==
                                 dual_basis_action[3][0] % 2 ==
                                 orbital1 % 2) and
                                (dual_basis_action[1][0] % 2 ==
                                 dual_basis_action[4][0] % 2 ==
                                 orbital2 % 2) and
                                (dual_basis_action[2][0] % 2 ==
                                 dual_basis_action[5][0] % 2 ==
                                 orbital3 % 2)):
                            expectation_value -= numpy.exp(-1j * (
                                k1ad + k2be + k3cf))

                        # Handle -\delta_{ae} \delta_{bf} \delta_{cd} after FT.
                        # The Fourier transform is spin-conserving.
                        if spinless or (
                                (dual_basis_action[0][0] % 2 ==
                                 dual_basis_action[4][0] % 2 ==
                                 orbital1 % 2) and
                                (dual_basis_action[1][0] % 2 ==
                                 dual_basis_action[5][0] % 2 ==
                                 orbital2 % 2) and
                                (dual_basis_action[2][0] % 2 ==
                                 dual_basis_action[3][0] % 2 ==
                                 orbital3 % 2)):
                            expectation_value -= numpy.exp(-1j * (
                                k1ae + k2bf + k3cd))

                        # Handle \delta_{ae} \delta_{bd} \delta_{cf} after FT.
                        # The Fourier transform is spin-conserving.
                        if spinless or (
                                (dual_basis_action[0][0] % 2 ==
                                 dual_basis_action[4][0] % 2 ==
                                 orbital1 % 2) and
                                (dual_basis_action[1][0] % 2 ==
                                 dual_basis_action[3][0] % 2 ==
                                 orbital2 % 2) and
                                (dual_basis_action[2][0] % 2 ==
                                 dual_basis_action[5][0] % 2 ==
                                 orbital3 % 2)):
                            expectation_value += numpy.exp(-1j * (
                                k1ae + k2bd + k3cf))

                        # Handle \delta_{af} \delta_{be} \delta_{cd} after FT.
                        # The Fourier transform is spin-conserving.
                        if spinless or (
                                (dual_basis_action[0][0] % 2 ==
                                 dual_basis_action[5][0] % 2 ==
                                 orbital1 % 2) and
                                (dual_basis_action[1][0] % 2 ==
                                 dual_basis_action[4][0] % 2 ==
                                 orbital2 % 2) and
                                (dual_basis_action[2][0] % 2 ==
                                 dual_basis_action[3][0] % 2 ==
                                 orbital3 % 2)):
                            expectation_value += numpy.exp(-1j * (
                                k1af + k2be + k3cd))

                        # Handle -\delta_{af} \delta_{bd} \delta_{ce} after FT.
                        # The Fourier transform is spin-conserving.
                        if spinless or (
                                (dual_basis_action[0][0] % 2 ==
                                 dual_basis_action[5][0] % 2 ==
                                 orbital1 % 2) and
                                (dual_basis_action[1][0] % 2 ==
                                 dual_basis_action[3][0] % 2 ==
                                 orbital2 % 2) and
                                (dual_basis_action[2][0] % 2 ==
                                 dual_basis_action[4][0] % 2 ==
                                 orbital3 % 2)):
                            expectation_value -= numpy.exp(-1j * (
                                k1af + k2bd + k3ce))

    return expectation_value


def get_gap(sparse_operator):
    """Compute gap between lowest eigenvalue and first excited state.

    Returns: A real float giving eigenvalue gap.
    """
    if not is_hermitian(sparse_operator):
        raise ValueError('sparse_operator must be Hermitian.')

    values, _ = scipy.sparse.linalg.eigsh(
        sparse_operator, 2, which='SA', maxiter=1e7)

    gap = abs(values[1] - values[0])
    return gap
