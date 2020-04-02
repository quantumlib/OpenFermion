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
import itertools
from functools import reduce
import numpy.linalg
import numpy

import scipy
import scipy.sparse
import scipy.sparse.linalg

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import (FermionOperator, QuadraticHamiltonian,
                             QubitOperator, BosonOperator,
                             QuadOperator, up_index, down_index)
from openfermion.utils import (count_qubits, gaussian_state_preparation_circuit,
                               is_hermitian,
                               slater_determinant_preparation_circuit)


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
    r"""Make a matrix representation of a fermion ladder operator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Args:
        index: This is a nonzero integer. The integer indicates the tensor
            factor and the sign indicates raising or lowering.
        n_qubits(int): Number qubits in the system Hilbert space.

    Returns:
        The corresponding Scipy sparse matrix.
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
    r"""Initialize a Scipy sparse matrix from a FermionOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Args:
        fermion_operator(FermionOperator): instance of the FermionOperator
            class.
        n_qubits(int): Number of qubits.

    Returns:
        The corresponding Scipy sparse matrix.
    """
    if n_qubits is None:
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

    # Construct the Scipy sparse matrix.
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
    """Initialize a Scipy sparse matrix from a QubitOperator.

    Args:
        qubit_operator(QubitOperator): instance of the QubitOperator class.
        n_qubits (int): Number of qubits.

    Returns:
        The corresponding Scipy sparse matrix.
    """
    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits < count_qubits(qubit_operator):
        raise ValueError('Invalid number of qubits specified.')

    # Construct the Scipy sparse matrix.
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

def get_linear_qubit_operator_diagonal(qubit_operator, n_qubits=None):
    """ Return a linear operator's diagonal elements.

    The main motivation is to use it for Davidson's algorithm, to find out the
    lowest n eigenvalues and associated eigenvectors.

    Qubit terms with X or Y operators will contribute nothing to the diagonal
    elements, while I or Z will contribute a factor of 1 or -1 together with
    the coefficient.

    Args:
        qubit_operator(QubitOperator): A qubit operator.

    Returns:
        linear_operator_diagonal(numpy.ndarray): The diagonal elements for
            LinearQubitOperator(qubit_operator).
    """
    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits < count_qubits(qubit_operator):
        raise ValueError('Invalid number of qubits specified.')

    n_hilbert = 2 ** n_qubits
    zeros_diagonal = numpy.zeros(n_hilbert)
    ones_diagonal = numpy.ones(n_hilbert)
    linear_operator_diagonal = zeros_diagonal
    # Loop through the terms.
    for qubit_term in qubit_operator.terms:
        is_zero = False
        tensor_factor = 0
        vecs = [ones_diagonal]
        for pauli_operator in qubit_term:
            op = pauli_operator[1]
            if op in ['X', 'Y']:
                is_zero = True
                break

            # Split vector by half and half for each bit.
            if pauli_operator[0] > tensor_factor:
                vecs = [v for iter_v in vecs for v in numpy.split(
                    iter_v, 2 ** (pauli_operator[0] - tensor_factor))]

            vec_pairs = [numpy.split(v, 2) for v in vecs]
            vecs = [v for vp in vec_pairs for v in (vp[0], -vp[1])]
            tensor_factor = pauli_operator[0] + 1
        if not is_zero:
            linear_operator_diagonal += (qubit_operator.terms[qubit_term] *
                                         numpy.concatenate(vecs))
    return linear_operator_diagonal


def jw_configuration_state(occupied_orbitals, n_qubits):
    """Function to produce a basis state in the occupation number basis.

    Args:
        occupied_orbitals(list): A list of integers representing the indices
            of the occupied orbitals in the desired basis state
        n_qubits(int): The total number of qubits

    Returns:
        basis_vector(sparse): The basis state as a sparse matrix
    """
    one_index = sum(2 ** (n_qubits - 1 - i) for i in occupied_orbitals)
    basis_vector = numpy.zeros(2 ** n_qubits, dtype=float)
    basis_vector[one_index] = 1
    return basis_vector


def jw_hartree_fock_state(n_electrons, n_orbitals):
    """Function to produce Hartree-Fock state in JW representation."""
    hartree_fock_state = jw_configuration_state(range(n_electrons),
                                                n_orbitals)
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
    indices = [sum([2 ** n for n in occupation])
               for occupation in occupations]
    return indices


def jw_sz_indices(sz_value, n_qubits, n_electrons=None,
                  up_index=up_index, down_index=down_index):
    r"""Return the indices of basis vectors with fixed Sz under JW encoding.

    The returned indices label computational basis vectors which lie within
    the corresponding eigenspace of the Sz operator,

    .. math::
        \begin{align}
        S^{z} = \frac{1}{2}\sum_{i = 1}^{n}(n_{i, \alpha} - n_{i, \beta})
        \end{align}

    Args:
        sz_value(float): Desired Sz value. Should be an integer or
            half-integer.
        n_qubits(int): Number of qubits defining the total state
        n_electrons(int, optional): Number of particles to restrict the
            operator to, if such a restriction is desired
        up_index (Callable, optional): Function that maps a spatial index
            to the index of the corresponding up site
        down_index (Callable, optional): Function that maps a spatial index
            to the index of the corresponding down site

    Returns:
        indices(list): The list of indices
    """
    if n_qubits % 2 != 0:
        raise ValueError('Number of qubits must be even')

    if not (2. * sz_value).is_integer():
        raise ValueError('Sz value must be an integer or half-integer')

    n_sites = n_qubits // 2
    sz_integer = int(2. * sz_value)
    indices = []

    if n_electrons is not None:
        # Particle number is fixed, so the number of spin-up electrons
        # (as well as the number of spin-down electrons) is fixed
        if ((n_electrons + sz_integer) % 2 != 0 or
                n_electrons < abs(sz_integer)):
            raise ValueError('The specified particle number and sz value are '
                             'incompatible.')
        num_up = (n_electrons + sz_integer) // 2
        num_down = n_electrons - num_up
        up_occupations = itertools.combinations(range(n_sites), num_up)
        down_occupations = list(
            itertools.combinations(range(n_sites), num_down))
        # Each arrangement of up spins can be paired with an arrangement
        # of down spins
        for up_occupation in up_occupations:
            up_occupation = [up_index(index) for index in up_occupation]
            for down_occupation in down_occupations:
                down_occupation = [down_index(index)
                                   for index in down_occupation]
                occupation = up_occupation + down_occupation
                indices.append(sum(2 ** (n_qubits - 1 - k)
                               for k in occupation))
    else:
        # Particle number is not fixed
        if sz_integer < 0:
            # There are more down spins than up spins
            more_map = down_index
            less_map = up_index
        else:
            # There are at least as many up spins as down spins
            more_map = up_index
            less_map = down_index
        for n in range(abs(sz_integer), n_sites + 1):
            # Choose n of the 'more' spin and n - abs(sz_integer) of the
            # 'less' spin
            more_occupations = itertools.combinations(range(n_sites), n)
            less_occupations = list(itertools.combinations(
                                    range(n_sites), n - abs(sz_integer)))
            # Each arrangement of the 'more' spins can be paired with an
            # arrangement of the 'less' spin
            for more_occupation in more_occupations:
                more_occupation = [more_map(index)
                                   for index in more_occupation]
                for less_occupation in less_occupations:
                    less_occupation = [less_map(index)
                                       for index in less_occupation]
                    occupation = more_occupation + less_occupation
                    indices.append(sum(2 ** (n_qubits - 1 - k)
                                   for k in occupation))

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


def jw_sz_restrict_operator(operator, sz_value,
                            n_electrons=None,
                            n_qubits=None,
                            up_index=up_index,
                            down_index=down_index):
    """Restrict a Jordan-Wigner encoded operator to a given Sz value

    Args:
        operator(ndarray or sparse): Numpy operator acting on
            the space of n_qubits.
        sz_value(float): Desired Sz value. Should be an integer or
            half-integer.
        n_electrons(int, optional): Number of particles to restrict the
            operator to, if such a restriction is desired.
        n_qubits(int, optional): Number of qubits defining the total state
        up_index (Callable, optional): Function that maps a spatial index
            to the index of the corresponding up site
        down_index (Callable, optional): Function that maps a spatial index
            to the index of the corresponding down site

    Returns:
        new_operator(ndarray or sparse): Numpy operator restricted to
            acting on states with the desired Sz value.
    """
    if n_qubits is None:
        n_qubits = int(numpy.log2(operator.shape[0]))

    select_indices = jw_sz_indices(
            sz_value, n_qubits,
            n_electrons=n_electrons,
            up_index=up_index,
            down_index=down_index)
    return operator[numpy.ix_(select_indices, select_indices)]


def jw_number_restrict_state(state, n_electrons, n_qubits=None):
    """Restrict a Jordan-Wigner encoded state to a given particle number

    Args:
        state(ndarray or sparse): Numpy vector in
            the space of n_qubits.
        n_electrons(int): Number of particles to restrict the state to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        new_operator(ndarray or sparse): Numpy vector restricted to
            states with the same particle number. May not be normalized.
    """
    if n_qubits is None:
        n_qubits = int(numpy.log2(state.shape[0]))

    select_indices = jw_number_indices(n_electrons, n_qubits)
    return state[select_indices]


def jw_sz_restrict_state(state, sz_value,
                         n_electrons=None,
                         n_qubits=None,
                         up_index=up_index,
                         down_index=down_index):
    """Restrict a Jordan-Wigner encoded state to a given Sz value

    Args:
        state(ndarray or sparse): Numpy vector in
            the space of n_qubits.
        sz_value(float): Desired Sz value. Should be an integer or
            half-integer.
        n_electrons(int, optional): Number of particles to restrict the
            operator to, if such a restriction is desired.
        n_qubits(int, optional): Number of qubits defining the total state
        up_index (Callable, optional): Function that maps a spatial index
            to the index of the corresponding up site
        down_index (Callable, optional): Function that maps a spatial index
            to the index of the corresponding down site

    Returns:
        new_operator(ndarray or sparse): Numpy vector restricted to
            states with the desired Sz value. May not be normalized.
    """
    if n_qubits is None:
        n_qubits = int(numpy.log2(state.shape[0]))

    select_indices = jw_sz_indices(
            sz_value, n_qubits,
            n_electrons=n_electrons,
            up_index=up_index,
            down_index=down_index)
    return state[select_indices]


def jw_get_ground_state_at_particle_number(sparse_operator, particle_number):
    """Compute ground energy and state at a specified particle number.

    Assumes the Jordan-Wigner transform. The input operator should be Hermitian
    and particle-number-conserving.

    Args:
        sparse_operator(sparse): A Jordan-Wigner encoded sparse matrix.
        particle_number(int): The particle number at which to compute the ground
            energy and states

    Returns:
        ground_energy(float): The lowest eigenvalue of sparse_operator within
            the eigenspace of the number operator corresponding to
            particle_number.
        ground_state(ndarray): The ground state at the particle number
    """

    n_qubits = int(numpy.log2(sparse_operator.shape[0]))

    # Get the operator restricted to the subspace of the desired particle number
    restricted_operator = jw_number_restrict_operator(sparse_operator,
                                                      particle_number,
                                                      n_qubits)

    # Compute eigenvalues and eigenvectors
    if restricted_operator.shape[0] - 1 <= 1:
        # Restricted operator too small for sparse eigensolver
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = numpy.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator,
                                                     k=1,
                                                     which='SA')


    # Expand the state
    state = eigvecs[:, 0]
    expanded_state = numpy.zeros(2 ** n_qubits, dtype=complex)
    expanded_state[jw_number_indices(particle_number, n_qubits)] = state

    return eigvals[0], expanded_state


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
    state = jw_configuration_state(start_orbitals, n_qubits)

    # Apply the circuit
    if not quadratic_hamiltonian.conserves_particle_number:
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


def jw_slater_determinant(slater_determinant_matrix):
    r"""Obtain a Slater determinant.

    The input is an :math:`N_f \times N` matrix :math:`Q` with orthonormal
    rows. Such a matrix describes the Slater determinant

    .. math::

        b^\dagger_1 \cdots b^\dagger_{N_f} \lvert \text{vac} \rangle,

    where

    .. math::

        b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k.

    Args:
        slater_determinant_matrix: The matrix :math:`Q` which describes the
            Slater determinant to be prepared.
    Returns:
        The Slater determinant as a sparse matrix.
    """
    circuit_description = slater_determinant_preparation_circuit(
        slater_determinant_matrix)
    start_orbitals = range(slater_determinant_matrix.shape[0])
    n_qubits = slater_determinant_matrix.shape[1]

    # Initialize the starting state
    state = jw_configuration_state(start_orbitals, n_qubits)

    # Apply the circuit
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            i, j, theta, phi = op
            state = jw_sparse_givens_rotation(
                i, j, theta, phi, n_qubits).dot(state)

    return state


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
        state = scipy.sparse.csc_matrix(state.reshape((len(state), 1)))
        density_matrix = density_matrix + probability * state * state.getH()
    return density_matrix


def get_ground_state(sparse_operator, initial_guess=None):
    """Compute lowest eigenvalue and eigenstate.

    Args:
        sparse_operator (LinearOperator): Operator to find the ground state of.
        initial_guess (ndarray): Initial guess for ground state.  A good
            guess dramatically reduces the cost required to converge.

    Returns
    -------
        eigenvalue:
            The lowest eigenvalue, a float.
        eigenstate:
            The lowest eigenstate in scipy.sparse csc format.
    """
    values, vectors = scipy.sparse.linalg.eigsh(
        sparse_operator, k=1, v0=initial_guess, which='SA', maxiter=1e7)

    order = numpy.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eigenvalue = values[0]
    eigenstate = vectors[:, 0]
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


def expectation(operator, state):
    """Compute the expectation value of an operator with a state.

    Args:
        operator(scipy.sparse.spmatrix or scipy.sparse.linalg.LinearOperator):
            The operator whose expectation value is desired.
        state(numpy.ndarray or scipy.sparse.spmatrix): A numpy array
            representing a pure state or a sparse matrix representing a density
            matrix. If `operator` is a LinearOperator, then this must be a
            numpy array.

    Returns:
        A complex number giving the expectation value.

    Raises:
        ValueError: Input state has invalid format.
    """

    if isinstance(state, scipy.sparse.spmatrix):
        # Handle density matrix.
        if isinstance(operator, scipy.sparse.linalg.LinearOperator):
            raise ValueError('Taking the expectation of a LinearOperator with '
                             'a density matrix is not supported.')
        product = state * operator
        expectation = numpy.sum(product.diagonal())

    elif isinstance(state, numpy.ndarray):
        # Handle state vector.
        if len(state.shape) == 1:
            # Row vector
            expectation = numpy.dot(numpy.conjugate(state), operator * state)
        else:
            # Column vector
            expectation = numpy.dot(numpy.conjugate(state.T),
                                    operator * state)[0, 0]

    else:
        # Handle exception.
        raise ValueError(
                'Input state must be a numpy array or a sparse matrix.')

    # Return.
    return expectation


def variance(operator, state):
    """Compute variance of operator with a state.

    Args:
        operator(scipy.sparse.spmatrix or scipy.sparse.linalg.LinearOperator):
            The operator whose expectation value is desired.
        state(numpy.ndarray or scipy.sparse.spmatrix): A numpy array
            representing a pure state or a sparse matrix representing a density
            matrix.

    Returns:
        A complex number giving the variance.

    Raises:
        ValueError: Input state has invalid format.
    """
    return (expectation(operator ** 2, state) -
            expectation(operator, state) ** 2)


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

    for single_action, coefficient in operator.terms.items():
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

    r_p = grid.position_vector(grid.grid_indices(dual_basis_action[0][0],
                                                 spinless))
    r_q = grid.position_vector(grid.grid_indices(dual_basis_action[1][0],
                                                 spinless))

    for orbital in plane_wave_occ_orbitals:
        # If there's spin, p and q have to have the same parity (spin),
        # and the new orbital has to have the same spin as these.
        k_orbital = grid.momentum_vector(grid.grid_indices(orbital,
                                                           spinless))
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
        r[i] = grid.position_vector(grid.grid_indices(dual_basis_action[i][0],
                                                      spinless))

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
        k = grid.momentum_vector(grid.grid_indices(o, spinless))
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
        r[i] = grid.position_vector(grid.grid_indices(dual_basis_action[i][0],
                                                      spinless))

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
        k = grid.momentum_vector(grid.grid_indices(o, spinless))
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


def get_gap(sparse_operator, initial_guess=None):
    """Compute gap between lowest eigenvalue and first excited state.

    Args:
        sparse_operator (LinearOperator): Operator to find the ground state of.
        initial_guess (ndarray): Initial guess for eigenspace.  A good
            guess dramatically reduces the cost required to converge.
    Returns: A real float giving eigenvalue gap.
    """
    if not is_hermitian(sparse_operator):
        raise ValueError('sparse_operator must be Hermitian.')

    values, _ = scipy.sparse.linalg.eigsh(
        sparse_operator, k=2, v0=initial_guess, which='SA', maxiter=1e7)

    gap = abs(values[1] - values[0])
    return gap


def inner_product(state_1, state_2):
    """Compute inner product of two states."""
    return numpy.dot(state_1.conjugate(), state_2)


def boson_ladder_sparse(n_modes, mode, ladder_type, trunc):
    r"""Make a matrix representation of a singular bosonic ladder operator
    in the Fock space.

    Since the bosonic operator lies in an infinite Fock space,
    a truncation value needs to be provide so that a sparse matrix
    of finite size can be returned.

    Args:
        n_modes (int): Number of modes in the system Hilbert space.
        mode (int): The mode the ladder operator targets.
        ladder_type (int): This is a nonzero integer. 0 indicates a lowering
            operator, 1 a raising operator.
        trunc (int): The size at which the Fock space should be truncated
            when returning the matrix representing the ladder operator.

    Returns:
        The corresponding trunc x trunc Scipy sparse matrix.
    """
    if trunc < 1 or not isinstance(trunc, int):
        raise ValueError("Fock space truncation must be a positive integer.")

    if ladder_type:
        lop = scipy.sparse.spdiags(numpy.sqrt(range(1, trunc)),
                                   -1, trunc, trunc, format='csc')
    else:
        lop = scipy.sparse.spdiags(numpy.sqrt(range(trunc)),
                                   1, trunc, trunc, format='csc')

    Id = [scipy.sparse.identity(trunc, format='csc', dtype=complex)]
    operator_list = Id*mode + [lop] + Id*(n_modes - mode - 1)
    operator = kronecker_operators(operator_list)

    return operator


def single_quad_op_sparse(n_modes, mode, quadrature, hbar, trunc):
    r"""Make a matrix representation of a singular quadrature
    operator in the Fock space.

    Since the bosonic operators lie in an infinite Fock space,
    a truncation value needs to be provide so that a sparse matrix
    of finite size can be returned.

    Args:
        n_modes (int): Number of modes in the system Hilbert space.
        mode (int): The mode the ladder operator targets.
        quadrature (str): 'q' for the canonical position operator,
            'p' for the canonical moment]um operator.
        hbar (float): the value of hbar to use in the definition of the
            canonical commutation relation [q_i, p_j] = \delta_{ij} i hbar.
        trunc (int): The size at which the Fock space should be truncated
            when returning the matrix representing the ladder operator.

    Returns:
        The corresponding trunc x trunc Scipy sparse matrix.
    """
    if trunc < 1 or not isinstance(trunc, int):
        raise ValueError("Fock space truncation must be a positive integer.")

    b = boson_ladder_sparse(1, 0, 0, trunc)

    if quadrature == 'q':
        op = numpy.sqrt(hbar/2) * (b + b.conj().T)
    elif quadrature == 'p':
        op = -1j*numpy.sqrt(hbar/2) * (b - b.conj().T)

    Id = [scipy.sparse.identity(trunc, dtype=complex, format='csc')]
    operator_list = Id*mode + [op] + Id*(n_modes - mode - 1)
    operator = kronecker_operators(operator_list)

    return operator


def boson_operator_sparse(operator, trunc, hbar=1.):
    r"""Initialize a Scipy sparse matrix in the Fock space
    from a bosonic operator.

    Since the bosonic operators lie in an infinite Fock space,
    a truncation value needs to be provide so that a sparse matrix
    of finite size can be returned.

    Args:
        operator: One of either BosonOperator or QuadOperator.
        trunc (int): The size at which the Fock space should be truncated
            when returning the matrix representing the ladder operator.
        hbar (float): the value of hbar to use in the definition of the
            canonical commutation relation [q_i, p_j] = \delta_{ij} i hbar.
            This only applies if calcualating the sparse representation of
            a quadrature operator.

    Returns:
        The corresponding Scipy sparse matrix of size [trunc, trunc].
    """
    if isinstance(operator, QuadOperator):
        from openfermion.transforms._conversion import get_boson_operator
        boson_operator = get_boson_operator(operator, hbar)
    elif isinstance(operator, BosonOperator):
        boson_operator = operator
    else:
        raise ValueError("Only BosonOperator and QuadOperator are supported.")

    if trunc < 1 or not isinstance(trunc, int):
        raise ValueError("Fock space truncation must be a positive integer.")

    # count the number of modes
    n_modes = 0
    for term in boson_operator.terms:
        for ladder_operator in term:
            if ladder_operator[0] + 1 > n_modes:
                n_modes = ladder_operator[0] + 1

    # Construct the Scipy sparse matrix.
    n_hilbert = trunc ** n_modes
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]

    # Loop through the terms.
    for term in boson_operator.terms:
        coefficient = boson_operator.terms[term]
        term_operator = coefficient*scipy.sparse.identity(
            n_hilbert, dtype=complex, format='csc')

        for ladder_op in term:
            # Add actual operator to the list.
            b = boson_ladder_sparse(n_modes, ladder_op[0], ladder_op[1], trunc)
            term_operator = term_operator.dot(b)

        # Extract triplets from sparse_term.
        values_list.append(term_operator.tocoo(copy=False).data)
        (row, column) = term_operator.nonzero()
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
