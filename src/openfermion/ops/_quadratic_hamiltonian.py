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

"""Hamiltonians that are quadratic in the fermionic ladder operators."""
from __future__ import absolute_import

import numpy
from scipy.linalg import schur

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import PolynomialTensor


class QuadraticHamiltonianError(Exception):
    pass


class QuadraticHamiltonian(PolynomialTensor):
    """Class for storing Hamiltonians that are quadratic in the fermionic
    ladder operators. The operators stored in this class take the form

    .. math::

        \sum_{p, q} (M_{pq} - \mu \delta_{pq}) a^\dagger_p a_q
        + \\frac12 \sum_{p, q}
            (\\Delta_{pq} a^\dagger_p a^\dagger_q + \\text{h.c.})
        + \\text{constant}

    where

        - :math:`M` is a Hermitian `n_qubits` x `n_qubits` matrix.
        - :math:`\\Delta` is an antisymmetric `n_qubits` x `n_qubits` matrix.
        - :math:`\mu` is a real number representing the chemical potential.
        - :math:`\delta_{pq}` is the Kronecker delta symbol.

    We separate the chemical potential :math:`\mu` from :math:`M` so that
    we can use it to adjust the expectation value of the total number of
    particles.

    Attributes:
        chemical_potential(float): The chemical potential :math:`\mu`.
    """

    def __init__(self, hermitian_part, antisymmetric_part=None,
                 constant=0., chemical_potential=0.):
        """
        Initialize the QuadraticHamiltonian class.

        Args:
            hermitian_part(ndarray): The matrix :math:`M`, which represents the
                coefficients of the particle-number-conserving terms.
                This is an `n_qubits` x `n_qubits` numpy array of complex
                numbers.
            antisymmetric_part(ndarray): The matrix :math:`\\Delta`,
                which represents the coefficients of the
                non-particle-number-conserving terms.
                This is an `n_qubits` x `n_qubits` numpy array of complex
                numbers.
            constant(float, optional): A constant term in the operator.
            chemical_potential(float, optional): The chemical potential
                :math:`\mu`.
        """
        n_qubits = hermitian_part.shape[0]

        # Initialize combined Hermitian part
        if not chemical_potential:
            combined_hermitian_part = hermitian_part
        else:
            combined_hermitian_part = (
                hermitian_part - chemical_potential * numpy.eye(n_qubits))

        # Initialize the PolynomialTensor
        if antisymmetric_part is None:
            super(QuadraticHamiltonian, self).__init__(
                {(): constant, (1, 0): combined_hermitian_part})
        else:
            super(QuadraticHamiltonian, self).__init__(
                {(): constant, (1, 0): combined_hermitian_part,
                 (1, 1): 0.5 * antisymmetric_part,
                 (0, 0): -0.5 * antisymmetric_part.conj()})

        # Add remaining attributes
        self.chemical_potential = chemical_potential

    @property
    def combined_hermitian_part(self):
        """The Hermitian part including the chemical potential."""
        return self.n_body_tensors[1, 0]

    @property
    def antisymmetric_part(self):
        """The antisymmetric part."""
        if (1, 1) in self.n_body_tensors:
            return 2. * self.n_body_tensors[1, 1]
        else:
            return numpy.zeros((self.n_qubits, self.n_qubits), complex)

    @property
    def hermitian_part(self):
        """The Hermitian part not including the chemical potential."""
        return (self.combined_hermitian_part +
                self.chemical_potential * numpy.eye(self.n_qubits))

    @property
    def conserves_particle_number(self):
        """Whether this Hamiltonian conserves particle number."""
        discrepancy = numpy.max(numpy.abs(self.antisymmetric_part))
        return discrepancy < EQ_TOLERANCE

    def add_chemical_potential(self, chemical_potential):
        """Increase (or decrease) the chemical potential by some value."""
        self.n_body_tensors[1, 0] -= (chemical_potential *
                                      numpy.eye(self.n_qubits))
        self.chemical_potential += chemical_potential

    def orbital_energies(self, non_negative=False):
        """Return the orbital energies.

        Any quadratic Hamiltonian is unitarily equivalent to a Hamiltonian
        of the form

        .. math::

            \sum_{j} \\varepsilon_j b^\dagger_j b_j + \\text{constant}.

        We call the :math:`\\varepsilon_j` the orbital energies.
        The eigenvalues of the Hamiltonian are sums of subsets of the
        orbital energies (up to the additive constant).

        Args:
            non_negative(bool): If True, always return a list of orbital
                energies that are non-negative. This option is ignored if
                the Hamiltonian does not conserve particle number, in which
                case the returned orbital energies are always non-negative.

        Returns
        -------
        orbital_energies(ndarray)
            A one-dimensional array containing the :math:`\\varepsilon_j`
        constant(float)
            The constant
        """
        if self.conserves_particle_number and not non_negative:
            hermitian_matrix = self.combined_hermitian_part
            orbital_energies, diagonalizing_unitary = numpy.linalg.eigh(
                hermitian_matrix)
            constant = self.constant
        else:
            majorana_matrix, majorana_constant = self.majorana_form()
            canonical, orthogonal = antisymmetric_canonical_form(
                majorana_matrix)
            orbital_energies = canonical[
                range(self.n_qubits), range(self.n_qubits, 2 * self.n_qubits)]
            constant = -0.5 * numpy.sum(orbital_energies) + majorana_constant

        return orbital_energies, constant

    def ground_energy(self):
        """Return the ground energy."""
        _, constant = self.orbital_energies(non_negative=True)
        return constant

    def majorana_form(self):
        """Return the Majorana represention of the Hamiltonian.

        Any quadratic Hamiltonian can be written in the form

        .. math::

            \\frac{i}{2} \sum_{j, k} A_{jk} f_j f_k + \\text{constant}

        where the :math:`f_i` are normalized Majorana fermion operators:

        .. math::

            f_j = \\frac{1}{\\sqrt{2}} (a^\dagger_j + a_j)

            f_{j + N} = \\frac{i}{\\sqrt{2}} (a^\dagger_j - a_j)

        and :math:`A` is a (2 * `n_qubits`) x (2 * `n_qubits`) real
        antisymmetric matrix. This function returns the matrix
        :math:`A` and the constant.
        """
        hermitian_part = self.combined_hermitian_part
        antisymmetric_part = self.antisymmetric_part

        # Compute the Majorana matrix using block matrix manipulations
        majorana_matrix = numpy.zeros((2 * self.n_qubits, 2 * self.n_qubits))
        # Set upper left block
        majorana_matrix[:self.n_qubits, :self.n_qubits] = numpy.real(-0.5j * (
            hermitian_part - hermitian_part.conj() +
            antisymmetric_part - antisymmetric_part.conj()))
        # Set upper right block
        majorana_matrix[:self.n_qubits, self.n_qubits:] = numpy.real(0.5 * (
            hermitian_part + hermitian_part.conj() -
            antisymmetric_part - antisymmetric_part.conj()))
        # Set lower left block
        majorana_matrix[self.n_qubits:, :self.n_qubits] = numpy.real(-0.5 * (
            hermitian_part + hermitian_part.conj() +
            antisymmetric_part + antisymmetric_part.conj()))
        # Set lower right block
        majorana_matrix[self.n_qubits:, self.n_qubits:] = numpy.real(-0.5j * (
            hermitian_part - hermitian_part.conj() -
            antisymmetric_part + antisymmetric_part.conj()))

        # Compute the constant
        majorana_constant = (0.5 * numpy.real(numpy.trace(hermitian_part)) +
                             self.n_body_tensors[()])

        return majorana_matrix, majorana_constant

    def diagonalizing_bogoliubov_transform(self):
        """Compute the unitary that diagonalizes a quadratic Hamiltonian.

        Any quadratic Hamiltonian can be rewritten in the form

        .. math::

            \sum_{j} \\varepsilon_j b^\dagger_j b_j + \\text{constant},

        where the :math:`b_j` are a new set fermionic operators
        that satisfy the canonical anticommutation relations.
        The new fermionic operators are linear combinations of the
        original ones. In the most general case, creation and annihilation
        operators are mixed together:

        .. math::

           \\begin{pmatrix}
                b^\dagger_1 \\\\
                \\vdots \\\\
                b^\dagger_N \\\\
                b_1 \\\\
                \\vdots \\\\
                b_N
           \\end{pmatrix}
           = W
           \\begin{pmatrix}
                a^\dagger_1 \\\\
                \\vdots \\\\
                a^\dagger_N \\\\
                a_1 \\\\
                \\vdots \\\\
                a_N
           \\end{pmatrix},

        where :math:`W` is a :math:`2N \\times 2N` unitary matrix.
        However, if the Hamiltonian conserves particle number then
        creation operators don't need to be mixed with annihilation operators
        and :math:`W` only needs to be an :math:`N \\times N` matrix:

        .. math::

           \\begin{pmatrix}
                b^\dagger_1 \\\\
                \\vdots \\\\
                b^\dagger_N \\\\
           \\end{pmatrix}
           = W
           \\begin{pmatrix}
                a^\dagger_1 \\\\
                \\vdots \\\\
                a^\dagger_N \\\\
           \\end{pmatrix},

        This method returns the matrix :math:`W`.

        Returns:
            diagonalizing_unitary (ndarray):
                A matrix representing the transformation :math:`W` of the
                fermionic ladder operators. If the Hamiltonian conserves
                particle number then this is :math:`N \\times N`; otherwise
                it is :math:`2N \\times 2N`.
        """
        if self.conserves_particle_number:
            energies, diagonalizing_unitary_T = numpy.linalg.eigh(
                    self.combined_hermitian_part)
            return diagonalizing_unitary_T.T
        else:
            majorana_matrix, majorana_constant = self.majorana_form()

            # Get the orthogonal transformation that puts majorana_matrix
            # into canonical form
            canonical, orthogonal = antisymmetric_canonical_form(
                    majorana_matrix)

            # Create the matrix that converts between fermionic ladder and
            # Majorana bases
            normalized_identity = (numpy.eye(self.n_qubits, dtype=complex) /
                                   numpy.sqrt(2.))
            majorana_basis_change = numpy.eye(
                2 * self.n_qubits, dtype=complex) / numpy.sqrt(2.)
            majorana_basis_change[self.n_qubits:, self.n_qubits:] *= -1.j
            majorana_basis_change[:self.n_qubits,
                                  self.n_qubits:] = normalized_identity
            majorana_basis_change[self.n_qubits:,
                                  :self.n_qubits] = 1.j * normalized_identity

            # Compute the unitary and return
            diagonalizing_unitary = majorana_basis_change.T.conj().dot(
                orthogonal.dot(majorana_basis_change))

            return diagonalizing_unitary

    def diagonalizing_circuit(self):
        """Get a circuit for a unitary that diagonalizes this Hamiltonian

        This circuit performs the transformation to a basis in which the
        Hamiltonian takes the diagonal form

        .. math::

            \sum_{j} \\varepsilon_j b^\dagger_j b_j + \\text{constant}.

        Returns
        -------
            circuit_description (list[tuple]):
                A list of operations describing the circuit. Each operation
                is a tuple of objects describing elementary operations that
                can be performed in parallel. Each elementary operation
                is either the string 'pht' indicating a particle-hole
                transformation on the last fermionic mode, a tuple of
                the form :math:`(i, j, \\theta, \\varphi)`,
                indicating a Givens rotation
                of modes :math:`i` and :math:`j` by angles :math:`\\theta`
                and :math:`\\varphi`, or a tuple of the form
                :math:`(j, \\varphi)`, indicating the operation
                :math:`e^{i \\varphi a^\dagger_j a_j}`.
        """
        diagonalizing_unitary = self.diagonalizing_bogoliubov_transform()

        if self.conserves_particle_number:
            # The Hamiltonian conserves particle number, so we don't need
            # to use the most general procedure.
            decomposition, diagonal = givens_decomposition_square(
                    diagonalizing_unitary)
            circuit_description = list(reversed(decomposition))
        else:
            # The Hamiltonian does not conserve particle number, so we
            # need to use the most general procedure.
            # Get the unitary rows which represent the Gaussian unitary
            gaussian_unitary_matrix = diagonalizing_unitary[self.n_qubits:]

            # Get the circuit description
            decomposition, left_decomposition, diagonal, left_diagonal = (
                fermionic_gaussian_decomposition(gaussian_unitary_matrix))
            # need to use left_diagonal too
            circuit_description = list(reversed(
                decomposition + left_decomposition))

        return circuit_description


def antisymmetric_canonical_form(antisymmetric_matrix):
    """Compute the canonical form of an antisymmetric matrix.

    The input is a real, antisymmetric n x n matrix A, where n is even.
    Its canonical form is::

        A = R^T C R

    where R is a real, orthogonal matrix and C has the form::

        [  0     D ]
        [ -D     0 ]

    where D is a diagonal matrix with nonnegative entries.

    Args:
        antisymmetric_matrix(ndarray): An antisymmetric matrix with even
            dimension.

    Returns:
        canonical(ndarray): The canonical form C of antisymmetric_matrix
        orthogonal(ndarray): The orthogonal transformation R.
    """
    m, p = antisymmetric_matrix.shape

    if m != p or p % 2 != 0:
        raise ValueError('The input matrix must be square with even '
                         'dimension.')

    # Check that input matrix is antisymmetric
    matrix_plus_transpose = antisymmetric_matrix + antisymmetric_matrix.T
    maxval = numpy.max(numpy.abs(matrix_plus_transpose))
    if maxval > EQ_TOLERANCE:
        raise ValueError('The input matrix must be antisymmetric.')

    # Compute Schur decomposition
    canonical, orthogonal = schur(antisymmetric_matrix, output='real')

    # The returned form is block diagonal; we need to permute rows and columns
    # to put it into the form we want
    n = p // 2
    for i in range(1, n, 2):
        swap_rows(canonical, i, n + i - 1)
        swap_columns(canonical, i, n + i - 1)
        swap_columns(orthogonal, i, n + i - 1)
        if n % 2 != 0:
            swap_rows(canonical, n - 1, n + i)
            swap_columns(canonical, n - 1, n + i)
            swap_columns(orthogonal, n - 1, n + i)

    # Now we permute so that the upper right block is non-negative
    for i in range(n):
        if canonical[i, n + i] < -EQ_TOLERANCE:
            swap_rows(canonical, i, n + i)
            swap_columns(canonical, i, n + i)
            swap_columns(orthogonal, i, n + i)

    # Now we permute so that the nonzero entries are ordered by magnitude
    # We use insertion sort
    diagonal = canonical[range(n), range(n, 2 * n)]
    for i in range(n):
        # Insert the smallest element from the unsorted part of the list into
        # index i
        arg_min = numpy.argmin(diagonal[i:]) + i
        if arg_min != i:
            # Permute the upper right block
            swap_rows(canonical, i, arg_min)
            swap_columns(canonical, n + i, n + arg_min)
            swap_columns(orthogonal, n + i, n + arg_min)
            # Permute the lower left block
            swap_rows(canonical, n + i, n + arg_min)
            swap_columns(canonical, i, arg_min)
            swap_columns(orthogonal, i, arg_min)
            # Update diagonal
            swap_rows(diagonal, i, arg_min)

    return canonical, orthogonal.T


def fermionic_gaussian_decomposition(unitary_rows):
    """Decompose a matrix into a sequence of Givens rotations and
    particle-hole transformations on the last fermionic mode.

    The input is an :math:`N \\times 2N` matrix :math:`W` with orthonormal
    rows. Furthermore, :math:`W` must have the block form

    .. math::

        W = ( W_1 \hspace{4pt} W_2 )

    where :math:`W_1` and :math:`W_2` satisfy

    .. math::

        W_1  W_1^\dagger + W_2  W_2^\dagger &= I

        W_1  W_2^T + W_2  W_1^T &= 0.

    Then :math:`W` can be decomposed as

    .. math::

        V  W  U^\dagger = ( 0 \hspace{6pt} D )

    where :math:`V` and :math:`U` are unitary matrices and :math:`D`
    is a diagonal unitary matrix. Furthermore, :math:`U` can be decomposed
    as follows:

    .. math::

        U = B G_{k} \cdots B G_3 G_2 B G_1 B,

    where each :math:`G_i` is a Givens rotation, and :math:`B` represents
    swapping the :math:`N`-th column with the :math:`2N`-th column,
    which corresponds to a particle-hole transformation
    on the last fermionic mode. This particle-hole transformation maps
    :math:`a^\dagger_N` to :math:`a_N` and vice versa, while leaving the
    other fermionic ladder operators invariant.

    The decomposition of :math:`U` is returned as a list of tuples of objects
    describing rotations and particle-hole transformations. The list looks
    something like [('pht', ), (G_1, ), ('pht', G_2), ... ].
    The objects within a tuple are either the string 'pht', which indicates
    a particle-hole transformation on the last fermionic mode, or a tuple
    of the form :math:`(i, j, \\theta, \\varphi)`, which indicates a
    Givens rotation of rows :math:`i` and :math:`j` by angles
    :math:`\\theta` and :math:`\\varphi`.

    The matrix :math:`V^T D^*` can also be decomposed as a sequence of
    Givens rotations. This decomposition is needed for a circuit that
    prepares an excited state.

    Args:
        unitary_rows(ndarray): A matrix with orthonormal rows and
            additional structure described above.

    Returns
    -------
        decomposition (list[tuple]):
            The decomposition of :math:`U`.
        left_decomposition (list[tuple]):
            The decomposition of :math:`V^T D^*`.
        diagonal (ndarray):
            A list of the nonzero entries of :math:`D`.
        left_diagonal (ndarray):
            A list of the nonzero entries left from the decomposition
            of :math:`V^T D^*`.
    """
    current_matrix = numpy.copy(unitary_rows)
    n, p = current_matrix.shape

    # Check that p = 2 * n
    if p != 2 * n:
        raise ValueError('The input matrix must have twice as many columns '
                         'as rows.')

    # Check that left and right parts of unitary_rows satisfy the constraints
    # necessary for the transformed fermionic operators to satisfy
    # the fermionic anticommutation relations
    left_part = unitary_rows[:, :n]
    right_part = unitary_rows[:, n:]
    constraint_matrix_1 = (left_part.dot(left_part.T.conj()) +
                           right_part.dot(right_part.T.conj()))
    constraint_matrix_2 = (left_part.dot(right_part.T) +
                           right_part.dot(left_part.T))
    discrepancy_1 = numpy.amax(abs(constraint_matrix_1 - numpy.eye(n)))
    discrepancy_2 = numpy.amax(abs(constraint_matrix_2))
    if discrepancy_1 > EQ_TOLERANCE or discrepancy_2 > EQ_TOLERANCE:
        raise ValueError('The input matrix does not satisfy the constraints '
                         'necessary for a proper transformation of the '
                         'fermionic ladder operators.')

    # Compute left_unitary using Givens rotations
    left_unitary = numpy.eye(n, dtype=complex)
    for k in range(n - 1):
        # Zero out entries in column k
        for l in range(n - 1 - k):
            # Zero out entry in row l if needed
            if abs(current_matrix[l, k]) > EQ_TOLERANCE:
                givens_rotation = givens_matrix_elements(
                    current_matrix[l, k], current_matrix[l + 1, k])
                # Apply Givens rotation
                givens_rotate(current_matrix, givens_rotation, l, l + 1)
                givens_rotate(left_unitary, givens_rotation, l, l + 1)

    # Initialize list to store decomposition of current_matrix
    decomposition = []
    # There are 2 * n - 1 iterations (that is the circuit depth)
    for k in range(2 * n - 1):
        # Initialize the list of parallel operations to perform
        # in this iteration
        parallel_ops = []

        # Perform a particle-hole transformation if necessary
        if k % 2 == 0 and abs(current_matrix[k // 2, n - 1]) > EQ_TOLERANCE:
            parallel_ops.append('pht')
            swap_columns(current_matrix, n - 1, 2 * n - 1)

        # Get the (row, column) indices of elements to zero out in parallel.
        if k < n:
            end_row = k
            end_column = n - 1 - k
        else:
            end_row = n - 1
            end_column = k - (n - 1)
        column_indices = range(end_column, n - 1, 2)
        row_indices = range(end_row, end_row - len(column_indices), -1)
        indices_to_zero_out = zip(row_indices, column_indices)

        for i, j in indices_to_zero_out:
            # Compute the Givens rotation to zero out the (i, j) element,
            # if needed
            left_element = current_matrix[i, j].conj()
            if abs(left_element) > EQ_TOLERANCE:
                # We actually need to perform a Givens rotation
                right_element = current_matrix[i, j + 1].conj()
                givens_rotation = givens_matrix_elements(left_element,
                                                         right_element)

                # Add the parameters to the list
                theta = numpy.arcsin(numpy.real(givens_rotation[1, 0]))
                phi = numpy.angle(givens_rotation[1, 1])
                parallel_ops.append((j, j + 1, theta, phi))

                # Update the matrix
                double_givens_rotate(current_matrix, givens_rotation,
                                     j, j + 1, which='col')

        # If the current list of parallel operations is not empty,
        # append it to the list,
        if parallel_ops:
            decomposition.append(tuple(parallel_ops))

    # Get the diagonal entries
    diagonal = current_matrix[range(n), range(n, 2 * n)]

    # Compute the decomposition of left_unitary^T * diagonal^*
    current_matrix = left_unitary.T
    for k in range(n):
        current_matrix[:, k] *= diagonal[k].conj()

    left_decomposition, left_diagonal = givens_decomposition_square(
            current_matrix)

    return decomposition, left_decomposition, diagonal, left_diagonal


def givens_decomposition_square(unitary_matrix):
    """Decompose a square matrix into a sequence of Givens rotations.

    The input is a square :math:`n \\times n` matrix :math:`Q`.
    :math:`Q` can be decomposed as follows:

    .. math::

        Q = DU

    where :math:`U` is unitary and :math:`D` is diagonal.
    Furthermore, we can decompose :math:`U` as

    .. math::

        U = G_k ... G_1

    where :math:`G_1, \\ldots, G_k` are complex Givens rotations.
    A Givens rotation is a rotation within the two-dimensional subspace
    spanned by two coordinate axes. Within the two relevant coordinate
    axes, a Givens rotation has the form

    .. math::

        \\begin{pmatrix}
            \\cos(\\theta) & -e^{i \\varphi} \\sin(\\theta) \\\\
            \\sin(\\theta) &     e^{i \\varphi} \\cos(\\theta)
        \\end{pmatrix}.

    Args:
        unitary_matrix: A numpy array with orthonormal rows,
            representing the matrix Q.

    Returns
    -------
        decomposition (list[tuple]):
            A list of tuples of objects describing Givens
            rotations. The list looks like [(G_1, ), (G_2, G_3), ... ].
            The Givens rotations within a tuple can be implemented in parallel.
            The description of a Givens rotation is itself a tuple of the
            form :math:`(i, j, \\theta, \\varphi)`, which represents a
            Givens rotation of coordinates
            :math:`i` and :math:`j` by angles :math:`\\theta` and
            :math:`\\varphi`.
        diagonal (ndarray):
            A list of the nonzero entries of :math:`D`.
    """
    current_matrix = numpy.copy(unitary_matrix)
    n = current_matrix.shape[0]

    decomposition = []

    for k in range(2 * (n - 1) - 1):
        # Initialize the list of parallel operations to perform
        # in this iteration
        parallel_ops = []

        # Get the (row, column) indices of elements to zero out in parallel.
        if k < n - 1:
            start_row = 0
            start_column = n - 1 - k
        else:
            start_row = k - (n - 2)
            start_column = k - (n - 3)
        column_indices = range(start_column, n, 2)
        row_indices = range(start_row, start_row + len(column_indices))
        indices_to_zero_out = zip(row_indices, column_indices)

        for i, j in indices_to_zero_out:
            # Compute the Givens rotation to zero out the (i, j) element,
            # if needed
            right_element = current_matrix[i, j].conj()
            if abs(right_element) > EQ_TOLERANCE:
                # We actually need to perform a Givens rotation
                left_element = current_matrix[i, j - 1].conj()
                givens_rotation = givens_matrix_elements(left_element,
                                                         right_element,
                                                         which='right')

                # Add the parameters to the list
                theta = numpy.arcsin(numpy.real(givens_rotation[1, 0]))
                phi = numpy.angle(givens_rotation[1, 1])
                parallel_ops.append((j - 1, j, theta, phi))

                # Update the matrix
                givens_rotate(current_matrix, givens_rotation,
                              j - 1, j, which='col')

        # If the current list of parallel operations is not empty,
        # append it to the list,
        if parallel_ops:
            decomposition.append(tuple(parallel_ops))

    # Get the diagonal entries
    diagonal = current_matrix[range(n), range(n)]

    return decomposition, diagonal


def givens_matrix_elements(a, b, which='left'):
    """Compute the matrix elements of the Givens rotation that zeroes out one
    of two row entries.

    If `which='left'` then returns a matrix G such that

        G * [a  b]^T= [0  r]^T

    otherwise, returns a matrix G such that

        G * [a  b]^T= [r  0]^T

    where r is a complex number.

    Args:
        a(complex or float): A complex number representing the upper row entry
        b(complex or float): A complex number representing the lower row entry
        which(string): Either 'left' or 'right', indicating whether to
            zero out the left element (first argument) or right element
            (second argument). Default is `left`.
    Returns:
        G(ndarray): A 2 x 2 numpy array representing the matrix G.
            The numbers in the first column of G are real.
    """
    # Handle case that a is zero
    if abs(a) < EQ_TOLERANCE:
        cosine = 1.
        sine = 0.
        phase = 1.
    # Handle case that b is zero and a is nonzero
    elif abs(b) < EQ_TOLERANCE:
        cosine = 0.
        sine = 1.
        phase = 1.
    # Handle case that a and b are both nonzero
    else:
        denominator = numpy.sqrt(abs(a) ** 2 + abs(b) ** 2)
        cosine = abs(b) / denominator
        sine = abs(a) / denominator
        sign_b = b / abs(b)
        sign_a = a / abs(a)
        phase = sign_a * sign_b.conjugate()
        # If phase is a real number, convert it to a float
        if numpy.isreal(phase):
            phase = numpy.real(phase)

    # Construct matrix and return
    if which == 'left':
        # We want to zero out a
        if numpy.isreal(a) and numpy.isreal(b):
            givens_rotation = numpy.array([[cosine, -phase * sine],
                                           [phase * sine, cosine]])
        else:
            givens_rotation = numpy.array([[cosine, -phase * sine],
                                           [sine, phase * cosine]])
    elif which == 'right':
        # We want to zero out b
        if numpy.isreal(a) and numpy.isreal(b):
            givens_rotation = numpy.array([[sine, phase * cosine],
                                           [-phase * cosine, sine]])
        else:
            givens_rotation = numpy.array([[sine, phase * cosine],
                                           [cosine, -phase * sine]])
    else:
        raise ValueError('"which" must be equal to "left" or "right".')
    return givens_rotation


def givens_rotate(operator, givens_rotation, i, j, which='row'):
    """Apply a Givens rotation to coordinates i and j of an operator."""
    if which == 'row':
        # Rotate rows i and j
        row_i = operator[i].copy()
        row_j = operator[j].copy()
        operator[i] = (givens_rotation[0, 0] * row_i +
                       givens_rotation[0, 1] * row_j)
        operator[j] = (givens_rotation[1, 0] * row_i +
                       givens_rotation[1, 1] * row_j)
    elif which == 'col':
        # Rotate columns i and j
        col_i = operator[:, i].copy()
        col_j = operator[:, j].copy()
        operator[:, i] = (givens_rotation[0, 0] * col_i +
                          givens_rotation[0, 1].conj() * col_j)
        operator[:, j] = (givens_rotation[1, 0] * col_i +
                          givens_rotation[1, 1].conj() * col_j)
    else:
        raise ValueError('"which" must be equal to "row" or "col".')


def double_givens_rotate(operator, givens_rotation, i, j, which='row'):
    """Apply a double Givens rotation.

    Applies a Givens rotation to coordinates i and j and the the conjugate
    Givens rotation to coordinates n + i and n + j, where
    n = dim(operator) / 2. dim(operator) must be even.
    """
    m, p = operator.shape

    if which == 'row':
        if m % 2 != 0:
            raise ValueError('To apply a double Givens rotation on rows, '
                             'the number of rows must be even.')
        n = m // 2
        # Rotate rows i and j
        givens_rotate(operator[:n], givens_rotation, i, j, which='row')
        # Rotate rows n + i and n + j
        givens_rotate(operator[n:], givens_rotation.conj(), i, j, which='row')
    elif which == 'col':
        if p % 2 != 0:
            raise ValueError('To apply a double Givens rotation on columns, '
                             'the number of columns must be even.')
        n = p // 2
        # Rotate columns i and j
        givens_rotate(operator[:, :n], givens_rotation, i, j, which='col')
        # Rotate cols n + i and n + j
        givens_rotate(operator[:, n:], givens_rotation.conj(), i, j,
                      which='col')
    else:
        raise ValueError('"which" must be equal to "row" or "col".')


def swap_rows(M, i, j):
    """Swap rows i and j of matrix M."""
    if len(M.shape) == 1:
        M[i], M[j] = M[j], M[i]
    else:
        row_i = M[i, :].copy()
        row_j = M[j, :].copy()
        M[i, :], M[j, :] = row_j, row_i


def swap_columns(M, i, j):
    """Swap columns i and j of matrix M."""
    if len(M.shape) == 1:
        M[i], M[j] = M[j], M[i]
    else:
        column_i = M[:, i].copy()
        column_j = M[:, j].copy()
        M[:, i], M[:, j] = column_j, column_i
