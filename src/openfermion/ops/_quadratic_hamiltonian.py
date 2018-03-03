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

"""Class and functions to store and manipulate Hamiltonians that are quadratic
in the fermionic ladder operators."""
from __future__ import absolute_import
from scipy.linalg import schur

import numpy

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

    def __init__(self, constant, hermitian_part,
                 antisymmetric_part=None, chemical_potential=0.):
        """
        Initialize the QuadraticHamiltonian class.

        Args:
            constant(float): A constant term in the operator.
            hermitian_part(ndarray): The matrix :math:`M`, which represents the
                coefficients of the particle-number-conserving terms.
                This is an `n_qubits` x `n_qubits` numpy array of complex
                numbers.
            antisymmetric_part(ndarray): The matrix :math:`\\Delta`,
                which represents the coefficients of the
                non-particle-number-conserving terms.
                This is an `n_qubits` x `n_qubits` numpy array of complex
                numbers.
            chemical_potential(float): The chemical potential :math:`\mu`.
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
        original ones:

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
        This method returns the matrix :math:`W`.

        Returns:
            diagonalizing_unitary (ndarray):
                A (2 * `n_qubits`) x (2 * `n_qubits`) matrix representing
                the transformation :math:`W` of the fermionic ladder operators.
        """
        majorana_matrix, majorana_constant = self.majorana_form()

        # Get the orthogonal transformation that puts majorana_matrix
        # into canonical form
        canonical, orthogonal = antisymmetric_canonical_form(majorana_matrix)

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
