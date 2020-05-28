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

"""This module contains functions for compiling circuits to prepare
Slater determinants and fermionic Gaussian states."""

import numpy

from openfermion.ops import QuadraticHamiltonian
from openfermion.ops._givens_rotations import (
    fermionic_gaussian_decomposition, givens_decomposition)


def gaussian_state_preparation_circuit(
        quadratic_hamiltonian,
        occupied_orbitals=None,
        spin_sector=None):
    r"""Obtain the description of a circuit which prepares a fermionic Gaussian
    state.

    Fermionic Gaussian states can be regarded as eigenstates of quadratic
    Hamiltonians. If the Hamiltonian conserves particle number, then these are
    just Slater determinants. See arXiv:1711.05395 for a detailed description
    of how this procedure works.

    The circuit description is returned as a sequence of elementary
    operations; operations that can be performed in parallel are grouped
    together. Each elementary operation is either

    - the string 'pht', indicating the particle-hole transformation
      on the last fermionic mode, which is the operator :math:`\mathcal{B}`
      such that

      .. math::

          \begin{align}
              \mathcal{B} a_N \mathcal{B}^\dagger &= a_N^\dagger,\\
              \mathcal{B} a_j \mathcal{B}^\dagger &= a_j, \quad
                  j = 1, \ldots, N-1,
          \end{align}

      or

    - a tuple :math:`(i, j, \theta, \varphi)`, indicating the operation

      .. math::
          \exp[i \varphi a_j^\dagger a_j]
          \exp[\theta (a_i^\dagger a_j - a_j^\dagger a_i)],

      a Givens rotation of modes :math:`i` and :math:`j` by angles
      :math:`\theta` and :math:`\varphi`.

    Args:
        quadratic_hamiltonian(QuadraticHamiltonian):
            The Hamiltonian whose eigenstate is desired.
        occupied_orbitals(list):
            A list of integers representing the indices of the occupied
            orbitals in the desired Gaussian state. If this is None
            (the default), then it is assumed that the ground state is
            desired, i.e., the orbitals with negative energies are filled.
        spin_sector (optional str): An optional integer specifying
            a spin sector to restrict to: 0 for spin-up and 1 for
            spin-down. If specified, the returned circuit acts on modes
            indexed by spatial indices (rather than spin indices).
            Should only be specified if the Hamiltonian
            includes a spin degree of freedom and spin-up modes
            do not interact with spin-down modes.

    Returns
    -------
        circuit_description (list[tuple]):
            A list of operations describing the circuit. Each operation
            is a tuple of objects describing elementary operations that
            can be performed in parallel. Each elementary operation
            is either the string 'pht', indicating a particle-hole
            transformation on the last fermionic mode, or a tuple of
            the form :math:`(i, j, \theta, \varphi)`,
            indicating a Givens rotation
            of modes :math:`i` and :math:`j` by angles :math:`\theta`
            and :math:`\varphi`.
        start_orbitals (list):
            The occupied orbitals to start with. This describes the
            initial state that the circuit should be applied to: it should
            be a Slater determinant (in the computational basis) with these
            orbitals filled.
    """
    if not isinstance(quadratic_hamiltonian, QuadraticHamiltonian):
        raise ValueError('Input must be an instance of QuadraticHamiltonian.')

    orbital_energies, transformation_matrix, _ = (
            quadratic_hamiltonian.diagonalizing_bogoliubov_transform(
                spin_sector=spin_sector)
    )

    if quadratic_hamiltonian.conserves_particle_number:
        if occupied_orbitals is None:
            # The ground state is desired, so we fill the orbitals that have
            # negative energy
            occupied_orbitals = numpy.where(orbital_energies < 0.0)[0]

        # Get the unitary rows which represent the Slater determinant
        slater_determinant_matrix = transformation_matrix[occupied_orbitals]

        # Get the circuit description
        circuit_description = slater_determinant_preparation_circuit(
            slater_determinant_matrix)
        start_orbitals = range(len(occupied_orbitals))
    else:
        # TODO implement this
        if spin_sector is not None:
            raise NotImplementedError(
                    'Specifying spin sector for non-particle-conserving '
                    'Hamiltonians is not yet supported.'
                    )
        # Rearrange the transformation matrix because the circuit generation
        # routine expects it to describe annihilation operators rather than
        # creation operators
        n_qubits = quadratic_hamiltonian.n_qubits
        left_block = transformation_matrix[:, :n_qubits]
        right_block = transformation_matrix[:, n_qubits:]
        # Can't use numpy.block because that requires numpy>=1.13.0
        new_transformation_matrix = numpy.empty((n_qubits, 2 * n_qubits),
                                                dtype=complex)
        new_transformation_matrix[:, :n_qubits] = numpy.conjugate(right_block)
        new_transformation_matrix[:, n_qubits:] = numpy.conjugate(left_block)

        # Get the circuit description
        decomposition, left_decomposition, _, _ = (
            fermionic_gaussian_decomposition(new_transformation_matrix))
        if occupied_orbitals is None:
            # The ground state is desired, so the circuit should be applied
            # to the vaccuum state
            start_orbitals = []
            circuit_description = list(reversed(decomposition))
        else:
            start_orbitals = occupied_orbitals
            # The circuit won't be applied to the ground state, so we need to
            # use left_decomposition
            circuit_description = list(reversed(
                decomposition + left_decomposition))

    return circuit_description, start_orbitals


def slater_determinant_preparation_circuit(slater_determinant_matrix):
    r"""Obtain the description of a circuit which prepares a Slater determinant.

    The input is an :math:`N_f \times N` matrix :math:`Q` with orthonormal
    rows. Such a matrix describes the Slater determinant

    .. math::

        b^\dagger_1 \cdots b^\dagger_{N_f} \lvert \text{vac} \rangle,

    where

    .. math::

        b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k.

    The output is the description of a circuit which prepares this
    Slater determinant, up to a global phase.
    The starting state which the circuit should be applied to
    is a Slater determinant (in the computational basis) with
    the first :math:`N_f` orbitals filled.

    Args:
        slater_determinant_matrix: The matrix :math:`Q` which describes the
            Slater determinant to be prepared.
    Returns:
        circuit_description:
            A list of operations describing the circuit. Each operation
            is a tuple of elementary operations that can be performed in
            parallel. Each elementary operation is a tuple of the form
            :math:`(i, j, \theta, \varphi)`, indicating a Givens rotation
            of modes :math:`i` and :math:`j` by angles :math:`\theta`
            and :math:`\varphi`.
    """
    decomposition, _, _ = givens_decomposition(slater_determinant_matrix)
    circuit_description = list(reversed(decomposition))
    return circuit_description
