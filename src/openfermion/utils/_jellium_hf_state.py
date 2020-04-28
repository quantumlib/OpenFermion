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

"""This module constructs the uniform electron gas' Hartree-Fock state."""

import numpy
from openfermion.ops import FermionOperator
from openfermion.utils import (count_qubits,
                               inverse_fourier_transform,
                               normal_ordered)

from scipy.sparse import csr_matrix, dok_matrix


def lowest_single_particle_energy_states(hamiltonian, n_states):
    """Find the lowest single-particle states of the given Hamiltonian.

    Args:
        hamiltonian (FermionOperator)
        n_states (int): Number of lowest energy states to give."""
    # Enumerate the single-particle states.
    n_single_particle_states = count_qubits(hamiltonian)

    # Compute the energies for each of the single-particle states.
    single_particle_energies = numpy.zeros(n_single_particle_states,
                                           dtype=float)
    for i in range(n_single_particle_states):
        single_particle_energies[i] = hamiltonian.terms.get(
            ((i, 1), (i, 0)), 0.0)

    # Find the n_states lowest states.
    occupied_states = single_particle_energies.argsort()[:n_states]

    return list(occupied_states)


def hartree_fock_state_jellium(grid, n_electrons, spinless=True,
                               plane_wave=False):
    """Give the Hartree-Fock state of jellium.

    Args:
        grid (Grid): The discretization to use.
        n_electrons (int): Number of electrons in the system.
        spinless (bool): Whether to use the spinless model or not.
        plane_wave (bool): Whether to return the Hartree-Fock state in
                           the plane wave (True) or dual basis (False).

    Notes:
        The jellium model is built up by filling the lowest-energy
        single-particle states in the plane-wave Hamiltonian until
        n_electrons states are filled.
    """
    from openfermion.hamiltonians import plane_wave_kinetic
    # Get the jellium Hamiltonian in the plane wave basis.
    # For determining the Hartree-Fock state in the PW basis, only the
    # kinetic energy terms matter.
    hamiltonian = plane_wave_kinetic(grid, spinless=spinless)
    hamiltonian = normal_ordered(hamiltonian)
    hamiltonian.compress()

    # The number of occupied single-particle states is the number of electrons.
    # Those states with the lowest single-particle energies are occupied first.
    occupied_states = lowest_single_particle_energy_states(
        hamiltonian, n_electrons)
    occupied_states = numpy.array(occupied_states)

    if plane_wave:
        # In the plane wave basis the HF state is a single determinant.
        hartree_fock_state_index = numpy.sum(2 ** occupied_states)
        hartree_fock_state = numpy.zeros(2 ** count_qubits(hamiltonian),
                                         dtype=complex)
        hartree_fock_state[hartree_fock_state_index] = 1.0

    else:
        # Inverse Fourier transform the creation operators for the state to get
        # to the dual basis state, then use that to get the dual basis state.
        hartree_fock_state_creation_operator = FermionOperator.identity()
        for state in occupied_states[::-1]:
            hartree_fock_state_creation_operator *= (
                FermionOperator(((int(state), 1),)))
        dual_basis_hf_creation_operator = inverse_fourier_transform(
            hartree_fock_state_creation_operator, grid, spinless)

        dual_basis_hf_creation = normal_ordered(
            dual_basis_hf_creation_operator)

        # Initialize the HF state.
        hartree_fock_state = numpy.zeros(2 ** count_qubits(hamiltonian),
                                         dtype=complex)

        # Populate the elements of the HF state in the dual basis.
        for term in dual_basis_hf_creation.terms:
            index = 0
            for operator in term:
                index += 2 ** operator[0]
            hartree_fock_state[index] = dual_basis_hf_creation.terms[term]

    return hartree_fock_state
