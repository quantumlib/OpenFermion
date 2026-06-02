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

import numpy
import pytest
import cirq
from cirq import LineQubit

import openfermion
from openfermion.linalg import get_sparse_operator
from openfermion.circuits.primitives.state_preparation import (
    prepare_slater_determinant,
    prepare_gaussian_state,
)
from openfermion.testing import random_quadratic_hamiltonian


@pytest.mark.parametrize(
    'n_qubits, conserves_particle_number, occupied_orbitals, initial_state',
    [
        (4, True, None, 0b0010),
        (4, False, None, 0b1001),
        (5, True, None, 0),
        (5, False, None, 0b10101),
        (5, True, range(4), 0),
        (5, False, (0, 2, 3), [1, 2, 3, 4]),
    ],
)
def test_prepare_gaussian_state(
    n_qubits, conserves_particle_number, occupied_orbitals, initial_state, atol=1e-5
):
    qubits = LineQubit.range(n_qubits)

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(n_qubits, conserves_particle_number, real=False)
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the energy of the desired state
    if occupied_orbitals is None:
        energy = quad_ham.ground_energy()
    else:
        orbital_energies, _, constant = quad_ham.diagonalizing_bogoliubov_transform()
        energy = sum(orbital_energies[i] for i in occupied_orbitals) + constant

    # Get the state using a circuit simulation
    circuit = cirq.Circuit(
        prepare_gaussian_state(qubits, quad_ham, occupied_orbitals, initial_state=initial_state)
    )
    if isinstance(initial_state, list):
        initial_state = sum(1 << (n_qubits - 1 - i) for i in initial_state)
    state = circuit.final_state_vector(initial_state=initial_state)

    # Check that the result is an eigenstate with the correct eigenvalue
    numpy.testing.assert_allclose(quad_ham_sparse.dot(state), energy * state, atol=atol)


@pytest.mark.parametrize(
    'n_spatial_orbitals, conserves_particle_number, occupied_orbitals, ' 'initial_state',
    [
        (4, True, [range(1), range(1)], 0b00100100),
        (5, True, [range(2), range(1)], list(range(3))),
        (5, True, [[0, 2], [1, 3]], 0),
    ],
)
def test_prepare_gaussian_state_with_spin_symmetry(
    n_spatial_orbitals, conserves_particle_number, occupied_orbitals, initial_state, atol=1e-5
):
    n_qubits = 2 * n_spatial_orbitals
    qubits = LineQubit.range(n_qubits)

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(
        n_spatial_orbitals, conserves_particle_number, real=True, expand_spin=True, seed=639
    )

    # Reorder the Hamiltonian and get sparse matrix
    quad_ham = openfermion.get_quadratic_hamiltonian(
        openfermion.reorder(openfermion.get_fermion_operator(quad_ham), openfermion.up_then_down)
    )
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the energy of the desired state
    energy = 0.0
    for spin_sector in range(2):
        orbital_energies, _, _ = quad_ham.diagonalizing_bogoliubov_transform(
            spin_sector=spin_sector
        )
        energy += sum(orbital_energies[i] for i in occupied_orbitals[spin_sector])
    energy += quad_ham.constant

    # Get the state using a circuit simulation
    circuit = cirq.Circuit(
        prepare_gaussian_state(qubits, quad_ham, occupied_orbitals, initial_state=initial_state)
    )

    if isinstance(initial_state, list):
        initial_state = sum(1 << (n_qubits - 1 - i) for i in initial_state)
    state = circuit.final_state_vector(initial_state=initial_state)

    # Check that the result is an eigenstate with the correct eigenvalue
    numpy.testing.assert_allclose(quad_ham_sparse.dot(state), energy * state, atol=atol)


@pytest.mark.parametrize(
    'slater_determinant_matrix, correct_state, initial_state',
    [
        (numpy.array([[1, 1]]) / numpy.sqrt(2), numpy.array([0, 1, 1, 0]) / numpy.sqrt(2), 0),
        (numpy.array([[1, 1j]]) / numpy.sqrt(2), numpy.array([0, 1j, 1, 0]) / numpy.sqrt(2), 0b01),
        (
            numpy.array(
                [[1, 1, 1], [1, numpy.exp(2j * numpy.pi / 3), numpy.exp(4j * numpy.pi / 3)]]
            )
            / numpy.sqrt(3),
            numpy.array(
                [0, 0, 0, numpy.exp(2j * numpy.pi / 3), 0, 1 + numpy.exp(2j * numpy.pi / 3), 1, 0]
            )
            / numpy.sqrt(3),
            0,
        ),
        (
            numpy.array(
                [[1, 1, 1], [1, numpy.exp(2j * numpy.pi / 3), numpy.exp(4j * numpy.pi / 3)]]
            )
            / numpy.sqrt(3),
            numpy.array(
                [0, 0, 0, numpy.exp(2j * numpy.pi / 3), 0, 1 + numpy.exp(2j * numpy.pi / 3), 1, 0]
            )
            / numpy.sqrt(3),
            [0, 2],
        ),
    ],
)
def test_prepare_slater_determinant(
    slater_determinant_matrix, correct_state, initial_state, atol=1e-7
):
    n_qubits = slater_determinant_matrix.shape[1]
    qubits = LineQubit.range(n_qubits)

    circuit = cirq.Circuit(
        prepare_slater_determinant(qubits, slater_determinant_matrix, initial_state=initial_state)
    )
    if isinstance(initial_state, list):
        initial_state = sum(1 << (n_qubits - 1 - i) for i in initial_state)
    state = circuit.final_state_vector(initial_state=initial_state)

    assert cirq.allclose_up_to_global_phase(state, correct_state, atol=atol)
