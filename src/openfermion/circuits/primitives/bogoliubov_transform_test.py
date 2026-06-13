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

from typing import Container

import numpy
import pytest
import cirq
from cirq import LineQubit

import openfermion
from openfermion import bogoliubov_transform, get_sparse_operator
from openfermion.testing import random_quadratic_hamiltonian, random_unitary_matrix


def fourier_transform_matrix(n_modes):
    root_of_unity = numpy.exp(2j * numpy.pi / n_modes)
    return numpy.array(
        [[root_of_unity ** (j * k) for k in range(n_modes)] for j in range(n_modes)]
    ) / numpy.sqrt(n_modes)


@pytest.mark.parametrize(
    'transformation_matrix, initial_state, correct_state',
    [
        (fourier_transform_matrix(3), 4, numpy.array([0, 1, 1, 0, 1, 0, 0, 0]) / numpy.sqrt(3)),
        (
            fourier_transform_matrix(3),
            [1, 2],
            numpy.array(
                [
                    0,
                    0,
                    0,
                    numpy.exp(2j * numpy.pi / 3) - 1,
                    0,
                    1 - numpy.exp(2j * numpy.pi / 3),
                    numpy.exp(2j * numpy.pi / 3) - 1,
                    0,
                ]
            )
            / 3,
        ),
    ],
)
def test_bogoliubov_transform_fourier_transform(
    transformation_matrix, initial_state, correct_state, atol=5e-6
):
    n_qubits = transformation_matrix.shape[0]
    sim = cirq.Simulator(dtype=numpy.complex128)
    qubits = LineQubit.range(n_qubits)
    if isinstance(initial_state, Container):
        initial_state = sum(1 << (n_qubits - 1 - i) for i in initial_state)

    circuit = cirq.Circuit(
        bogoliubov_transform(qubits, transformation_matrix, initial_state=initial_state)
    )
    state = sim.simulate(circuit, initial_state=initial_state).final_state_vector
    cirq.testing.assert_allclose_up_to_global_phase(state, correct_state, atol=atol)


@pytest.mark.parametrize('n_spatial_orbitals, conserves_particle_number', [(4, True), (5, True)])
def test_spin_symmetric_bogoliubov_transform(
    n_spatial_orbitals, conserves_particle_number, atol=5e-5
):
    n_qubits = 2 * n_spatial_orbitals
    qubits = LineQubit.range(n_qubits)
    sim = cirq.Simulator(dtype=numpy.complex128)

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(
        n_spatial_orbitals, conserves_particle_number, real=True, expand_spin=True, seed=28166
    )

    # Reorder the Hamiltonian and get sparse matrix
    quad_ham = openfermion.get_quadratic_hamiltonian(
        openfermion.reorder(openfermion.get_fermion_operator(quad_ham), openfermion.up_then_down)
    )
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the orbital energies and transformation_matrix
    up_orbital_energies, _, _ = quad_ham.diagonalizing_bogoliubov_transform(spin_sector=0)
    down_orbital_energies, _, _ = quad_ham.diagonalizing_bogoliubov_transform(spin_sector=1)
    _, transformation_matrix, _ = quad_ham.diagonalizing_bogoliubov_transform()

    # Pick some orbitals to occupy
    up_orbitals = list(range(2))
    down_orbitals = [0, 2, 3]
    energy = (
        sum(up_orbital_energies[up_orbitals])
        + sum(down_orbital_energies[down_orbitals])
        + quad_ham.constant
    )

    # Construct initial state
    initial_state = sum(2 ** (n_qubits - 1 - int(i)) for i in up_orbitals) + sum(
        2 ** (n_qubits - 1 - int(i + n_spatial_orbitals)) for i in down_orbitals
    )

    # Apply the circuit
    circuit = cirq.Circuit(
        bogoliubov_transform(qubits, transformation_matrix, initial_state=initial_state)
    )
    state = sim.simulate(circuit, initial_state=initial_state).final_state_vector

    # Check that the result is an eigenstate with the correct eigenvalue
    numpy.testing.assert_allclose(quad_ham_sparse.dot(state), energy * state, atol=atol)


def _independent_spin_sector_transform(n_spatial_orbitals, seed_up, seed_down, real):
    """Assemble an N x 2N Bogoliubov transformation that does not mix spin.

    Builds two independent non-particle-conserving quadratic Hamiltonians, one
    per spin sector, and places their diagonalizing transforms into a combined
    matrix. The matrix acts on the column vector of all creation operators
    followed by all annihilation operators, so each sector's creation
    (annihilation) block lands in the left (right) half, offset by the sector's
    position within that half.
    """
    quad_ham_up = random_quadratic_hamiltonian(
        n_spatial_orbitals, conserves_particle_number=False, real=real, seed=seed_up
    )
    quad_ham_down = random_quadratic_hamiltonian(
        n_spatial_orbitals, conserves_particle_number=False, real=real, seed=seed_down
    )
    up_energies, up_matrix, up_constant = quad_ham_up.diagonalizing_bogoliubov_transform()
    down_energies, down_matrix, down_constant = quad_ham_down.diagonalizing_bogoliubov_transform()

    n = n_spatial_orbitals
    n_qubits = 2 * n
    transformation_matrix = numpy.zeros((n_qubits, 2 * n_qubits), dtype=complex)
    transformation_matrix[:n, :n] = up_matrix[:, :n]
    transformation_matrix[:n, n_qubits : n_qubits + n] = up_matrix[:, n:]
    transformation_matrix[n:, n:n_qubits] = down_matrix[:, :n]
    transformation_matrix[n:, n_qubits + n :] = down_matrix[:, n:]
    return (
        transformation_matrix,
        (quad_ham_up, up_energies, up_constant),
        (quad_ham_down, down_energies, down_constant),
    )


def test_bogoliubov_transform_spin_block_gaussian_regression():
    """Regression test for issue #776.

    A non-square transformation matrix that does not mix spin sectors used to
    be misidentified as a square spin-block-diagonal matrix, leading to bogus
    slicing and a ValueError when constructing the circuit.
    """
    qubits = LineQubit.range(2)
    phase = -9.57167901e-01 - 2.89533435e-01j
    phase /= abs(phase)
    transformation_matrix = numpy.array([[phase, 0, 0, 0], [0, 1, 0, 0]], dtype=complex)

    circuit = cirq.Circuit(
        bogoliubov_transform(qubits, transformation_matrix), cirq.I.on_each(*qubits)
    )

    # The transformation is a phase rotation of mode 0, so it should leave
    # computational basis states invariant up to phase.
    state = circuit.final_state_vector(initial_state=0, qubit_order=qubits, dtype=numpy.complex128)
    expected_state = numpy.zeros(4, dtype=numpy.complex128)
    expected_state[0] = 1
    cirq.testing.assert_allclose_up_to_global_phase(state, expected_state, atol=1e-6)


@pytest.mark.parametrize(
    'n_spatial_orbitals, real, up_orbitals, down_orbitals',
    [(2, True, [0], [0, 1]), (2, False, [1], []), (3, True, [0, 2], [1]), (3, False, [], [0, 2])],
)
def test_bogoliubov_transform_spin_block_gaussian(
    n_spatial_orbitals, real, up_orbitals, down_orbitals, atol=5e-5
):
    """A non-particle-conserving transform that does not mix spin sectors must
    prepare eigenstates of the combined Hamiltonian with the correct energy."""
    n_qubits = 2 * n_spatial_orbitals
    qubits = LineQubit.range(n_qubits)
    sim = cirq.Simulator(dtype=numpy.complex128)

    transformation_matrix, up_data, down_data = _independent_spin_sector_transform(
        n_spatial_orbitals, 51624, 48201, real
    )
    quad_ham_up, up_energies, up_constant = up_data
    quad_ham_down, down_energies, down_constant = down_data

    # Combined Hamiltonian with spin-down modes shifted to come after spin-up
    ferm_op = openfermion.get_fermion_operator(quad_ham_up)
    for term, coefficient in openfermion.get_fermion_operator(quad_ham_down).terms.items():
        shifted_term = tuple((index + n_spatial_orbitals, action) for index, action in term)
        ferm_op += openfermion.FermionOperator(shifted_term, coefficient)
    combined_ham_sparse = get_sparse_operator(ferm_op, n_qubits=n_qubits)

    energy = (
        sum(up_energies[i] for i in up_orbitals)
        + sum(down_energies[i] for i in down_orbitals)
        + up_constant
        + down_constant
    )
    occupied_orbitals = list(up_orbitals) + [i + n_spatial_orbitals for i in down_orbitals]
    initial_state = sum(2 ** (n_qubits - 1 - int(i)) for i in occupied_orbitals)

    circuit = cirq.Circuit(
        bogoliubov_transform(qubits, transformation_matrix, initial_state=initial_state)
    )
    state = sim.simulate(
        circuit, initial_state=initial_state, qubit_order=qubits
    ).final_state_vector

    numpy.testing.assert_allclose(combined_ham_sparse.dot(state), energy * state, atol=atol)


@pytest.mark.parametrize(
    'n_spatial_orbitals, seed_up, seed_down, real',
    [
        # The (2, 1000, ...) case has a parity-preserving spin-up sector while
        # the (2, 1001, ...) case has a parity-flipping one; the latter is the
        # regression guard for the spin-down Jordan-Wigner sign correction.
        (2, 1000, 2000, True),
        (2, 1001, 2001, True),
        (3, 1000, 2000, False),
        (3, 1001, 2001, True),
    ],
)
def test_bogoliubov_transform_spin_block_operator_algebra(
    n_spatial_orbitals, seed_up, seed_down, real, atol=1e-6
):
    """The factorized circuit must implement the documented transformation
    exactly: U a_p^dagger U^-1 = b_p^dagger for every mode p, including the
    cross-sector Jordan-Wigner sign carried by spin-down operators. Eigenstate
    energies cannot detect a wrong sign on b_p^dagger, so this checks the full
    operator algebra of the circuit's unitary directly.
    """
    n_qubits = 2 * n_spatial_orbitals
    qubits = LineQubit.range(n_qubits)
    transformation_matrix, _, _ = _independent_spin_sector_transform(
        n_spatial_orbitals, seed_up, seed_down, real
    )

    circuit = cirq.Circuit(
        bogoliubov_transform(qubits, transformation_matrix), cirq.I.on_each(*qubits)
    )
    unitary = circuit.unitary(qubit_order=qubits)

    def ladder(mode, action):
        return get_sparse_operator(
            openfermion.FermionOperator(((mode, action),)), n_qubits=n_qubits
        ).toarray()

    for p in range(n_qubits):
        transformed = unitary @ ladder(p, 1) @ unitary.conj().T
        expected = numpy.zeros_like(transformed)
        for q in range(n_qubits):
            expected += transformation_matrix[p, q] * ladder(q, 1)
            expected += transformation_matrix[p, n_qubits + q] * ladder(q, 0)
        numpy.testing.assert_allclose(transformed, expected, atol=atol)


def test_bogoliubov_transform_spin_mixing_gaussian_not_split(atol=5e-5):
    """A transform whose annihilation block mixes spin sectors must use the
    general procedure even if its creation block is spin-block-diagonal."""
    n_qubits = 4
    qubits = LineQubit.range(n_qubits)
    sim = cirq.Simulator(dtype=numpy.complex128)

    # A pairing Hamiltonian whose antisymmetric part couples the two sectors
    quad_ham = random_quadratic_hamiltonian(
        n_qubits, conserves_particle_number=False, real=True, seed=63003
    )
    quad_ham_sparse = get_sparse_operator(quad_ham)
    energies, transformation_matrix, constant = quad_ham.diagonalizing_bogoliubov_transform()

    occupied_orbitals = [0, 2]
    energy = sum(energies[i] for i in occupied_orbitals) + constant
    initial_state = sum(2 ** (n_qubits - 1 - int(i)) for i in occupied_orbitals)

    circuit = cirq.Circuit(
        bogoliubov_transform(qubits, transformation_matrix, initial_state=initial_state)
    )
    state = sim.simulate(
        circuit, initial_state=initial_state, qubit_order=qubits
    ).final_state_vector

    numpy.testing.assert_allclose(quad_ham_sparse.dot(state), energy * state, atol=atol)


@pytest.mark.parametrize(
    'n_qubits, conserves_particle_number', [(4, True), (4, False), (5, True), (5, False)]
)
def test_bogoliubov_transform_quadratic_hamiltonian(n_qubits, conserves_particle_number, atol=5e-5):
    qubits = LineQubit.range(n_qubits)
    sim = cirq.Simulator(dtype=numpy.complex128)

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(n_qubits, conserves_particle_number, real=False)
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the orbital energies and circuit
    orbital_energies, transformation_matrix, constant = (
        quad_ham.diagonalizing_bogoliubov_transform()
    )
    circuit = cirq.Circuit(bogoliubov_transform(qubits, transformation_matrix))

    # Pick some random eigenstates to prepare, which correspond to random
    # subsets of [0 ... n_qubits - 1]
    n_eigenstates = min(2**n_qubits, 5)
    subsets = [
        numpy.random.choice(range(n_qubits), numpy.random.randint(1, n_qubits + 1), False)
        for _ in range(n_eigenstates)
    ]
    # Also test empty subset
    subsets += [()]

    for occupied_orbitals in subsets:
        # Compute the energy of this eigenstate
        energy = sum(orbital_energies[i] for i in occupied_orbitals) + constant

        # Construct initial state
        initial_state = sum(2 ** (n_qubits - 1 - int(i)) for i in occupied_orbitals)

        # Get the state using a circuit simulation
        state1 = sim.simulate(circuit, initial_state=initial_state).final_state_vector

        # Also test the option to start with a computational basis state
        special_circuit = cirq.Circuit(
            bogoliubov_transform(qubits, transformation_matrix, initial_state=initial_state)
        )
        state2 = sim.simulate(
            special_circuit, initial_state=initial_state, qubit_order=qubits
        ).final_state_vector

        # Check that the result is an eigenstate with the correct eigenvalue
        numpy.testing.assert_allclose(quad_ham_sparse.dot(state1), energy * state1, atol=atol)
        numpy.testing.assert_allclose(quad_ham_sparse.dot(state2), energy * state2, atol=atol)


@pytest.mark.parametrize('n_qubits, atol', [(4, 1e-7), (5, 1e-7)])
def test_bogoliubov_transform_fourier_transform_inverse_is_dagger(n_qubits, atol):
    u = fourier_transform_matrix(n_qubits)

    qubits = cirq.LineQubit.range(n_qubits)

    circuit1 = cirq.Circuit(cirq.inverse(bogoliubov_transform(qubits, u)))

    circuit2 = cirq.Circuit(bogoliubov_transform(qubits, u.T.conj()))

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit1.unitary(), circuit2.unitary(), atol=atol
    )


@pytest.mark.parametrize(
    'n_qubits, real, particle_conserving, atol',
    [
        (5, True, True, 1e-7),
        (5, False, True, 1e-7),
        (5, True, False, 1e-7),
        (5, False, False, 1e-7),
    ],
)
def test_bogoliubov_transform_quadratic_hamiltonian_inverse_is_dagger(
    n_qubits, real, particle_conserving, atol
):
    quad_ham = random_quadratic_hamiltonian(
        n_qubits, real=real, conserves_particle_number=particle_conserving, seed=46533
    )
    _, transformation_matrix, _ = quad_ham.diagonalizing_bogoliubov_transform()

    qubits = cirq.LineQubit.range(n_qubits)

    if transformation_matrix.shape == (n_qubits, n_qubits):
        daggered_transformation_matrix = transformation_matrix.T.conj()
    else:
        left_block = transformation_matrix[:, :n_qubits]
        right_block = transformation_matrix[:, n_qubits:]
        daggered_transformation_matrix = numpy.block([left_block.T.conj(), right_block.T])

    circuit1 = cirq.Circuit(cirq.inverse(bogoliubov_transform(qubits, transformation_matrix)))

    circuit2 = cirq.Circuit(bogoliubov_transform(qubits, daggered_transformation_matrix))

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit1.unitary(), circuit2.unitary(), atol=atol
    )


@pytest.mark.parametrize('n_qubits, atol', [(4, 1e-7), (5, 1e-7)])
def test_bogoliubov_transform_compose(n_qubits, atol):
    u = random_unitary_matrix(n_qubits, seed=24964)
    v = random_unitary_matrix(n_qubits, seed=33656)

    qubits = cirq.LineQubit.range(n_qubits)

    circuit1 = cirq.Circuit(bogoliubov_transform(qubits, u), bogoliubov_transform(qubits, v))

    circuit2 = cirq.Circuit(bogoliubov_transform(qubits, u.dot(v)))

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit1.unitary(), circuit2.unitary(), atol=atol
    )


def test_bogoliubov_transform_bad_shape_raises_error():
    with pytest.raises(ValueError):
        _ = next(bogoliubov_transform(cirq.LineQubit.range(4), numpy.zeros((4, 7))))
