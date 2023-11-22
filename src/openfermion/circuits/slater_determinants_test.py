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
"""Tests for slater_determinants.py."""
import unittest

import pytest

import numpy

from openfermion.linalg.sparse_tools import (
    jw_sparse_givens_rotation,
    jw_sparse_particle_hole_transformation_last_mode,
    get_sparse_operator,
    get_ground_state,
    jw_configuration_state,
)
from openfermion.testing.testing_utils import random_quadratic_hamiltonian

from openfermion.circuits.slater_determinants import (
    gaussian_state_preparation_circuit,
    jw_get_gaussian_state,
    jw_slater_determinant,
)


class JWSlaterDeterminantTest(unittest.TestCase):
    def test_hadamard_transform(self):
        r"""Test creating the states
        1 / sqrt(2) (a^\dagger_0 + a^\dagger_1) |vac>
        and
        1 / sqrt(2) (a^\dagger_0 - a^\dagger_1) |vac>.
        """
        slater_determinant_matrix = numpy.array([[1.0, 1.0]]) / numpy.sqrt(2.0)
        slater_determinant = jw_slater_determinant(slater_determinant_matrix)
        self.assertAlmostEqual(slater_determinant[1], slater_determinant[2])
        self.assertAlmostEqual(abs(slater_determinant[1]), 1.0 / numpy.sqrt(2.0))
        self.assertAlmostEqual(abs(slater_determinant[0]), 0.0)
        self.assertAlmostEqual(abs(slater_determinant[3]), 0.0)

        slater_determinant_matrix = numpy.array([[1.0, -1.0]]) / numpy.sqrt(2.0)
        slater_determinant = jw_slater_determinant(slater_determinant_matrix)
        self.assertAlmostEqual(slater_determinant[1], -slater_determinant[2])
        self.assertAlmostEqual(abs(slater_determinant[1]), 1.0 / numpy.sqrt(2.0))
        self.assertAlmostEqual(abs(slater_determinant[0]), 0.0)
        self.assertAlmostEqual(abs(slater_determinant[3]), 0.0)


class GaussianStatePreparationCircuitTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits_range = range(3, 6)

    def test_ground_state_particle_conserving(self):
        """Test getting the ground state preparation circuit for a Hamiltonian
        that conserves particle number."""
        for n_qubits in self.n_qubits_range:
            print(n_qubits)
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, True, True)
            print(quadratic_hamiltonian)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, _ = get_ground_state(sparse_operator)

            # Obtain the circuit
            circuit_description, start_orbitals = gaussian_state_preparation_circuit(
                quadratic_hamiltonian
            )

            # Initialize the starting state
            state = jw_configuration_state(start_orbitals, n_qubits)

            # Apply the circuit
            for parallel_ops in circuit_description:
                for op in parallel_ops:
                    self.assertTrue(op != 'pht')
                    i, j, theta, phi = op
                    state = jw_sparse_givens_rotation(i, j, theta, phi, n_qubits).dot(state)

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * state - ground_energy * state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_ground_state_particle_nonconserving(self):
        """Test getting the ground state preparation circuit for a Hamiltonian
        that does not conserve particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, False, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, _ = get_ground_state(sparse_operator)

            # Obtain the circuit
            circuit_description, start_orbitals = gaussian_state_preparation_circuit(
                quadratic_hamiltonian
            )

            # Initialize the starting state
            state = jw_configuration_state(start_orbitals, n_qubits)

            # Apply the circuit
            particle_hole_transformation = jw_sparse_particle_hole_transformation_last_mode(
                n_qubits
            )
            for parallel_ops in circuit_description:
                for op in parallel_ops:
                    if op == 'pht':
                        state = particle_hole_transformation.dot(state)
                    else:
                        i, j, theta, phi = op
                        state = jw_sparse_givens_rotation(i, j, theta, phi, n_qubits).dot(state)

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * state - ground_energy * state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            gaussian_state_preparation_circuit('a')


class JWGetGaussianStateTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits_range = range(2, 10)

    def test_ground_state_particle_conserving(self):
        """Test getting the ground state of a Hamiltonian that conserves
        particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, _ = get_ground_state(sparse_operator)

            # Compute the ground state using the circuit
            circuit_energy, circuit_state = jw_get_gaussian_state(quadratic_hamiltonian)

            # Check that the energies match
            self.assertAlmostEqual(ground_energy, circuit_energy)

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * circuit_state - ground_energy * circuit_state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_ground_state_particle_nonconserving(self):
        """Test getting the ground state of a Hamiltonian that does not
        conserve particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a non-particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, False)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, _ = get_ground_state(sparse_operator)

            # Compute the ground state using the circuit
            circuit_energy, circuit_state = jw_get_gaussian_state(quadratic_hamiltonian)

            # Check that the energies match
            self.assertAlmostEqual(ground_energy, circuit_energy)

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * circuit_state - ground_energy * circuit_state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_excited_state_particle_conserving(self):
        """Test getting an excited state of a Hamiltonian that conserves
        particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, True)

            # Pick some orbitals to occupy
            num_occupied_orbitals = numpy.random.randint(1, n_qubits + 1)
            occupied_orbitals = numpy.random.choice(range(n_qubits), num_occupied_orbitals, False)

            # Compute the Gaussian state
            circuit_energy, gaussian_state = jw_get_gaussian_state(
                quadratic_hamiltonian, occupied_orbitals
            )

            # Compute the true energy
            orbital_energies, constant = quadratic_hamiltonian.orbital_energies()
            energy = numpy.sum(orbital_energies[occupied_orbitals]) + constant

            # Check that the energies match
            self.assertAlmostEqual(energy, circuit_energy)

            # Check that the state obtained using the circuit is an eigenstate
            # with the correct eigenvalue
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            difference = sparse_operator * gaussian_state - energy * gaussian_state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_excited_state_particle_nonconserving(self):
        """Test getting an excited state of a Hamiltonian that conserves
        particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a non-particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, False)

            # Pick some orbitals to occupy
            num_occupied_orbitals = numpy.random.randint(1, n_qubits + 1)
            occupied_orbitals = numpy.random.choice(range(n_qubits), num_occupied_orbitals, False)

            # Compute the Gaussian state
            circuit_energy, gaussian_state = jw_get_gaussian_state(
                quadratic_hamiltonian, occupied_orbitals
            )

            # Compute the true energy
            orbital_energies, constant = quadratic_hamiltonian.orbital_energies()
            energy = numpy.sum(orbital_energies[occupied_orbitals]) + constant

            # Check that the energies match
            self.assertAlmostEqual(energy, circuit_energy)

            # Check that the state obtained using the circuit is an eigenstate
            # with the correct eigenvalue
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            difference = sparse_operator * gaussian_state - energy * gaussian_state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            jw_get_gaussian_state('a')


def test_not_implemented_spinr_reduced():
    """Tests that currently un-implemented functionality is caught."""
    msg = "Specifying spin sector for non-particle-conserving "
    msg += "Hamiltonians is not yet supported."
    for n_qubits in [2, 4, 6]:
        # Initialize a particle-number-conserving Hamiltonian
        quadratic_hamiltonian = random_quadratic_hamiltonian(n_qubits, False, True)

        # Obtain the circuit
        with pytest.raises(NotImplementedError):
            _ = gaussian_state_preparation_circuit(quadratic_hamiltonian, spin_sector=1)


# if __name__ == "__main__":
#     inst = GaussianStatePreparationCircuitTest()
#     inst.setUp()
#     inst.test_ground_state_particle_conserving()
