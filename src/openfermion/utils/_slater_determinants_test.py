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

import numpy

from openfermion.config import EQ_TOLERANCE
from openfermion.transforms import get_sparse_operator
from openfermion.utils import (jw_configuration_state,
                               get_ground_state)

from openfermion.utils._sparse_tools import (
    jw_sparse_givens_rotation,
    jw_sparse_particle_hole_transformation_last_mode)
from openfermion.utils._testing_utils import random_quadratic_hamiltonian

from openfermion.utils._slater_determinants import (
    gaussian_state_preparation_circuit)


class GaussianStatePreparationCircuitTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits_range = range(3, 6)

    def test_ground_state_particle_conserving(self):
        """Test getting the ground state preparation circuit for a Hamiltonian
        that conserves particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, True, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, _ = get_ground_state(sparse_operator)

            # Obtain the circuit
            circuit_description, start_orbitals = (
                gaussian_state_preparation_circuit(quadratic_hamiltonian))

            # Initialize the starting state
            state = jw_configuration_state(start_orbitals, n_qubits)

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

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * state - ground_energy * state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_ground_state_particle_nonconserving(self):
        """Test getting the ground state preparation circuit for a Hamiltonian
        that does not conserve particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, False, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, _ = get_ground_state(sparse_operator)

            # Obtain the circuit
            circuit_description, start_orbitals = (
                gaussian_state_preparation_circuit(quadratic_hamiltonian))

            # Initialize the starting state
            state = jw_configuration_state(start_orbitals, n_qubits)

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

            # Check that the state obtained using the circuit is a ground state
            difference = sparse_operator * state - ground_energy * state
            discrepancy = numpy.amax(numpy.abs(difference))
            self.assertAlmostEqual(discrepancy, 0)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            gaussian_state_preparation_circuit('a')
