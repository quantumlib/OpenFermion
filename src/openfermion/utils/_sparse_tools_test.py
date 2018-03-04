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

"""Tests for sparse_tools.py."""
from __future__ import absolute_import

import numpy
import unittest

from scipy.linalg import eigh, norm
from scipy.sparse import csc_matrix
from scipy.special import comb

from openfermion.hamiltonians import (fermi_hubbard, jellium_model,
                                      wigner_seitz_length_scale)
from openfermion.ops import FermionOperator, normal_ordered
from openfermion.transforms import (get_fermion_operator, get_sparse_operator,
                                    jordan_wigner)
from openfermion.utils import (Grid, fourier_transform, number_operator,
                               up_index, down_index) 
from openfermion.utils._jellium_hf_state import (
    lowest_single_particle_energy_states)
from openfermion.utils._slater_determinants_test import (
    random_quadratic_hamiltonian)
from openfermion.utils._sparse_tools import *


class SparseOperatorTest(unittest.TestCase):

    def test_kronecker_operators(self):

        self.assertAlmostEqual(
            0, numpy.amax(numpy.absolute(
                kronecker_operators(3 * [identity_csc]) -
                kronecker_operators(3 * [pauli_x_csc]) ** 2)))

    def test_qubit_jw_fermion_integration(self):

        # Initialize a random fermionic operator.
        fermion_operator = FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)),
                                           -4.3)
        fermion_operator += FermionOperator(((3, 1), (1, 0)), 8.17)
        fermion_operator += 3.2 * FermionOperator()

        # Map to qubits and compare matrix versions.
        qubit_operator = jordan_wigner(fermion_operator)
        qubit_sparse = get_sparse_operator(qubit_operator)
        qubit_spectrum = sparse_eigenspectrum(qubit_sparse)
        fermion_sparse = jordan_wigner_sparse(fermion_operator)
        fermion_spectrum = sparse_eigenspectrum(fermion_sparse)
        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(fermion_spectrum - qubit_spectrum)))


class JordanWignerSparseTest(unittest.TestCase):

    def test_jw_sparse_0create(self):
        expected = csc_matrix(([1], ([1], [0])), shape=(2, 2))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('0^')).A,
            expected.A))

    def test_jw_sparse_1annihilate(self):
        expected = csc_matrix(([1, -1], ([0, 2], [1, 3])), shape=(4, 4))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('1')).A,
            expected.A))

    def test_jw_sparse_0create_2annihilate(self):
        expected = csc_matrix(([-1j, 1j],
                               ([4, 6], [1, 3])),
                              shape=(8, 8))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('0^ 2', -1j)).A,
            expected.A))

    def test_jw_sparse_0create_3annihilate(self):
        expected = csc_matrix(([-1j, 1j, 1j, -1j],
                               ([8, 10, 12, 14], [1, 3, 5, 7])),
                              shape=(16, 16))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('0^ 3', -1j)).A,
            expected.A))

    def test_jw_sparse_twobody(self):
        expected = csc_matrix(([1, 1], ([6, 14], [5, 13])), shape=(16, 16))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('2^ 1^ 1 3')).A,
            expected.A))

    def test_qubit_operator_sparse_n_qubits_too_small(self):
        with self.assertRaises(ValueError):
            qubit_operator_sparse(QubitOperator('X3'), 1)

    def test_qubit_operator_sparse_n_qubits_not_specified(self):
        expected = csc_matrix(([1, 1, 1, 1], ([1, 0, 3, 2], [0, 1, 2, 3])),
                              shape=(4, 4))
        self.assertTrue(numpy.allclose(
            qubit_operator_sparse(QubitOperator('X1')).A,
            expected.A))


class ComputationalBasisStateTest(unittest.TestCase):
    def test_computational_basis_state(self):
        comp_basis_state = jw_configuration_state([0, 2, 5], 7)
        dense_array = comp_basis_state.toarray()
        self.assertAlmostEqual(dense_array[82, 0], 1.)
        self.assertAlmostEqual(sum(dense_array), 1.)


class JWHartreeFockStateTest(unittest.TestCase):
    def test_jw_hartree_fock_state(self):
        hartree_fock_state = jw_hartree_fock_state(3, 7)
        dense_array = hartree_fock_state.toarray()
        self.assertAlmostEqual(dense_array[112, 0], 1.)
        self.assertAlmostEqual(sum(dense_array), 1.)


class JWNumberIndicesTest(unittest.TestCase):

    def test_jw_sparse_index(self):
        """Test the indexing scheme for selecting specific particle numbers"""
        expected = [1, 2]
        calculated_indices = jw_number_indices(1, 2)
        self.assertEqual(expected, calculated_indices)

        expected = [3]
        calculated_indices = jw_number_indices(2, 2)
        self.assertEqual(expected, calculated_indices)

    def test_jw_number_indices(self):
        n_qubits = numpy.random.randint(1, 12)
        n_particles = numpy.random.randint(n_qubits + 1)

        number_indices = jw_number_indices(n_particles, n_qubits)
        subspace_dimension = len(number_indices)

        self.assertEqual(subspace_dimension, comb(n_qubits, n_particles))

        for index in number_indices:
            binary_string = bin(index)[2:].zfill(n_qubits)
            n_ones = binary_string.count('1')
            self.assertEqual(n_ones, n_particles)


class JWSzIndicesTest(unittest.TestCase):

    def test_jw_sz_indices(self):
        """Test the indexing scheme for selecting specific sz value"""

        def sz_integer(bitstring, up_map=up_index, down_map=down_index):
            """Computes the total number of occupied up sites
            minus the total number of occupied down sites."""
            n_sites = len(bitstring) // 2

            n_up = len([site for site in range(n_sites)
                        if bitstring[up_map(site)] == '1'])
            n_down = len([site for site in range(n_sites)
                          if bitstring[down_map(site)] == '1'])

            return n_up - n_down

        def jw_sz_indices_brute_force(sz_value, n_qubits,
                                      up_map=up_index, down_map=down_index):
            """Computes the correct indices by brute force."""
            indices = []
            for bitstring in itertools.product(['0', '1'], repeat=n_qubits):
                if (sz_integer(bitstring, up_map, down_map) ==
                        int(2 * sz_value)):
                    indices.append(int(''.join(bitstring), 2))

            return indices

        # General test
        n_sites = numpy.random.randint(1, 10)
        n_qubits = 2 * n_sites
        sz_int = ((-1) ** numpy.random.randint(2) *
                  numpy.random.randint(n_sites + 1))
        sz_value = sz_int / 2.

        correct_indices = jw_sz_indices_brute_force(sz_value, n_qubits)
        subspace_dimension = len(correct_indices)

        calculated_indices = jw_sz_indices(sz_value, n_qubits)

        self.assertEqual(len(calculated_indices), subspace_dimension)

        for index in calculated_indices:
            binary_string = bin(index)[2:].zfill(n_qubits)
            self.assertEqual(sz_integer(binary_string), sz_int)

        # Test fixing particle number
        n_particles = abs(sz_int)

        correct_indices = [index for index in correct_indices
                           if bin(index)[2:].count('1') == n_particles]
        subspace_dimension = len(correct_indices)

        calculated_indices = jw_sz_indices(sz_value, n_qubits,
                                           n_electrons=n_particles)

        self.assertEqual(len(calculated_indices), subspace_dimension)

        for index in calculated_indices:
            binary_string = bin(index)[2:].zfill(n_qubits)
            self.assertEqual(sz_integer(binary_string), sz_int)
            self.assertEqual(binary_string.count('1'), n_particles)

        # Test exceptions
        with self.assertRaises(ValueError):
            indices = jw_sz_indices(3, 3)

        with self.assertRaises(ValueError):
            indices = jw_sz_indices(3.1, 4)

        with self.assertRaises(ValueError):
            indices = jw_sz_indices(1.5, 8, n_electrons=6)

        with self.assertRaises(ValueError):
            indices = jw_sz_indices(1.5, 8, n_electrons=1)


class JWNumberRestrictOperatorTest(unittest.TestCase):

    def test_jw_restrict_operator(self):
        """Test the scheme for restricting JW encoded operators to number"""
        # Make a Hamiltonian that cares mostly about number of electrons
        n_qubits = 6
        target_electrons = 3
        penalty_const = 100.
        number_sparse = jordan_wigner_sparse(number_operator(n_qubits))
        bias_sparse = jordan_wigner_sparse(
            sum([FermionOperator(((i, 1), (i, 0)), 1.0) for i
                 in range(n_qubits)], FermionOperator()))
        hamiltonian_sparse = penalty_const * (
            number_sparse - target_electrons *
            scipy.sparse.identity(2**n_qubits)).dot(
            number_sparse - target_electrons *
            scipy.sparse.identity(2**n_qubits)) + bias_sparse

        restricted_hamiltonian = jw_number_restrict_operator(
            hamiltonian_sparse, target_electrons, n_qubits)
        true_eigvals, _ = eigh(hamiltonian_sparse.A)
        test_eigvals, _ = eigh(restricted_hamiltonian.A)

        self.assertAlmostEqual(norm(true_eigvals[:20] - test_eigvals[:20]),
                               0.0)

    def test_jw_restrict_operator_hopping_to_1_particle(self):
        hop = FermionOperator('3^ 1') + FermionOperator('1^ 3')
        hop_sparse = jordan_wigner_sparse(hop, n_qubits=4)
        hop_restrict = jw_number_restrict_operator(hop_sparse, 1, n_qubits=4)
        expected = csc_matrix(([1, 1], ([0, 2], [2, 0])), shape=(4, 4))

        self.assertTrue(numpy.allclose(hop_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_1_particle(self):
        interaction = FermionOperator('3^ 2^ 4 1')
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 1, n_qubits=6)
        expected = csc_matrix(([], ([], [])), shape=(6, 6))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_2_particles(self):
        interaction = (FermionOperator('3^ 2^ 4 1') +
                       FermionOperator('4^ 1^ 3 2'))
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 2, n_qubits=6)

        dim = 6 * 5 / 2  # shape of new sparse array
        # 3^ 2^ 4 1 maps 2**4 + 2 = 18 to 2**3 + 2**2 = 12 and vice versa;
        # in the 2-particle subspace (1, 4) and (2, 3) are 7th and 9th.
        expected = csc_matrix(([-1, -1], ([7, 9], [9, 7])), shape=(dim, dim))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_hopping_to_1_particle_default_nqubits(self):
        interaction = (FermionOperator('3^ 2^ 4 1') +
                       FermionOperator('4^ 1^ 3 2'))
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        # n_qubits should default to 6
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 2)

        dim = 6 * 5 / 2  # shape of new sparse array
        # 3^ 2^ 4 1 maps 2**4 + 2 = 18 to 2**3 + 2**2 = 12 and vice versa;
        # in the 2-particle subspace (1, 4) and (2, 3) are 7th and 9th.
        expected = csc_matrix(([-1, -1], ([7, 9], [9, 7])), shape=(dim, dim))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_jellium_ground_state_integration(self):
        n_qubits = 4
        grid = Grid(dimensions=1, length=n_qubits, scale=1.0)
        jellium_hamiltonian = jordan_wigner_sparse(
            jellium_model(grid, spinless=False))

        #  2 * n_qubits because of spin
        number_sparse = jordan_wigner_sparse(number_operator(2 * n_qubits))

        restricted_number = jw_number_restrict_operator(number_sparse, 2)
        restricted_jellium_hamiltonian = jw_number_restrict_operator(
            jellium_hamiltonian, 2)

        energy, ground_state = get_ground_state(restricted_jellium_hamiltonian)

        number_expectation = expectation(restricted_number, ground_state)
        self.assertAlmostEqual(number_expectation, 2)


class JWSzRestrictOperatorTest(unittest.TestCase):

    def test_restrict_interaction_hamiltonian(self):
        """Test restricting a coulomb repulsion Hamiltonian to a specified
        Sz manifold."""
        x_dim = 3
        y_dim = 2

        interaction_term = fermi_hubbard(x_dim, y_dim, 0., 1.)
        interaction_sparse = get_sparse_operator(interaction_term)
        sz_value = 2
        interaction_restricted = jw_sz_restrict_operator(interaction_sparse,
                                                         sz_value)
        restricted_interaction_values = set([
            int(value.real) for value in interaction_restricted.diagonal()])
        # Originally the eigenvalues run from 0 to 6 but after restricting,
        # they should run from 0 to 2
        self.assertEqual(restricted_interaction_values, {0, 1, 2})


class JWNumberRestrictStateTest(unittest.TestCase):

    def test_jw_number_restrict_state(self):
        n_qubits = numpy.random.randint(1, 12)
        n_particles = numpy.random.randint(0, n_qubits)

        number_indices = jw_number_indices(n_particles, n_qubits)
        subspace_dimension = len(number_indices)

        # Create a vector that has entry 1 for every coordinate with
        # the specified particle number, and 0 everywhere else
        vector = csc_matrix(
                    ([1.] * subspace_dimension,
                     number_indices,
                     [0, 1]),
                    shape=(2 ** n_qubits, 1))

        # Restrict the vector
        restricted_vector = jw_number_restrict_state(vector, n_particles)

        # Check that it has the correct shape
        self.assertEqual(restricted_vector.shape[0], subspace_dimension)

        # Check that it has the same norm as the original vector
        self.assertAlmostEqual(inner_product(vector, vector),
                               inner_product(restricted_vector,
                                             restricted_vector))


class JWSzRestrictStateTest(unittest.TestCase):

    def test_jw_sz_restrict_state(self):
        n_sites = numpy.random.randint(1, 10)
        n_qubits = 2 * n_sites
        sz_int = ((-1) ** numpy.random.randint(2) *
                  numpy.random.randint(n_sites + 1))
        sz_value = sz_int / 2

        sz_indices = jw_sz_indices(sz_value, n_qubits)
        subspace_dimension = len(sz_indices)

        # Create a vector that has entry 1 for every coordinate in
        # the specified subspace, and 0 everywhere else
        vector = csc_matrix(
                    ([1.] * subspace_dimension,
                     sz_indices,
                     [0, 1]),
                    shape=(2 ** n_qubits, 1))

        # Restrict the vector
        restricted_vector = jw_sz_restrict_state(vector, sz_value)

        # Check that it has the correct shape
        self.assertEqual(restricted_vector.shape[0], subspace_dimension)

        # Check that it has the same norm as the original vector
        self.assertAlmostEqual(inner_product(vector, vector),
                               inner_product(restricted_vector,
                                             restricted_vector))


class JWGetGroundStatesByParticleNumberTest(unittest.TestCase):
    def test_jw_get_ground_states_by_particle_number_herm_conserving(self):
        # Initialize a particle-number-conserving Hermitian operator
        ferm_op = FermionOperator('0^ 1') + FermionOperator('1^ 0') + \
            FermionOperator('1^ 2') + FermionOperator('2^ 1') + \
            FermionOperator('1^ 3', -.4) + FermionOperator('3^ 1', -.4)
        jw_hamiltonian = jordan_wigner(ferm_op)
        sparse_operator = get_sparse_operator(jw_hamiltonian)
        n_qubits = 4

        # Test each possible particle number
        for particle_number in range(n_qubits):
            # Get the ground energy and ground states at this particle number
            energy, states = jw_get_ground_states_by_particle_number(
                sparse_operator, particle_number)
            # Construct particle number operator
            num_op = get_sparse_operator(number_operator(n_qubits))
            # For each vector returned, make sure that it is indeed an
            # eigenvector of the original operator with the returned eigenvalue
            # and that it has the correct particle number
            for vec in states:
                # Check that it's an eigenvector with the correct eigenvalue
                op_vec_product = sparse_operator.dot(vec)
                difference = op_vec_product - energy * vec
                discrepancy = 0.
                if difference.nnz:
                    discrepancy = max(map(abs, difference.data))
                self.assertAlmostEqual(0., discrepancy)
                # Check that it has the correct particle number
                num = expectation(num_op, vec)
                self.assertAlmostEqual(num, particle_number)

    def test_jw_get_ground_states_by_particle_number_herm_nonconserving(self):
        # Initialize a non-particle-number-conserving Hermitian operator
        ferm_op = FermionOperator('0^ 1') + FermionOperator('1^ 0') + \
            FermionOperator('1^ 2^') + FermionOperator('2 1')
        jw_hamiltonian = jordan_wigner(ferm_op)
        sparse_operator = get_sparse_operator(jw_hamiltonian)

        with self.assertRaises(ValueError):
            jw_get_ground_states_by_particle_number(sparse_operator, 0)

    def test_get_ground_states_by_particle_number_nonhermitian(self):
        # Initialize a non-Hermitian operator
        ferm_op = FermionOperator('0^ 1') + FermionOperator('2^ 1')
        jw_hamiltonian = jordan_wigner(ferm_op)
        sparse_operator = get_sparse_operator(jw_hamiltonian)

        with self.assertRaises(ValueError):
            jw_get_ground_states_by_particle_number(sparse_operator, 0)


class JWGetGaussianStateTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits_range = range(2, 10)

    def test_ground_state_particle_conserving(self):
        """Test getting the ground state of a Hamiltonian that conserves
        particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, True)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, ground_state = get_ground_state(sparse_operator)

            # Compute the ground state using the circuit
            circuit_energy, circuit_state = jw_get_gaussian_state(
                quadratic_hamiltonian)

            # Check that the energies match
            self.assertAlmostEqual(ground_energy, circuit_energy)

            # Check that the state obtained using the circuit is a ground state
            difference = (sparse_operator * circuit_state -
                          ground_energy * circuit_state)
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_ground_state_particle_nonconserving(self):
        """Test getting the ground state of a Hamiltonian that does not
        conserve particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a non-particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, False)

            # Compute the true ground state
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            ground_energy, ground_state = get_ground_state(sparse_operator)

            # Compute the ground state using the circuit
            circuit_energy, circuit_state = (
                jw_get_gaussian_state(quadratic_hamiltonian))

            # Check that the energies match
            self.assertAlmostEqual(ground_energy, circuit_energy)

            # Check that the state obtained using the circuit is a ground state
            difference = (sparse_operator * circuit_state -
                          ground_energy * circuit_state)
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_excited_state_particle_conserving(self):
        """Test getting an excited state of a Hamiltonian that conserves
        particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, True)

            # Pick some orbitals to occupy
            num_occupied_orbitals = numpy.random.randint(1, n_qubits + 1)
            occupied_orbitals = numpy.random.choice(
                range(n_qubits), num_occupied_orbitals, False)

            # Compute the Gaussian state
            circuit_energy, gaussian_state = jw_get_gaussian_state(
                quadratic_hamiltonian, occupied_orbitals)

            # Compute the true energy
            orbital_energies, constant = (
                quadratic_hamiltonian.orbital_energies())
            energy = numpy.sum(orbital_energies[occupied_orbitals]) + constant

            # Check that the energies match
            self.assertAlmostEqual(energy, circuit_energy)

            # Check that the state obtained using the circuit is an eigenstate
            # with the correct eigenvalue
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            difference = (sparse_operator * gaussian_state -
                          energy * gaussian_state)
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_excited_state_particle_nonconserving(self):
        """Test getting an excited state of a Hamiltonian that conserves
        particle number."""
        for n_qubits in self.n_qubits_range:
            # Initialize a non-particle-number-conserving Hamiltonian
            quadratic_hamiltonian = random_quadratic_hamiltonian(
                n_qubits, False)

            # Pick some orbitals to occupy
            num_occupied_orbitals = numpy.random.randint(1, n_qubits + 1)
            occupied_orbitals = numpy.random.choice(
                range(n_qubits), num_occupied_orbitals, False)

            # Compute the Gaussian state
            circuit_energy, gaussian_state = jw_get_gaussian_state(
                quadratic_hamiltonian, occupied_orbitals)

            # Compute the true energy
            orbital_energies, constant = (
                quadratic_hamiltonian.orbital_energies())
            energy = numpy.sum(orbital_energies[occupied_orbitals]) + constant

            # Check that the energies match
            self.assertAlmostEqual(energy, circuit_energy)

            # Check that the state obtained using the circuit is an eigenstate
            # with the correct eigenvalue
            sparse_operator = get_sparse_operator(quadratic_hamiltonian)
            difference = (sparse_operator * gaussian_state -
                          energy * gaussian_state)
            discrepancy = 0.
            if difference.nnz:
                discrepancy = max(abs(difference.data))

            self.assertTrue(discrepancy < EQ_TOLERANCE)

    def test_bad_input(self):
        """Test bad input."""
        with self.assertRaises(ValueError):
            energy, state = jw_get_gaussian_state('a')


class JWSparseGivensRotationTest(unittest.TestCase):

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            givens_matrix = jw_sparse_givens_rotation(0, 2, 1., 1., 5)
        with self.assertRaises(ValueError):
            givens_matrix = jw_sparse_givens_rotation(4, 5, 1., 1., 5)


class JWSlaterDeterminantTest(unittest.TestCase):

    def test_hadamard_transform(self):
        """Test creating the states
        1 / sqrt(2) (a^\dagger_0 + a^\dagger_1) |vac>
        and
        1 / sqrt(2) (a^\dagger_0 - a^\dagger_1) |vac>.
        """
        slater_determinant_matrix = numpy.array([[1., 1.]]) / numpy.sqrt(2.)
        slater_determinant = jw_slater_determinant(slater_determinant_matrix)
        self.assertAlmostEqual(slater_determinant[1, 0],
                               slater_determinant[2, 0])
        self.assertAlmostEqual(abs(slater_determinant[1, 0]),
                               1. / numpy.sqrt(2.))
        self.assertAlmostEqual(abs(slater_determinant[0, 0]), 0.)
        self.assertAlmostEqual(abs(slater_determinant[3, 0]), 0.)

        slater_determinant_matrix = numpy.array([[1., -1.]]) / numpy.sqrt(2.)
        slater_determinant = jw_slater_determinant(slater_determinant_matrix)
        self.assertAlmostEqual(slater_determinant[1, 0],
                               -slater_determinant[2, 0])
        self.assertAlmostEqual(abs(slater_determinant[1, 0]),
                               1. / numpy.sqrt(2.))
        self.assertAlmostEqual(abs(slater_determinant[0, 0]), 0.)
        self.assertAlmostEqual(abs(slater_determinant[3, 0]), 0.)


class GroundStateTest(unittest.TestCase):
    def test_get_ground_state_hermitian(self):
        ground = get_ground_state(get_sparse_operator(
            QubitOperator('Y0 X1') + QubitOperator('Z0 Z1')))
        expected_state = csc_matrix(([1j, 1], ([1, 2], [0, 0])),
                                    shape=(4, 1)).A
        expected_state /= numpy.sqrt(2.0)

        self.assertAlmostEqual(ground[0], -2)
        self.assertAlmostEqual(
            numpy.absolute(
                expected_state.T.conj().dot(ground[1]))[0], 1.)

    def test_get_ground_state_nonhermitian(self):
        with self.assertRaises(ValueError):
            get_ground_state(get_sparse_operator(1j * QubitOperator('X1')))


class ExpectationTest(unittest.TestCase):
    def test_expectation_correct(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = csc_matrix(([1j, 1j], ([1, 3], [0, 0])), shape=(4, 1))
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

    def test_expectation_correct_zero(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = csc_matrix(([1j, -1j, -1j, -1j],
                             ([0, 1, 2, 3], [0, 0, 0, 0])), shape=(4, 1))
        self.assertAlmostEqual(expectation(operator, vector), 0.0)

    def test_expectation_invalid_state_length(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = csc_matrix(([1j, -1j, -1j],
                             ([0, 1, 2], [0, 0, 0])), shape=(3, 1))
        with self.assertRaises(ValueError):
            expectation(operator, vector)


class VarianceTest(unittest.TestCase):
    def test_variance(self):
        X = pauli_matrix_map['X']
        Z = pauli_matrix_map['Z']
        zero = csc_matrix(numpy.array([[1.], [0.]]))
        plus = csc_matrix(numpy.array([[1.], [1.]]) / numpy.sqrt(2))
        minus = csc_matrix(numpy.array([[1.], [-1.]]) / numpy.sqrt(2))

        self.assertAlmostEqual(variance(Z, zero), 0.)
        self.assertAlmostEqual(variance(X, zero), 1.)

        self.assertAlmostEqual(variance(Z, plus), 1.)
        self.assertAlmostEqual(variance(X, plus), 0.)

        self.assertAlmostEqual(variance(Z, minus), 1.)
        self.assertAlmostEqual(variance(X, minus), 0.)


class ExpectationComputationalBasisStateTest(unittest.TestCase):
    def test_expectation_fermion_operator_single_number_terms(self):
        operator = FermionOperator('3^ 3', 1.9) + FermionOperator('2^ 1')
        state = csc_matrix(([1], ([15], [0])), shape=(16, 1))

        self.assertAlmostEqual(
            expectation_computational_basis_state(operator, state), 1.9)

    def test_expectation_fermion_operator_two_number_terms(self):
        operator = (FermionOperator('2^ 2', 1.9) + FermionOperator('2^ 1') +
                    FermionOperator('2^ 1^ 2 1', -1.7))
        state = csc_matrix(([1], ([6], [0])), shape=(16, 1))

        self.assertAlmostEqual(
            expectation_computational_basis_state(operator, state), 3.6)

    def test_expectation_identity_fermion_operator(self):
        operator = FermionOperator.identity() * 1.1
        state = csc_matrix(([1], ([6], [0])), shape=(16, 1))

        self.assertAlmostEqual(
            expectation_computational_basis_state(operator, state), 1.1)

    def test_expectation_state_is_list_single_number_terms(self):
        operator = FermionOperator('3^ 3', 1.9) + FermionOperator('2^ 1')
        state = [1, 1, 1, 1]

        self.assertAlmostEqual(
            expectation_computational_basis_state(operator, state), 1.9)

    def test_expectation_state_is_list_fermion_operator_two_number_terms(self):
        operator = (FermionOperator('2^ 2', 1.9) + FermionOperator('2^ 1') +
                    FermionOperator('2^ 1^ 2 1', -1.7))
        state = [0, 1, 1]

        self.assertAlmostEqual(
            expectation_computational_basis_state(operator, state), 3.6)

    def test_expectation_state_is_list_identity_fermion_operator(self):
        operator = FermionOperator.identity() * 1.1
        state = [0, 1, 1]

        self.assertAlmostEqual(
            expectation_computational_basis_state(operator, state), 1.1)

    def test_expectation_bad_operator_type(self):
        with self.assertRaises(TypeError):
            expectation_computational_basis_state(
                'never', csc_matrix(([1], ([6], [0])), shape=(16, 1)))

    def test_expectation_qubit_operator_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            expectation_computational_basis_state(
                QubitOperator(), csc_matrix(([1], ([6], [0])), shape=(16, 1)))


class ExpectationDualBasisOperatorWithPlaneWaveBasisState(unittest.TestCase):
    def setUp(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.
        self.spinless = True
        self.n_spatial_orbitals = grid_length ** dimension

        n_qubits = self.n_spatial_orbitals
        self.n_particles = 3

        # Compute appropriate length scale and the corresponding grid.
        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, self.n_particles, dimension)

        self.grid1 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid1, self.spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(lowest_single_particle_energy_states(
            hamiltonian, self.n_particles))
        self.hf_state_index1 = numpy.sum(2 ** occupied_states)

        self.hf_state1 = csc_matrix(
            ([1.0], ([self.hf_state_index1], [0])), shape=(2 ** n_qubits, 1))

        self.orbital_occupations1 = [digit == '1' for digit in
                                     bin(self.hf_state_index1)[2:]][::-1]
        self.occupied_orbitals1 = [index for index, occupied in
                                   enumerate(self.orbital_occupations1)
                                   if occupied]

        self.reversed_occupied_orbitals1 = list(self.occupied_orbitals1)
        for i in range(len(self.reversed_occupied_orbitals1)):
            self.reversed_occupied_orbitals1[i] = -1 + int(numpy.log2(
                self.hf_state1.shape[0])) - self.reversed_occupied_orbitals1[i]

        self.reversed_hf_state_index1 = sum(
            2 ** index for index in self.reversed_occupied_orbitals1)

    def test_1body_hopping_operator_1D(self):
        operator = FermionOperator('2^ 0')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(
            transformed_operator), self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1,
            self.n_spatial_orbitals, self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_1body_number_operator_1D(self):
        operator = FermionOperator('2^ 2')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(
            transformed_operator), self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1,
            self.n_spatial_orbitals, self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_2body_partial_number_operator_high_1D(self):
        operator = FermionOperator('2^ 1^ 2 0')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(
            transformed_operator), self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1,
            self.n_spatial_orbitals, self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_2body_partial_number_operator_mid_1D(self):
        operator = FermionOperator('1^ 0^ 1 2')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(
            transformed_operator), self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1,
            self.n_spatial_orbitals, self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_3body_double_number_operator_1D(self):
        operator = FermionOperator('3^ 2^ 1^ 3 1 0')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(
            transformed_operator), self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1,
            self.n_spatial_orbitals, self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_2body_adjacent_number_operator_1D(self):
        operator = FermionOperator('3^ 2^ 2 1')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(
            transformed_operator), self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1,
            self.n_spatial_orbitals, self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_1d5_with_spin_10particles(self):
        dimension = 1
        grid_length = 5
        n_spatial_orbitals = grid_length ** dimension
        wigner_seitz_radius = 9.3

        spinless = False
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 10

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(lowest_single_particle_energy_states(
            hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2 ** occupied_states)

        self.hf_state3 = csc_matrix(
            ([1.0], ([self.hf_state_index3], [0])), shape=(2 ** n_qubits, 1))

        self.orbital_occupations3 = [digit == '1' for digit in
                                     bin(self.hf_state_index3)[2:]][::-1]
        self.occupied_orbitals3 = [index for index, occupied in
                                   enumerate(self.orbital_occupations3)
                                   if occupied]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(numpy.log2(
                self.hf_state3.shape[0])) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2 ** index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('6^ 0^ 1^ 3 5 4', 2) +
                    FermionOperator('7^ 6^ 5 4', -3.7j) +
                    FermionOperator('3^ 3', 2.1) +
                    FermionOperator('3^ 2', 1.7))
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid3, spinless))

        expected = 2.1
        # Calculated from expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3,
            n_spatial_orbitals, self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)

    def test_1d5_with_spin_7particles(self):
        dimension = 1
        grid_length = 5
        n_spatial_orbitals = grid_length ** dimension
        wigner_seitz_radius = 9.3

        spinless = False
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 7

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(lowest_single_particle_energy_states(
            hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2 ** occupied_states)

        self.hf_state3 = csc_matrix(
            ([1.0], ([self.hf_state_index3], [0])), shape=(2 ** n_qubits, 1))

        self.orbital_occupations3 = [digit == '1' for digit in
                                     bin(self.hf_state_index3)[2:]][::-1]
        self.occupied_orbitals3 = [index for index, occupied in
                                   enumerate(self.orbital_occupations3)
                                   if occupied]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(numpy.log2(
                self.hf_state3.shape[0])) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2 ** index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('6^ 0^ 1^ 3 5 4', 2) +
                    FermionOperator('7^ 2^ 4 1') +
                    FermionOperator('3^ 3', 2.1) +
                    FermionOperator('5^ 3^ 1 0', 7.3))
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid3, spinless))

        expected = 1.66 - 0.0615536707435j
        # Calculated with expected = expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3,
            n_spatial_orbitals, self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)

    def test_3d2_spinless(self):
        dimension = 3
        grid_length = 2
        n_spatial_orbitals = grid_length ** dimension
        wigner_seitz_radius = 9.3

        spinless = True
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 5

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(lowest_single_particle_energy_states(
            hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2 ** occupied_states)

        self.hf_state3 = csc_matrix(
            ([1.0], ([self.hf_state_index3], [0])), shape=(2 ** n_qubits, 1))

        self.orbital_occupations3 = [digit == '1' for digit in
                                     bin(self.hf_state_index3)[2:]][::-1]
        self.occupied_orbitals3 = [index for index, occupied in
                                   enumerate(self.orbital_occupations3)
                                   if occupied]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(numpy.log2(
                self.hf_state3.shape[0])) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2 ** index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('4^ 2^ 3^ 5 5 4', 2) +
                    FermionOperator('7^ 6^ 7 4', -3.7j) +
                    FermionOperator('3^ 7', 2.1))
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid3, spinless))

        expected = -0.2625 - 0.4625j
        # Calculated with expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3,
            n_spatial_orbitals, self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)

    def test_3d2_with_spin(self):
        dimension = 3
        grid_length = 2
        n_spatial_orbitals = grid_length ** dimension
        wigner_seitz_radius = 9.3

        spinless = False
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 9

        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(lowest_single_particle_energy_states(
            hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2 ** occupied_states)

        self.hf_state3 = csc_matrix(
            ([1.0], ([self.hf_state_index3], [0])), shape=(2 ** n_qubits, 1))

        self.orbital_occupations3 = [digit == '1' for digit in
                                     bin(self.hf_state_index3)[2:]][::-1]
        self.occupied_orbitals3 = [index for index, occupied in
                                   enumerate(self.orbital_occupations3)
                                   if occupied]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(numpy.log2(
                self.hf_state3.shape[0])) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2 ** index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('4^ 2^ 3^ 5 5 4', 2) +
                    FermionOperator('7^ 6^ 7 4', -3.7j) +
                    FermionOperator('3^ 7', 2.1))
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(fourier_transform(
            operator, self.grid3, spinless))

        expected = -0.2625 - 0.578125j
        # Calculated from expected = expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3,
            n_spatial_orbitals, self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)


class GetGapTest(unittest.TestCase):
    def test_get_gap(self):
        operator = QubitOperator('Y0 X1') + QubitOperator('Z0 Z1')
        self.assertAlmostEqual(get_gap(get_sparse_operator(operator)), 2.0)

    def test_get_gap_nonhermitian_error(self):
        operator = (QubitOperator('X0 Y1', 1 + 1j) +
                    QubitOperator('Z0 Z1', 1j) + QubitOperator((), 2 + 1j))
        with self.assertRaises(ValueError):
            get_gap(get_sparse_operator(operator))


class InnerProductTest(unittest.TestCase):
    def test_inner_product(self):
        state_1 = csc_matrix(([1., 1.j], ([0, 1], [0, 0])), shape=(2, 1))
        state_2 = csc_matrix(([1., -1.j], ([0, 1], [0, 0])), shape=(2, 1))

        self.assertAlmostEqual(inner_product(state_1, state_1), 2.)
        self.assertAlmostEqual(inner_product(state_1, state_2), 0.)
