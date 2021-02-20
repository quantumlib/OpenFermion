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
import os

import unittest

import itertools

import numpy
from numpy.linalg import multi_dot

from scipy.linalg import eigh, norm
from scipy.sparse import csc_matrix
import scipy
from scipy.special import comb

from openfermion.config import DATA_DIRECTORY
from openfermion.chem import MolecularData
from openfermion.hamiltonians import (fermi_hubbard, jellium_model,
                                      wigner_seitz_length_scale,
                                      number_operator)
from openfermion.ops.operators import (FermionOperator, BosonOperator,
                                       QuadOperator, QubitOperator,
                                       MajoranaOperator)
from openfermion.ops.representations import DiagonalCoulombHamiltonian
from openfermion.transforms.opconversions import (jordan_wigner,
                                                  get_fermion_operator,
                                                  normal_ordered)
from openfermion.transforms.repconversions import (fourier_transform,
                                                   get_interaction_operator)

from openfermion.testing.testing_utils import random_hermitian_matrix
from openfermion.linalg.sparse_tools import (
    get_sparse_operator, get_ground_state, eigenspectrum, expectation,
    jw_number_restrict_state, inner_product, jw_sz_restrict_state,
    jw_get_ground_state_at_particle_number, jw_sparse_givens_rotation,
    pauli_matrix_map, sparse_eigenspectrum, jordan_wigner_sparse,
    kronecker_operators, identity_csc, pauli_x_csc, qubit_operator_sparse,
    get_linear_qubit_operator_diagonal, jw_number_indices,
    get_number_preserving_sparse_operator, jw_configuration_state,
    jw_hartree_fock_state, jw_sz_restrict_operator, jw_sz_indices,
    jw_number_restrict_operator, variance,
    expectation_computational_basis_state,
    expectation_db_operator_with_pw_basis_state, get_gap, boson_ladder_sparse,
    boson_operator_sparse, single_quad_op_sparse, _iterate_basis_)

from openfermion.utils.operator_utils import (is_hermitian, count_qubits,
                                              hermitian_conjugated)
from openfermion.utils.indexing import up_index, down_index
from openfermion.utils.grid import Grid
from openfermion.hamiltonians.jellium_hf_state import (
    lowest_single_particle_energy_states)
from openfermion.linalg.linear_qubit_operator import LinearQubitOperator


class EigenSpectrumTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.majorana_operator = MajoranaOperator((1, 4, 9))
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(
            self.fermion_operator)

    def test_eigenspectrum(self):
        fermion_eigenspectrum = eigenspectrum(self.fermion_operator)
        qubit_eigenspectrum = eigenspectrum(self.qubit_operator)
        interaction_eigenspectrum = eigenspectrum(self.interaction_operator)
        for i in range(2**self.n_qubits):
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   qubit_eigenspectrum[i])
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   interaction_eigenspectrum[i])

        with self.assertRaises(TypeError):
            _ = eigenspectrum(BosonOperator())

        with self.assertRaises(TypeError):
            _ = eigenspectrum(QuadOperator())


class SparseOperatorTest(unittest.TestCase):

    def test_kronecker_operators(self):

        self.assertAlmostEqual(
            0,
            numpy.amax(
                numpy.absolute(
                    kronecker_operators(3 * [identity_csc]) -
                    kronecker_operators(3 * [pauli_x_csc])**2)))

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
        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(fermion_spectrum - qubit_spectrum)))


class JordanWignerSparseTest(unittest.TestCase):

    def test_jw_sparse_0create(self):
        expected = csc_matrix(([1], ([1], [0])), shape=(2, 2))
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator('0^')).A, expected.A))

    def test_jw_sparse_1annihilate(self):
        expected = csc_matrix(([1, -1], ([0, 2], [1, 3])), shape=(4, 4))
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator('1')).A, expected.A))

    def test_jw_sparse_0create_2annihilate(self):
        expected = csc_matrix(([-1j, 1j], ([4, 6], [1, 3])),
                              shape=(8, 8),
                              dtype=numpy.complex128)
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator('0^ 2', -1j)).A,
                expected.A))

    def test_jw_sparse_0create_3annihilate(self):
        expected = csc_matrix(
            ([-1j, 1j, 1j, -1j], ([8, 10, 12, 14], [1, 3, 5, 7])),
            shape=(16, 16),
            dtype=numpy.complex128)
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator('0^ 3', -1j)).A,
                expected.A))

    def test_jw_sparse_twobody(self):
        expected = csc_matrix(([1, 1], ([6, 14], [5, 13])), shape=(16, 16))
        self.assertTrue(
            numpy.allclose(
                jordan_wigner_sparse(FermionOperator('2^ 1^ 1 3')).A,
                expected.A))

    def test_qubit_operator_sparse_n_qubits_too_small(self):
        with self.assertRaises(ValueError):
            qubit_operator_sparse(QubitOperator('X3'), 1)

    def test_qubit_operator_sparse_n_qubits_not_specified(self):
        expected = csc_matrix(([1, 1, 1, 1], ([1, 0, 3, 2], [0, 1, 2, 3])),
                              shape=(4, 4))
        self.assertTrue(
            numpy.allclose(
                qubit_operator_sparse(QubitOperator('X1')).A, expected.A))

    def test_get_linear_qubit_operator_diagonal_wrong_n(self):
        """Testing with wrong n_qubits."""
        with self.assertRaises(ValueError):
            get_linear_qubit_operator_diagonal(QubitOperator('X3'), 1)

    def test_get_linear_qubit_operator_diagonal_0(self):
        """Testing with zero term."""
        qubit_operator = QubitOperator.zero()
        vec_expected = numpy.zeros(8)

        self.assertTrue(
            numpy.allclose(
                get_linear_qubit_operator_diagonal(qubit_operator, 3),
                vec_expected))

    def test_get_linear_qubit_operator_diagonal_zero(self):
        """Get zero diagonals from get_linear_qubit_operator_diagonal."""
        qubit_operator = QubitOperator('X0 Y1')
        vec_expected = numpy.zeros(4)

        self.assertTrue(
            numpy.allclose(get_linear_qubit_operator_diagonal(qubit_operator),
                           vec_expected))

    def test_get_linear_qubit_operator_diagonal_non_zero(self):
        """Get non zero diagonals from get_linear_qubit_operator_diagonal."""
        qubit_operator = QubitOperator('Z0 Z2')
        vec_expected = numpy.array([1, -1, 1, -1, -1, 1, -1, 1])

        self.assertTrue(
            numpy.allclose(get_linear_qubit_operator_diagonal(qubit_operator),
                           vec_expected))

    def test_get_linear_qubit_operator_diagonal_cmp_zero(self):
        """Compare get_linear_qubit_operator_diagonal with
            get_linear_qubit_operator."""
        qubit_operator = QubitOperator('Z1 X2 Y5')
        vec_expected = numpy.diag(
            LinearQubitOperator(qubit_operator) * numpy.eye(2**6))

        self.assertTrue(
            numpy.allclose(get_linear_qubit_operator_diagonal(qubit_operator),
                           vec_expected))

    def test_get_linear_qubit_operator_diagonal_cmp_non_zero(self):
        """Compare get_linear_qubit_operator_diagonal with
            get_linear_qubit_operator."""
        qubit_operator = QubitOperator('Z1 Z2 Z5')
        vec_expected = numpy.diag(
            LinearQubitOperator(qubit_operator) * numpy.eye(2**6))

        self.assertTrue(
            numpy.allclose(get_linear_qubit_operator_diagonal(qubit_operator),
                           vec_expected))


class ComputationalBasisStateTest(unittest.TestCase):

    def test_computational_basis_state(self):
        comp_basis_state = jw_configuration_state([0, 2, 5], 7)
        self.assertAlmostEqual(comp_basis_state[82], 1.)
        self.assertAlmostEqual(sum(comp_basis_state), 1.)


class JWHartreeFockStateTest(unittest.TestCase):

    def test_jw_hartree_fock_state(self):
        hartree_fock_state = jw_hartree_fock_state(3, 7)
        self.assertAlmostEqual(hartree_fock_state[112], 1.)
        self.assertAlmostEqual(sum(hartree_fock_state), 1.)


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

        def sz_integer(bitstring):
            """Computes the total number of occupied up sites
            minus the total number of occupied down sites."""
            n_sites = len(bitstring) // 2

            n_up = len([
                site for site in range(n_sites)
                if bitstring[up_index(site)] == '1'
            ])
            n_down = len([
                site for site in range(n_sites)
                if bitstring[down_index(site)] == '1'
            ])

            return n_up - n_down

        def jw_sz_indices_brute_force(sz_value, n_qubits):
            """Computes the correct indices by brute force."""
            indices = []
            for bitstring in itertools.product(['0', '1'], repeat=n_qubits):
                if (sz_integer(bitstring) == int(2 * sz_value)):
                    indices.append(int(''.join(bitstring), 2))

            return indices

        # General test
        n_sites = numpy.random.randint(1, 10)
        n_qubits = 2 * n_sites
        sz_int = ((-1)**numpy.random.randint(2) *
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

        correct_indices = [
            index for index in correct_indices
            if bin(index)[2:].count('1') == n_particles
        ]
        subspace_dimension = len(correct_indices)

        calculated_indices = jw_sz_indices(sz_value,
                                           n_qubits,
                                           n_electrons=n_particles)

        self.assertEqual(len(calculated_indices), subspace_dimension)

        for index in calculated_indices:
            binary_string = bin(index)[2:].zfill(n_qubits)
            self.assertEqual(sz_integer(binary_string), sz_int)
            self.assertEqual(binary_string.count('1'), n_particles)

        # Test exceptions
        with self.assertRaises(ValueError):
            jw_sz_indices(3, 3)

        with self.assertRaises(ValueError):
            jw_sz_indices(3.1, 4)

        with self.assertRaises(ValueError):
            jw_sz_indices(1.5, 8, n_electrons=6)

        with self.assertRaises(ValueError):
            jw_sz_indices(1.5, 8, n_electrons=1)


class JWNumberRestrictOperatorTest(unittest.TestCase):

    def test_jw_restrict_operator(self):
        """Test the scheme for restricting JW encoded operators to number"""
        # Make a Hamiltonian that cares mostly about number of electrons
        n_qubits = 4
        target_electrons = 2
        penalty_const = 10.
        number_sparse = jordan_wigner_sparse(number_operator(n_qubits))
        bias_sparse = jordan_wigner_sparse(
            sum([
                FermionOperator(((i, 1), (i, 0)), 1.0) for i in range(n_qubits)
            ], FermionOperator()))
        hamiltonian_sparse = penalty_const * (
            number_sparse -
            target_electrons * scipy.sparse.identity(2**n_qubits)
        ).dot(number_sparse - target_electrons *
              scipy.sparse.identity(2**n_qubits)) + bias_sparse

        restricted_hamiltonian = jw_number_restrict_operator(
            hamiltonian_sparse, target_electrons, n_qubits)
        true_eigvals, _ = eigh(hamiltonian_sparse.A)
        test_eigvals, _ = eigh(restricted_hamiltonian.A)

        self.assertAlmostEqual(norm(true_eigvals[:6] - test_eigvals[:6]), 0.0)

    def test_jw_restrict_operator_hopping_to_1_particle(self):
        hop = FermionOperator('3^ 1') + FermionOperator('1^ 3')
        hop_sparse = jordan_wigner_sparse(hop, n_qubits=4)
        hop_restrict = jw_number_restrict_operator(hop_sparse, 1, n_qubits=4)
        expected = csc_matrix(([1, 1], ([0, 2], [2, 0])), shape=(4, 4))

        self.assertTrue(numpy.allclose(hop_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_1_particle(self):
        interaction = FermionOperator('3^ 2^ 4 1')
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(interaction_sparse,
                                                           1,
                                                           n_qubits=6)
        expected = csc_matrix(([], ([], [])), shape=(6, 6))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_2_particles(self):
        interaction = (FermionOperator('3^ 2^ 4 1') +
                       FermionOperator('4^ 1^ 3 2'))
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(interaction_sparse,
                                                           2,
                                                           n_qubits=6)

        dim = 6 * 5 // 2  # shape of new sparse array

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

        dim = 6 * 5 // 2  # shape of new sparse array

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

        _, ground_state = get_ground_state(restricted_jellium_hamiltonian)

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
        interaction_restricted = jw_sz_restrict_operator(
            interaction_sparse, sz_value)
        restricted_interaction_values = set(
            [int(value.real) for value in interaction_restricted.diagonal()])
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
        vector = numpy.zeros(2**n_qubits, dtype=float)
        vector[number_indices] = 1

        # Restrict the vector
        restricted_vector = jw_number_restrict_state(vector, n_particles)

        # Check that it has the correct shape
        self.assertEqual(restricted_vector.shape[0], subspace_dimension)

        # Check that it has the same norm as the original vector
        self.assertAlmostEqual(
            inner_product(vector, vector),
            inner_product(restricted_vector, restricted_vector))


class JWSzRestrictStateTest(unittest.TestCase):

    def test_jw_sz_restrict_state(self):
        n_sites = numpy.random.randint(1, 10)
        n_qubits = 2 * n_sites
        sz_int = ((-1)**numpy.random.randint(2) *
                  numpy.random.randint(n_sites + 1))
        sz_value = sz_int / 2

        sz_indices = jw_sz_indices(sz_value, n_qubits)
        subspace_dimension = len(sz_indices)

        # Create a vector that has entry 1 for every coordinate in
        # the specified subspace, and 0 everywhere else
        vector = numpy.zeros(2**n_qubits, dtype=float)
        vector[sz_indices] = 1

        # Restrict the vector
        restricted_vector = jw_sz_restrict_state(vector, sz_value)

        # Check that it has the correct shape
        self.assertEqual(restricted_vector.shape[0], subspace_dimension)

        # Check that it has the same norm as the original vector
        self.assertAlmostEqual(
            inner_product(vector, vector),
            inner_product(restricted_vector, restricted_vector))


class JWGetGroundStatesByParticleNumberTest(unittest.TestCase):

    def test_jw_get_ground_state_at_particle_number_herm_conserving(self):
        # Initialize a particle-number-conserving Hermitian operator
        ferm_op = FermionOperator('0^ 1') + FermionOperator('1^ 0') + \
            FermionOperator('1^ 2') + FermionOperator('2^ 1') + \
            FermionOperator('1^ 3', -.4) + FermionOperator('3^ 1', -.4)
        jw_hamiltonian = jordan_wigner(ferm_op)
        sparse_operator = get_sparse_operator(jw_hamiltonian)
        n_qubits = 4

        num_op = get_sparse_operator(number_operator(n_qubits))

        # Test each possible particle number
        for particle_number in range(n_qubits):
            # Get the ground energy and ground state at this particle number
            energy, state = jw_get_ground_state_at_particle_number(
                sparse_operator, particle_number)

            # Check that it's an eigenvector with the correct eigenvalue
            self.assertTrue(
                numpy.allclose(sparse_operator.dot(state), energy * state))

            # Check that it has the correct particle number
            num = expectation(num_op, state)
            self.assertAlmostEqual(num, particle_number)

    def test_jw_get_ground_state_at_particle_number_hubbard(self):

        model = fermi_hubbard(2, 2, 1.0, 4.0)
        sparse_operator = get_sparse_operator(model)
        n_qubits = count_qubits(model)
        num_op = get_sparse_operator(number_operator(n_qubits))

        # Test each possible particle number
        for particle_number in range(n_qubits):
            # Get the ground energy and ground state at this particle number
            energy, state = jw_get_ground_state_at_particle_number(
                sparse_operator, particle_number)

            # Check that it's an eigenvector with the correct eigenvalue
            self.assertTrue(
                numpy.allclose(sparse_operator.dot(state), energy * state))

            # Check that it has the correct particle number
            num = expectation(num_op, state)
            self.assertAlmostEqual(num, particle_number)

    def test_jw_get_ground_state_at_particle_number_jellium(self):

        grid = Grid(2, 2, 1.0)
        model = jellium_model(grid, spinless=True, plane_wave=False)
        sparse_operator = get_sparse_operator(model)
        n_qubits = count_qubits(model)
        num_op = get_sparse_operator(number_operator(n_qubits))

        # Test each possible particle number
        for particle_number in range(n_qubits):
            # Get the ground energy and ground state at this particle number
            energy, state = jw_get_ground_state_at_particle_number(
                sparse_operator, particle_number)

            # Check that it's an eigenvector with the correct eigenvalue
            self.assertTrue(
                numpy.allclose(sparse_operator.dot(state), energy * state))

            # Check that it has the correct particle number
            num = expectation(num_op, state)
            self.assertAlmostEqual(num, particle_number)


class JWSparseGivensRotationTest(unittest.TestCase):

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            jw_sparse_givens_rotation(0, 2, 1., 1., 5)
        with self.assertRaises(ValueError):
            jw_sparse_givens_rotation(4, 5, 1., 1., 5)


class GroundStateTest(unittest.TestCase):

    def test_get_ground_state_hermitian(self):
        ground = get_ground_state(
            get_sparse_operator(
                QubitOperator('Y0 X1') + QubitOperator('Z0 Z1')))
        expected_state = csc_matrix(([1j, 1], ([1, 2], [0, 0])),
                                    shape=(4, 1),
                                    dtype=numpy.complex128).A
        expected_state /= numpy.sqrt(2.0)

        self.assertAlmostEqual(ground[0], -2)
        self.assertAlmostEqual(
            numpy.absolute(expected_state.T.conj().dot(ground[1]))[0], 1.)


class ExpectationTest(unittest.TestCase):

    def test_expectation_correct_sparse_matrix(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = numpy.array([0., 1.j, 0., 1.j])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

        density_matrix = scipy.sparse.csc_matrix(
            numpy.outer(vector, numpy.conjugate(vector)))
        self.assertAlmostEqual(expectation(operator, density_matrix), 2.0)

    def test_expectation_correct_linear_operator(self):
        operator = LinearQubitOperator(QubitOperator('X0'), n_qubits=2)
        vector = numpy.array([0., 1.j, 0., 1.j])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

    def test_expectation_handles_column_vector(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = numpy.array([[0.], [1.j], [0.], [1.j]])
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

    def test_expectation_correct_zero(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = numpy.array([1j, -1j, -1j, -1j])
        self.assertAlmostEqual(expectation(operator, vector), 0.0)

    def test_execptions(self):
        operator = LinearQubitOperator(QubitOperator('X0'), n_qubits=2)
        vector = scipy.sparse.csc_matrix(numpy.array([1, 0, 0, 0]))
        with self.assertRaises(ValueError):
            _ = expectation(operator, vector)
        with self.assertRaises(ValueError):
            _ = expectation(vector, operator)


class VarianceTest(unittest.TestCase):

    def test_variance_row_vector(self):

        X = pauli_matrix_map['X']
        Z = pauli_matrix_map['Z']
        zero = numpy.array([1., 0.])
        plus = numpy.array([1., 1.]) / numpy.sqrt(2)
        minus = numpy.array([1., -1.]) / numpy.sqrt(2)

        self.assertAlmostEqual(variance(Z, zero), 0.)
        self.assertAlmostEqual(variance(X, zero), 1.)

        self.assertAlmostEqual(variance(Z, plus), 1.)
        self.assertAlmostEqual(variance(X, plus), 0.)

        self.assertAlmostEqual(variance(Z, minus), 1.)
        self.assertAlmostEqual(variance(X, minus), 0.)

    def test_variance_column_vector(self):

        X = pauli_matrix_map['X']
        Z = pauli_matrix_map['Z']
        zero = numpy.array([[1.], [0.]])
        plus = numpy.array([[1.], [1.]]) / numpy.sqrt(2)
        minus = numpy.array([[1.], [-1.]]) / numpy.sqrt(2)

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
        self.n_spatial_orbitals = grid_length**dimension

        n_qubits = self.n_spatial_orbitals
        self.n_particles = 3

        # Compute appropriate length scale and the corresponding grid.
        length_scale = wigner_seitz_length_scale(wigner_seitz_radius,
                                                 self.n_particles, dimension)

        self.grid1 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid1, self.spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(
            lowest_single_particle_energy_states(hamiltonian, self.n_particles))
        self.hf_state_index1 = numpy.sum(2**occupied_states)

        self.hf_state1 = numpy.zeros(2**n_qubits)
        self.hf_state1[self.hf_state_index1] = 1.0

        self.orbital_occupations1 = [
            digit == '1' for digit in bin(self.hf_state_index1)[2:]
        ][::-1]
        self.occupied_orbitals1 = [
            index for index, occupied in enumerate(self.orbital_occupations1)
            if occupied
        ]

        self.reversed_occupied_orbitals1 = list(self.occupied_orbitals1)
        for i in range(len(self.reversed_occupied_orbitals1)):
            self.reversed_occupied_orbitals1[i] = -1 + int(
                numpy.log2(self.hf_state1.shape[0])
            ) - self.reversed_occupied_orbitals1[i]

        self.reversed_hf_state_index1 = sum(
            2**index for index in self.reversed_occupied_orbitals1)

    def test_1body_hopping_operator_1D(self):
        operator = FermionOperator('2^ 0')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(
            fourier_transform(operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(transformed_operator),
                               self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1, self.n_spatial_orbitals,
            self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_1body_number_operator_1D(self):
        operator = FermionOperator('2^ 2')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(
            fourier_transform(operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(transformed_operator),
                               self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1, self.n_spatial_orbitals,
            self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_2body_partial_number_operator_high_1D(self):
        operator = FermionOperator('2^ 1^ 2 0')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(
            fourier_transform(operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(transformed_operator),
                               self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1, self.n_spatial_orbitals,
            self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_2body_partial_number_operator_mid_1D(self):
        operator = FermionOperator('1^ 0^ 1 2')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(
            fourier_transform(operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(transformed_operator),
                               self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1, self.n_spatial_orbitals,
            self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_3body_double_number_operator_1D(self):
        operator = FermionOperator('3^ 2^ 1^ 3 1 0')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(
            fourier_transform(operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(transformed_operator),
                               self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1, self.n_spatial_orbitals,
            self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_2body_adjacent_number_operator_1D(self):
        operator = FermionOperator('3^ 2^ 2 1')
        operator = normal_ordered(operator)
        transformed_operator = normal_ordered(
            fourier_transform(operator, self.grid1, self.spinless))

        expected = expectation(get_sparse_operator(transformed_operator),
                               self.hf_state1)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals1, self.n_spatial_orbitals,
            self.grid1, self.spinless)
        self.assertAlmostEqual(expected, actual)

    def test_1d5_with_spin_10particles(self):
        dimension = 1
        grid_length = 5
        n_spatial_orbitals = grid_length**dimension
        wigner_seitz_radius = 9.3

        spinless = False
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 10

        length_scale = wigner_seitz_length_scale(wigner_seitz_radius,
                                                 n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(
            lowest_single_particle_energy_states(hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2**occupied_states)

        self.hf_state3 = csc_matrix(([1.0], ([self.hf_state_index3], [0])),
                                    shape=(2**n_qubits, 1))

        self.orbital_occupations3 = [
            digit == '1' for digit in bin(self.hf_state_index3)[2:]
        ][::-1]
        self.occupied_orbitals3 = [
            index for index, occupied in enumerate(self.orbital_occupations3)
            if occupied
        ]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(
                numpy.log2(self.hf_state3.shape[0])
            ) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2**index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('6^ 0^ 1^ 3 5 4', 2) +
                    FermionOperator('7^ 6^ 5 4', -3.7j) +
                    FermionOperator('3^ 3', 2.1) + FermionOperator('3^ 2', 1.7))
        operator = normal_ordered(operator)
        normal_ordered(fourier_transform(operator, self.grid3, spinless))

        expected = 2.1
        # Calculated from expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3, n_spatial_orbitals,
            self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)

    def test_1d5_with_spin_7particles(self):
        dimension = 1
        grid_length = 5
        n_spatial_orbitals = grid_length**dimension
        wigner_seitz_radius = 9.3

        spinless = False
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 7

        length_scale = wigner_seitz_length_scale(wigner_seitz_radius,
                                                 n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(
            lowest_single_particle_energy_states(hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2**occupied_states)

        self.hf_state3 = csc_matrix(([1.0], ([self.hf_state_index3], [0])),
                                    shape=(2**n_qubits, 1))

        self.orbital_occupations3 = [
            digit == '1' for digit in bin(self.hf_state_index3)[2:]
        ][::-1]
        self.occupied_orbitals3 = [
            index for index, occupied in enumerate(self.orbital_occupations3)
            if occupied
        ]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(
                numpy.log2(self.hf_state3.shape[0])
            ) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2**index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('6^ 0^ 1^ 3 5 4', 2) +
                    FermionOperator('7^ 2^ 4 1') +
                    FermionOperator('3^ 3', 2.1) +
                    FermionOperator('5^ 3^ 1 0', 7.3))
        operator = normal_ordered(operator)
        normal_ordered(fourier_transform(operator, self.grid3, spinless))

        expected = 1.66 - 0.0615536707435j
        # Calculated with expected = expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3, n_spatial_orbitals,
            self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)

    def test_3d2_spinless(self):
        dimension = 3
        grid_length = 2
        n_spatial_orbitals = grid_length**dimension
        wigner_seitz_radius = 9.3

        spinless = True
        n_qubits = n_spatial_orbitals
        n_particles_big = 5

        length_scale = wigner_seitz_length_scale(wigner_seitz_radius,
                                                 n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(
            lowest_single_particle_energy_states(hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2**occupied_states)

        self.hf_state3 = csc_matrix(([1.0], ([self.hf_state_index3], [0])),
                                    shape=(2**n_qubits, 1))

        self.orbital_occupations3 = [
            digit == '1' for digit in bin(self.hf_state_index3)[2:]
        ][::-1]
        self.occupied_orbitals3 = [
            index for index, occupied in enumerate(self.orbital_occupations3)
            if occupied
        ]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(
                numpy.log2(self.hf_state3.shape[0])
            ) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2**index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('4^ 2^ 3^ 5 5 4', 2) +
                    FermionOperator('7^ 6^ 7 4', -3.7j) +
                    FermionOperator('3^ 7', 2.1))
        operator = normal_ordered(operator)
        normal_ordered(fourier_transform(operator, self.grid3, spinless))
        expected = -0.2625 - 0.4625j
        # Calculated with expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3, n_spatial_orbitals,
            self.grid3, spinless)

        self.assertAlmostEqual(expected, actual)

    def test_3d2_with_spin(self):
        dimension = 3
        grid_length = 2
        n_spatial_orbitals = grid_length**dimension
        wigner_seitz_radius = 9.3

        spinless = False
        n_qubits = n_spatial_orbitals
        if not spinless:
            n_qubits *= 2
        n_particles_big = 9

        length_scale = wigner_seitz_length_scale(wigner_seitz_radius,
                                                 n_particles_big, dimension)

        self.grid3 = Grid(dimension, grid_length, length_scale)
        # Get the occupied orbitals of the plane-wave basis Hartree-Fock state.
        hamiltonian = jellium_model(self.grid3, spinless, plane_wave=True)
        hamiltonian = normal_ordered(hamiltonian)
        hamiltonian.compress()

        occupied_states = numpy.array(
            lowest_single_particle_energy_states(hamiltonian, n_particles_big))
        self.hf_state_index3 = numpy.sum(2**occupied_states)

        self.hf_state3 = csc_matrix(([1.0], ([self.hf_state_index3], [0])),
                                    shape=(2**n_qubits, 1))

        self.orbital_occupations3 = [
            digit == '1' for digit in bin(self.hf_state_index3)[2:]
        ][::-1]
        self.occupied_orbitals3 = [
            index for index, occupied in enumerate(self.orbital_occupations3)
            if occupied
        ]

        self.reversed_occupied_orbitals3 = list(self.occupied_orbitals3)
        for i in range(len(self.reversed_occupied_orbitals3)):
            self.reversed_occupied_orbitals3[i] = -1 + int(
                numpy.log2(self.hf_state3.shape[0])
            ) - self.reversed_occupied_orbitals3[i]

        self.reversed_hf_state_index3 = sum(
            2**index for index in self.reversed_occupied_orbitals3)

        operator = (FermionOperator('4^ 2^ 3^ 5 5 4', 2) +
                    FermionOperator('7^ 6^ 7 4', -3.7j) +
                    FermionOperator('3^ 7', 2.1))
        operator = normal_ordered(operator)
        normal_ordered(fourier_transform(operator, self.grid3, spinless))

        expected = -0.2625 - 0.578125j
        # Calculated from expected = expectation(get_sparse_operator(
        #    transformed_operator), self.hf_state3)
        actual = expectation_db_operator_with_pw_basis_state(
            operator, self.reversed_occupied_orbitals3, n_spatial_orbitals,
            self.grid3, spinless)

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
        state_1 = numpy.array([1., 1.j])
        state_2 = numpy.array([1., -1.j])

        self.assertAlmostEqual(inner_product(state_1, state_1), 2.)
        self.assertAlmostEqual(inner_product(state_1, state_2), 0.)


class BosonSparseTest(unittest.TestCase):

    def setUp(self):
        self.hbar = 1.
        self.d = 5
        self.b = numpy.diag(numpy.sqrt(numpy.arange(1, self.d)), 1)
        self.bd = self.b.conj().T
        self.q = numpy.sqrt(self.hbar / 2) * (self.b + self.bd)
        self.p = -1j * numpy.sqrt(self.hbar / 2) * (self.b - self.bd)
        self.Id = numpy.identity(self.d)

    def test_boson_ladder_noninteger_trunc(self):
        with self.assertRaises(ValueError):
            boson_ladder_sparse(1, 0, 0, 0.1)

        with self.assertRaises(ValueError):
            boson_ladder_sparse(1, 0, 0, -1)

        with self.assertRaises(ValueError):
            boson_ladder_sparse(1, 0, 0, 0)

    def test_boson_ladder_destroy_one_mode(self):
        b = boson_ladder_sparse(1, 0, 0, self.d).toarray()
        self.assertTrue(numpy.allclose(b, self.b))

    def test_boson_ladder_create_one_mode(self):
        bd = boson_ladder_sparse(1, 0, 1, self.d).toarray()
        self.assertTrue(numpy.allclose(bd, self.bd))

    def test_boson_ladder_single_adjoint(self):
        b = boson_ladder_sparse(1, 0, 0, self.d).toarray()
        bd = boson_ladder_sparse(1, 0, 1, self.d).toarray()
        self.assertTrue(numpy.allclose(b.conj().T, bd))

    def test_boson_ladder_two_mode(self):
        res = boson_ladder_sparse(2, 0, 0, self.d).toarray()
        expected = numpy.kron(self.b, self.Id)
        self.assertTrue(numpy.allclose(res, expected))

        res = boson_ladder_sparse(2, 1, 0, self.d).toarray()
        expected = numpy.kron(self.Id, self.b)
        self.assertTrue(numpy.allclose(res, expected))

    def test_single_quad_noninteger_trunc(self):
        with self.assertRaises(ValueError):
            single_quad_op_sparse(1, 0, 'q', self.hbar, 0.1)

        with self.assertRaises(ValueError):
            single_quad_op_sparse(1, 0, 'q', self.hbar, -1)

        with self.assertRaises(ValueError):
            single_quad_op_sparse(1, 0, 'q', self.hbar, 0)

    def test_single_quad_q_one_mode(self):
        res = single_quad_op_sparse(1, 0, 'q', self.hbar, self.d).toarray()
        self.assertTrue(numpy.allclose(res, self.q))
        self.assertTrue(numpy.allclose(res, res.conj().T))

    def test_single_quad_p_one_mode(self):
        res = single_quad_op_sparse(1, 0, 'p', self.hbar, self.d).toarray()
        self.assertTrue(numpy.allclose(res, self.p))
        self.assertTrue(numpy.allclose(res, res.conj().T))

    def test_single_quad_two_mode(self):
        res = single_quad_op_sparse(2, 0, 'q', self.hbar, self.d).toarray()
        expected = numpy.kron(self.q, self.Id)
        self.assertTrue(numpy.allclose(res, expected))

        res = single_quad_op_sparse(2, 1, 'p', self.hbar, self.d).toarray()
        expected = numpy.kron(self.Id, self.p)
        self.assertTrue(numpy.allclose(res, expected))

    def test_boson_operator_sparse_trunc(self):
        op = BosonOperator('0')
        with self.assertRaises(ValueError):
            boson_operator_sparse(op, 0.1)

        with self.assertRaises(ValueError):
            boson_operator_sparse(op, -1)

        with self.assertRaises(ValueError):
            boson_operator_sparse(op, 0)

    def test_boson_operator_invalid_op(self):
        op = FermionOperator('0')
        with self.assertRaises(ValueError):
            boson_operator_sparse(op, self.d)

    def test_boson_operator_sparse_empty(self):
        for op in (BosonOperator(), QuadOperator()):
            res = boson_operator_sparse(op, self.d)
            self.assertEqual(res, numpy.array([[0]]))

    def test_boson_operator_sparse_identity(self):
        for op in (BosonOperator(''), QuadOperator('')):
            res = boson_operator_sparse(op, self.d)
            self.assertEqual(res, numpy.array([[1]]))

    def test_boson_operator_sparse_single(self):
        op = BosonOperator('0')
        res = boson_operator_sparse(op, self.d).toarray()
        self.assertTrue(numpy.allclose(res, self.b))

        op = BosonOperator('0^')
        res = boson_operator_sparse(op, self.d).toarray()
        self.assertTrue(numpy.allclose(res, self.bd))

        op = QuadOperator('q0')
        res = boson_operator_sparse(op, self.d, self.hbar).toarray()
        self.assertTrue(numpy.allclose(res, self.q))

        op = QuadOperator('p0')
        res = boson_operator_sparse(op, self.d, self.hbar).toarray()
        self.assertTrue(numpy.allclose(res, self.p))

    def test_boson_operator_sparse_number(self):
        op = BosonOperator('0^ 0')
        res = boson_operator_sparse(op, self.d).toarray()
        self.assertTrue(numpy.allclose(res, numpy.dot(self.bd, self.b)))

    def test_boson_operator_sparse_multi_mode(self):
        op = BosonOperator('0^ 1 1^ 2')
        res = boson_operator_sparse(op, self.d).toarray()

        b0 = boson_ladder_sparse(3, 0, 0, self.d).toarray()
        b1 = boson_ladder_sparse(3, 1, 0, self.d).toarray()
        b2 = boson_ladder_sparse(3, 2, 0, self.d).toarray()

        expected = multi_dot([b0.T, b1, b1.T, b2])
        self.assertTrue(numpy.allclose(res, expected))

        op = QuadOperator('q0 p0 p1')
        res = boson_operator_sparse(op, self.d, self.hbar).toarray()

        expected = numpy.identity(self.d**2)
        for term in op.terms:
            for i, j in term:
                expected = expected.dot(
                    single_quad_op_sparse(2, i, j, self.hbar, self.d).toarray())
        self.assertTrue(numpy.allclose(res, expected))

    def test_boson_operator_sparse_addition(self):
        op = BosonOperator('0^ 1')
        op += BosonOperator('0 0^')
        res = boson_operator_sparse(op, self.d).toarray()

        b0 = boson_ladder_sparse(2, 0, 0, self.d).toarray()
        b1 = boson_ladder_sparse(2, 1, 0, self.d).toarray()

        expected = numpy.dot(b0.T, b1) + numpy.dot(b0, b0.T)
        self.assertTrue(numpy.allclose(res, expected))


class GetNumberPreservingSparseOperatorIntegrationTestLiH(unittest.TestCase):

    def setUp(self):
        # Set up molecule.
        geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(DATA_DIRECTORY, 'H1-Li1_sto-3g_singlet_1.45')
        self.molecule = MolecularData(geometry,
                                      basis,
                                      multiplicity,
                                      filename=filename)
        self.molecule.load()

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

        self.hubbard_hamiltonian = fermi_hubbard(2,
                                                 2,
                                                 1.0,
                                                 4.0,
                                                 chemical_potential=.2,
                                                 magnetic_field=0.0,
                                                 spinless=False)

    def test_exceptions(self):
        op = FermionOperator('1')
        with self.assertRaises(ValueError):
            _ = get_number_preserving_sparse_operator(op, 2, 1)

    def test_number_on_reference(self):
        sum_n_op = FermionOperator()
        sum_sparse_n_op = get_number_preserving_sparse_operator(
            sum_n_op,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        space_size = sum_sparse_n_op.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        for i in range(self.molecule.n_qubits):
            n_op = FermionOperator(((i, 1), (i, 0)))
            sum_n_op += n_op

            sparse_n_op = get_number_preserving_sparse_operator(
                n_op,
                self.molecule.n_qubits,
                self.molecule.n_electrons,
                spin_preserving=False)

            sum_sparse_n_op += sparse_n_op

            expectation = reference.dot(sparse_n_op.dot(reference))

            if i < self.molecule.n_electrons:
                assert expectation == 1.0
            else:
                assert expectation == 0.0

        convert_after_adding = get_number_preserving_sparse_operator(
            sum_n_op,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        assert scipy.sparse.linalg.norm(convert_after_adding -
                                        sum_sparse_n_op) < 1E-9

        assert reference.dot(sum_sparse_n_op.dot(reference)) - \
            self.molecule.n_electrons < 1E-9

    def test_space_size_correct(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        space_size = sparse_ham.shape[0]

        # Naive Hilbert space size is 2**12, or 4096.
        assert space_size == 225

    def test_hf_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        space_size = sparse_ham.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        sparse_hf_energy = reference.dot(sparse_ham.dot(reference))

        assert numpy.linalg.norm(sparse_hf_energy -
                                 self.molecule.hf_energy) < 1E-9

    def test_one_body_hf_energy(self):
        one_body_part = self.molecular_hamiltonian
        one_body_part.two_body_tensor = numpy.zeros_like(
            one_body_part.two_body_tensor)

        one_body_fop = get_fermion_operator(one_body_part)
        one_body_regular_sparse_op = get_sparse_operator(one_body_fop)

        make_hf_fop = FermionOperator(((3, 1), (2, 1), (1, 1), (0, 1)))
        make_hf_sparse_op = get_sparse_operator(make_hf_fop, n_qubits=12)

        hf_state = numpy.zeros((2**12))
        hf_state[0] = 1.0
        hf_state = make_hf_sparse_op.dot(hf_state)

        regular_sparse_hf_energy = \
            (hf_state.dot(one_body_regular_sparse_op.dot(hf_state))).real

        one_body_sparse_op = get_number_preserving_sparse_operator(
            one_body_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        space_size = one_body_sparse_op.shape[0]
        reference = numpy.zeros((space_size))
        reference[0] = 1.0

        sparse_hf_energy = reference.dot(one_body_sparse_op.dot(reference))

        assert numpy.linalg.norm(sparse_hf_energy -
                                 regular_sparse_hf_energy) < 1E-9

    def test_ground_state_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which='SA')

        assert numpy.abs(eig_val[0] - self.molecule.fci_energy) < 1E-9

    def test_doubles_are_subset(self):
        reference_determinants = [[
            True, True, True, True, False, False, False, False, False, False,
            False, False
        ],
                                  [
                                      True, True, False, False, False, False,
                                      True, True, False, False, False, False
                                  ]]

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=2,
                                    spin_preserving=True)))
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1])

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=4,
                                    spin_preserving=True)))
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1])

            for item in doubles_int_state_array:
                assert item in all_int_state_array

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=2,
                                    spin_preserving=True)))
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1])

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=4,
                                    spin_preserving=False)))
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1])

            for item in doubles_int_state_array:
                assert item in all_int_state_array

        for reference_determinant in reference_determinants:
            reference_determinant = numpy.asarray(reference_determinant)
            doubles_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=2,
                                    spin_preserving=False)))
            doubles_int_state_array = doubles_state_array.dot(
                1 << numpy.arange(doubles_state_array.shape[1])[::-1])

            all_state_array = numpy.asarray(
                list(
                    _iterate_basis_(reference_determinant,
                                    excitation_level=4,
                                    spin_preserving=False)))
            all_int_state_array = all_state_array.dot(
                1 << numpy.arange(all_state_array.shape[1])[::-1])

            for item in doubles_int_state_array:
                assert item in all_int_state_array

    def test_full_ham_hermitian(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_full_ham_hermitian_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_singles_simple_one_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (1, 0)))
        fop_conj = FermionOperator(((1, 1), (3, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_op -
                                        sparse_op_conj.getH()) < 1E-9

    def test_singles_simple_two_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (8, 1), (1, 0), (4, 0)))
        fop_conj = FermionOperator(((4, 1), (1, 1), (8, 0), (3, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_op -
                                        sparse_op_conj.getH()) < 1E-9

    def test_singles_repeating_two_body_term_hermitian(self):
        fop = FermionOperator(((3, 1), (1, 1), (5, 0), (1, 0)))
        fop_conj = FermionOperator(((5, 1), (1, 1), (3, 0), (1, 0)))

        sparse_op = get_number_preserving_sparse_operator(
            fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        sparse_op_conj = get_number_preserving_sparse_operator(
            fop_conj,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_op -
                                        sparse_op_conj.getH()) < 1E-9

    def test_singles_ham_hermitian(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_singles_ham_hermitian_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
            excitation_level=1)

        assert scipy.sparse.linalg.norm(sparse_ham - sparse_ham.getH()) < 1E-9

    def test_cisd_energy(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True,
            excitation_level=2)

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which='SA')

        assert numpy.abs(eig_val[0] - self.molecule.cisd_energy) < 1E-9

    def test_cisd_energy_non_spin_preserving(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False,
            excitation_level=2)

        eig_val, _ = scipy.sparse.linalg.eigsh(sparse_ham, k=1, which='SA')

        assert numpy.abs(eig_val[0] - self.molecule.cisd_energy) < 1E-9

    def test_cisd_matches_fci_energy_two_electron_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_cisd = get_number_preserving_sparse_operator(
            hamiltonian_fop, 8, 2, spin_preserving=True, excitation_level=2)

        sparse_ham_fci = get_sparse_operator(hamiltonian_fop, n_qubits=8)

        eig_val_cisd, _ = scipy.sparse.linalg.eigsh(sparse_ham_cisd,
                                                    k=1,
                                                    which='SA')
        eig_val_fci, _ = scipy.sparse.linalg.eigsh(sparse_ham_fci,
                                                   k=1,
                                                   which='SA')

        assert numpy.abs(eig_val_cisd[0] - eig_val_fci[0]) < 1E-9

    def test_weird_determinant_matches_fci_energy_two_electron_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_cisd = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            8,
            2,
            spin_preserving=True,
            excitation_level=2,
            reference_determinant=numpy.asarray(
                [False, False, True, True, False, False, False, False]))

        sparse_ham_fci = get_sparse_operator(hamiltonian_fop, n_qubits=8)

        eig_val_cisd, _ = scipy.sparse.linalg.eigsh(sparse_ham_cisd,
                                                    k=1,
                                                    which='SA')
        eig_val_fci, _ = scipy.sparse.linalg.eigsh(sparse_ham_fci,
                                                   k=1,
                                                   which='SA')

        assert numpy.abs(eig_val_cisd[0] - eig_val_fci[0]) < 1E-9

    def test_number_restricted_spectra_match_molecule(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham_number_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=False)

        sparse_ham = get_sparse_operator(hamiltonian_fop,
                                         self.molecule.n_qubits)

        sparse_ham_restricted_number_preserving = jw_number_restrict_operator(
            sparse_ham,
            n_electrons=self.molecule.n_electrons,
            n_qubits=self.molecule.n_qubits)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_preserving)

        spectral_deviation = numpy.amax(
            numpy.absolute(spectrum_from_new_sparse_method -
                           spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)

    def test_number_restricted_spectra_match_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_number_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop, 8, 4, spin_preserving=False)

        sparse_ham = get_sparse_operator(hamiltonian_fop, 8)

        sparse_ham_restricted_number_preserving = jw_number_restrict_operator(
            sparse_ham, n_electrons=4, n_qubits=8)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_preserving)

        spectral_deviation = numpy.amax(
            numpy.absolute(spectrum_from_new_sparse_method -
                           spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)

    def test_number_sz_restricted_spectra_match_molecule(self):
        hamiltonian_fop = get_fermion_operator(self.molecular_hamiltonian)

        sparse_ham_number_sz_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop,
            self.molecule.n_qubits,
            self.molecule.n_electrons,
            spin_preserving=True)

        sparse_ham = get_sparse_operator(hamiltonian_fop,
                                         self.molecule.n_qubits)

        sparse_ham_restricted_number_sz_preserving = jw_sz_restrict_operator(
            sparse_ham,
            0,
            n_electrons=self.molecule.n_electrons,
            n_qubits=self.molecule.n_qubits)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_sz_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_sz_preserving)

        spectral_deviation = numpy.amax(
            numpy.absolute(spectrum_from_new_sparse_method -
                           spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)

    def test_number_sz_restricted_spectra_match_hubbard(self):
        hamiltonian_fop = self.hubbard_hamiltonian

        sparse_ham_number_sz_preserving = get_number_preserving_sparse_operator(
            hamiltonian_fop, 8, 4, spin_preserving=True)

        sparse_ham = get_sparse_operator(hamiltonian_fop, 8)

        sparse_ham_restricted_number_sz_preserving = jw_sz_restrict_operator(
            sparse_ham, 0, n_electrons=4, n_qubits=8)

        spectrum_from_new_sparse_method = sparse_eigenspectrum(
            sparse_ham_number_sz_preserving)

        spectrum_from_old_sparse_method = sparse_eigenspectrum(
            sparse_ham_restricted_number_sz_preserving)

        spectral_deviation = numpy.amax(
            numpy.absolute(spectrum_from_new_sparse_method -
                           spectrum_from_old_sparse_method))
        self.assertAlmostEqual(spectral_deviation, 0.)


class GetSparseOperatorQubitTest(unittest.TestCase):

    def test_sparse_matrix_Y(self):
        term = QubitOperator(((0, 'Y'),))
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [1j, -1j])
        self.assertEqual(list(sparse_operator.indices), [1, 0])
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_ZX(self):
        coefficient = 2.
        operators = ((0, 'Z'), (1, 'X'))
        term = QubitOperator(operators, coefficient)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data), [2., 2., -2., -2.])
        self.assertEqual(list(sparse_operator.indices), [1, 0, 3, 2])
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_ZIZ(self):
        operators = ((0, 'Z'), (2, 'Z'))
        term = QubitOperator(operators)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.data),
                         [1, -1, 1, -1, -1, 1, -1, 1])
        self.assertEqual(list(sparse_operator.indices), list(range(8)))
        self.assertTrue(is_hermitian(sparse_operator))

    def test_sparse_matrix_combo(self):
        qop = (QubitOperator(((0, 'Y'), (1, 'X')), -0.1j) + QubitOperator(
            ((0, 'X'), (1, 'Z')), 3. + 2.j))
        sparse_operator = get_sparse_operator(qop)

        self.assertEqual(
            list(sparse_operator.data),
            [3 + 2j, 0.1, 0.1, -3 - 2j, 3 + 2j, -0.1, -0.1, -3 - 2j])
        self.assertEqual(list(sparse_operator.indices),
                         [2, 3, 2, 3, 0, 1, 0, 1])

    def test_sparse_matrix_zero_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator((), 0.0), 1)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (2, 2))

    def test_sparse_matrix_zero_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator((), 0.0), 5)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (32, 32))

    def test_sparse_matrix_identity_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(()), 1)
        self.assertEqual(list(sparse_operator.data), [1] * 2)
        self.assertEqual(sparse_operator.shape, (2, 2))

    def test_sparse_matrix_identity_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(()), 5)
        self.assertEqual(list(sparse_operator.data), [1] * 32)
        self.assertEqual(sparse_operator.shape, (32, 32))

    def test_sparse_matrix_linearity(self):
        identity = QubitOperator(())
        zzzz = QubitOperator(tuple((i, 'Z') for i in range(4)), 1.0)

        sparse1 = get_sparse_operator(identity + zzzz)
        sparse2 = get_sparse_operator(identity, 4) + get_sparse_operator(zzzz)

        self.assertEqual(list(sparse1.data), [2] * 8)
        self.assertEqual(list(sparse1.indices), [0, 3, 5, 6, 9, 10, 12, 15])
        self.assertEqual(list(sparse2.data), [2] * 8)
        self.assertEqual(list(sparse2.indices), [0, 3, 5, 6, 9, 10, 12, 15])


class GetSparseOperatorFermionTest(unittest.TestCase):

    def test_sparse_matrix_zero_n_qubit(self):
        sparse_operator = get_sparse_operator(FermionOperator.zero(), 4)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.data)), 0)
        self.assertEqual(sparse_operator.shape, (16, 16))


class GetSparseOperatorBosonTest(unittest.TestCase):

    def setUp(self):
        self.hbar = 1.
        self.d = 4
        self.b = numpy.diag(numpy.sqrt(numpy.arange(1, self.d)), 1)
        self.bd = self.b.conj().T
        self.q = numpy.sqrt(self.hbar / 2) * (self.b + self.bd)

    def test_sparse_matrix_ladder(self):
        sparse_operator = get_sparse_operator(BosonOperator('0'), trunc=self.d)
        self.assertTrue(numpy.allclose(sparse_operator.toarray(), self.b))
        self.assertEqual(sparse_operator.shape, (self.d, self.d))

    def test_sparse_matrix_quad(self):
        sparse_operator = get_sparse_operator(QuadOperator('q0'), trunc=self.d)
        self.assertTrue(numpy.allclose(sparse_operator.toarray(), self.q))
        self.assertEqual(sparse_operator.shape, (self.d, self.d))

    def test_sparse_matrix_error(self):
        with self.assertRaises(TypeError):
            _ = get_sparse_operator(1)


class GetSparseOperatorDiagonalCoulombHamiltonianTest(unittest.TestCase):

    def test_diagonal_coulomb_hamiltonian(self):
        n_qubits = 5
        one_body = random_hermitian_matrix(n_qubits, real=False)
        two_body = random_hermitian_matrix(n_qubits, real=True)
        constant = numpy.random.randn()
        op = DiagonalCoulombHamiltonian(one_body, two_body, constant)

        op1 = get_sparse_operator(op)
        op2 = get_sparse_operator(jordan_wigner(get_fermion_operator(op)))
        diff = op1 - op2
        discrepancy = 0.
        if diff.nnz:
            discrepancy = max(abs(diff.data))
        self.assertAlmostEqual(discrepancy, 0.)
