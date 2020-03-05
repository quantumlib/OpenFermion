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

import itertools
import os
import unittest
import numpy

from openfermion.config import THIS_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import get_fermion_operator
from openfermion.utils import (chemist_ordered, eigenspectrum,
                               get_chemist_two_body_coefficients,
                               is_hermitian,
                               low_rank_two_body_decomposition,
                               normal_ordered,
                               prepare_one_body_squared_evolution,
                               random_interaction_operator)


class ChemistTwoBodyTest(unittest.TestCase):

    def test_operator_consistency_w_spin(self):

        # Initialize a random InteractionOperator and FermionOperator.
        n_orbitals = 3
        n_qubits = 2 * n_orbitals
        random_interaction = random_interaction_operator(
                n_orbitals, expand_spin=True, real=False, seed=8)
        random_fermion = get_fermion_operator(random_interaction)

        # Convert to chemist ordered tensor.
        one_body_correction, chemist_tensor = \
            get_chemist_two_body_coefficients(
                random_interaction.two_body_tensor, spin_basis=True)

        # Convert output to FermionOperator.
        output_operator = FermionOperator(
            (), random_interaction.constant)

        # Convert one-body.
        one_body_coefficients = (random_interaction.one_body_tensor +
                                 one_body_correction)
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = one_body_coefficients[p, q]
            output_operator += FermionOperator(term, coefficient)

        # Convert two-body.
        for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
            coefficient = chemist_tensor[p, q, r, s]

            # Mixed spin.
            term = ((2 * p, 1), (2 * q, 0), (2 * r + 1, 1), (2 * s + 1, 0))
            output_operator += FermionOperator(term, coefficient)

            term = ((2 * p + 1, 1), (2 * q + 1, 0), (2 * r, 1), (2 * s, 0))
            output_operator += FermionOperator(term, coefficient)

            # Same spin.
            term = ((2 * p, 1), (2 * q, 0), (2 * r, 1), (2 * s, 0))
            output_operator += FermionOperator(term, coefficient)

            term = ((2 * p + 1, 1), (2 * q + 1, 0),
                    (2 * r + 1, 1), (2 * s + 1, 0))
            output_operator += FermionOperator(term, coefficient)

        # Check that difference is small.
        difference = normal_ordered(random_fermion - output_operator)
        self.assertAlmostEqual(0., difference.induced_norm())


class LowRankTest(unittest.TestCase):

    def test_random_operator_consistency(self):

        # Initialize a random InteractionOperator and FermionOperator.
        n_orbitals = 3
        n_qubits = 2 * n_orbitals
        random_interaction = random_interaction_operator(
                n_orbitals, expand_spin=True, real=True, seed=8)
        random_fermion = get_fermion_operator(random_interaction)

        # Decompose.
        eigenvalues, one_body_squares, one_body_correction, error = (
            low_rank_two_body_decomposition(
                random_interaction.two_body_tensor))
        self.assertFalse(error)

        # Build back operator constant and one-body components.
        decomposed_operator = FermionOperator((), random_interaction.constant)
        one_body_coefficients = (random_interaction.one_body_tensor +
                                 one_body_correction)
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = one_body_coefficients[p, q]
            decomposed_operator += FermionOperator(term, coefficient)

        # Build back two-body component.
        for l in range(n_orbitals ** 2):
            one_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_squares[l, p, q]
                one_body_operator += FermionOperator(term, coefficient)
            decomposed_operator += eigenvalues[l] * (one_body_operator ** 2)

        # Test for consistency.
        difference = normal_ordered(decomposed_operator - random_fermion)
        self.assertAlmostEqual(0., difference.induced_norm())

    def test_molecular_operator_consistency(self):

        # Initialize H2 InteractionOperator.
        n_qubits = 4
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        molecule = MolecularData(filename=filename)
        molecule_interaction = molecule.get_molecular_hamiltonian()
        molecule_operator = get_fermion_operator(molecule_interaction)

        constant = molecule_interaction.constant
        one_body_coefficients = molecule_interaction.one_body_tensor
        two_body_coefficients = molecule_interaction.two_body_tensor

        # Perform decomposition.
        eigenvalues, one_body_squares, one_body_corrections, trunc_error = (
            low_rank_two_body_decomposition(two_body_coefficients))
        self.assertAlmostEqual(trunc_error, 0.)

        # Build back operator constant and one-body components.
        decomposed_operator = FermionOperator((), constant)
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = (one_body_coefficients[p, q] +
                           one_body_corrections[p, q])
            decomposed_operator += FermionOperator(term, coefficient)

        # Build back two-body component.
        for l in range(one_body_squares.shape[0]):
            one_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_squares[l, p, q]
                if abs(eigenvalues[l]) > 1e-6:
                    self.assertTrue(is_hermitian(one_body_squares[l]))
                one_body_operator += FermionOperator(term, coefficient)
            decomposed_operator += eigenvalues[l] * (one_body_operator ** 2)

        # Test for consistency.
        difference = normal_ordered(decomposed_operator - molecule_operator)
        self.assertAlmostEqual(0., difference.induced_norm())

        # Decompose with slightly negative operator that must use eigen
        molecule = MolecularData(filename=filename)
        molecule.two_body_integrals[0, 0, 0, 0] -= 1

        eigenvalues, one_body_squares, one_body_corrections, trunc_error = (
            low_rank_two_body_decomposition(two_body_coefficients))
        self.assertAlmostEqual(trunc_error, 0.)

        # Check for property errors
        with self.assertRaises(TypeError):
            eigenvalues, one_body_squares, _, trunc_error = (
                low_rank_two_body_decomposition(
                    two_body_coefficients + 0.01j,
                    truncation_threshold=1.,
                    final_rank=1))

    def test_rank_reduction(self):

        # Initialize H2 InteractionOperator.
        n_qubits = 4
        n_orbitals = 2
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        molecule = MolecularData(filename=filename)
        molecule_interaction = molecule.get_molecular_hamiltonian()
        fermion_operator = get_fermion_operator(molecule_interaction)

        constant = molecule_interaction.constant
        two_body_coefficients = molecule_interaction.two_body_tensor

        # Rank reduce with threshold.
        errors = []
        for truncation_threshold in [1., 0.1, 0.01, 0.001]:

            # Decompose with threshold.
            (test_eigenvalues, one_body_squares, one_body_correction,
             trunc_error) = low_rank_two_body_decomposition(
                 two_body_coefficients,
                 truncation_threshold=truncation_threshold)

            # Make sure error is below truncation specification.
            self.assertTrue(trunc_error > 0.)
            self.assertTrue(trunc_error < truncation_threshold)
            self.assertTrue(len(test_eigenvalues) < n_orbitals ** 2)
            self.assertTrue(len(one_body_squares) == len(test_eigenvalues))

            # Build back operator constant and one-body components.
            decomposed_operator = FermionOperator((), constant)
            one_body_coefficients = (molecule_interaction.one_body_tensor +
                                     one_body_correction)
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_coefficients[p, q]
                decomposed_operator += FermionOperator(term, coefficient)

            # Build back two-body component.
            for l in range(len(test_eigenvalues)):
                one_body_operator = FermionOperator()
                for p, q in itertools.product(range(n_qubits), repeat=2):
                    term = ((p, 1), (q, 0))
                    coefficient = one_body_squares[l, p, q]
                    one_body_operator += FermionOperator(term, coefficient)
                decomposed_operator += (test_eigenvalues[l] *
                                        one_body_operator ** 2)

            # Test for consistency.
            difference = normal_ordered(
                decomposed_operator - fermion_operator)
            errors += [difference.induced_norm()]
            self.assertTrue(errors[-1] <= trunc_error)
        self.assertTrue(errors[3] <= errors[2] <= errors[1] <= errors[0])

        # Rank reduce by imposing final rank.
        errors = []
        for final_rank in [1, 2, 3, 4]:

            # Decompose with threshold.
            (test_eigenvalues, one_body_squares, one_body_correction,
             trunc_error) = low_rank_two_body_decomposition(
                 two_body_coefficients,
                 final_rank=final_rank)

            # Make sure error is below truncation specification.
            self.assertTrue(len(test_eigenvalues) == final_rank)

            # Build back operator constant and one-body components.
            decomposed_operator = FermionOperator((), constant)
            one_body_coefficients = (molecule_interaction.one_body_tensor +
                                     one_body_correction)
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_coefficients[p, q]
                decomposed_operator += FermionOperator(term, coefficient)

            # Build back two-body component.
            for l in range(final_rank):
                one_body_operator = FermionOperator()
                for p, q in itertools.product(range(n_qubits), repeat=2):
                    term = ((p, 1), (q, 0))
                    coefficient = one_body_squares[l, p, q]
                    one_body_operator += FermionOperator(term, coefficient)
                decomposed_operator += (test_eigenvalues[l] *
                                        one_body_operator ** 2)

            # Test for consistency.
            difference = normal_ordered(
                decomposed_operator - fermion_operator)
            errors += [difference.induced_norm()]
        self.assertTrue(errors[3] <= errors[2] <= errors[1] <= errors[0])

    def test_one_body_square_decomposition(self):

        # Initialize H2 InteractionOperator.
        n_qubits = 4
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        molecule = MolecularData(filename=filename)
        molecule_interaction = molecule.get_molecular_hamiltonian()
        get_fermion_operator(molecule_interaction)

        two_body_coefficients = molecule_interaction.two_body_tensor

        # Decompose.
        eigenvalues, one_body_squares, _, _ = (
            low_rank_two_body_decomposition(two_body_coefficients))
        rank = eigenvalues.size
        for l in range(rank):
            one_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_squares[l, p, q]
                one_body_operator += FermionOperator(term, coefficient)
            one_body_squared = one_body_operator ** 2

            # Get the squared one-body operator via one-body decomposition.
            if abs(eigenvalues[l]) < 1e-6:
                with self.assertRaises(ValueError):
                    prepare_one_body_squared_evolution(one_body_squares[l])
                continue
            else:
                density_density_matrix, basis_transformation_matrix = (
                    prepare_one_body_squared_evolution(one_body_squares[l]))
            two_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (p, 0), (q, 1), (q, 0))
                coefficient = density_density_matrix[p, q]
                two_body_operator += FermionOperator(term, coefficient)

            # Confirm that the rotations diagonalize the one-body squares.
            hopefully_diagonal = basis_transformation_matrix.dot(
                numpy.dot(one_body_squares[l],
                          numpy.transpose(numpy.conjugate(
                              basis_transformation_matrix))))
            diagonal = numpy.diag(hopefully_diagonal)
            difference = hopefully_diagonal - numpy.diag(diagonal)
            self.assertAlmostEqual(0., numpy.amax(numpy.absolute(difference)))
            density_density_alternative = numpy.outer(diagonal, diagonal)
            difference = density_density_alternative - density_density_matrix
            self.assertAlmostEqual(0., numpy.amax(numpy.absolute(difference)))

            # Test spectra.
            one_body_squared_spectrum = eigenspectrum(one_body_squared)
            two_body_spectrum = eigenspectrum(two_body_operator)
            difference = two_body_spectrum - one_body_squared_spectrum
            self.assertAlmostEqual(0., numpy.amax(numpy.absolute(difference)))
