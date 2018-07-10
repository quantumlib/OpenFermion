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
import numpy
import os
import unittest

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

    def test_operator_consistency(self):

        # Initialize a random InteractionOperator and FermionOperator.
        n_qubits = 4
        random_interaction = random_interaction_operator(
                n_qubits, real=False, seed=34281)
        random_fermion = get_fermion_operator(random_interaction)

        # Convert to chemist ordered tensor.
        io_constant, io_one_body_coefficients, io_chemist_tensor = \
            get_chemist_two_body_coefficients(random_interaction)
        fo_constant, fo_one_body_coefficients, fo_chemist_tensor = \
            get_chemist_two_body_coefficients(random_fermion)

        # Ensure consistency between FermionOperator and InteractionOperator.
        self.assertAlmostEqual(io_constant, fo_constant)

        one_body_difference = numpy.sum(numpy.absolute(
            io_one_body_coefficients - fo_one_body_coefficients))
        self.assertAlmostEqual(0., one_body_difference)

        two_body_difference = numpy.sum(numpy.absolute(
            io_chemist_tensor - fo_chemist_tensor))
        self.assertAlmostEqual(0., two_body_difference)

        # Convert output to FermionOperator.
        output_operator = FermionOperator()
        output_operator += FermionOperator((), fo_constant)

        # Convert one-body.
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = fo_one_body_coefficients[p, q]
            output_operator += FermionOperator(term, coefficient)

        # Convert two-body.
        for p, q, r, s in itertools.product(range(n_qubits), repeat=4):
            term = ((p, 1), (q, 0), (r, 1), (s, 0))
            coefficient = fo_chemist_tensor[p, q, r, s]
            output_operator += FermionOperator(term, coefficient)

        # Check that difference is small.
        difference = normal_ordered(random_fermion - output_operator)
        self.assertAlmostEqual(0., difference.induced_norm())

    def test_exception(self):

        # Initialize a bad FermionOperator.
        n_qubits = 4
        random_interaction = random_interaction_operator(n_qubits, seed=36229)
        random_fermion = get_fermion_operator(random_interaction)
        bad_term = ((1, 1), (2, 1))
        random_fermion += FermionOperator(bad_term)

        # Check for exception.
        with self.assertRaises(TypeError):
            fo_constant, fo_one_body_coefficients, fo_chemist_tensor = (
                get_chemist_two_body_coefficients(random_fermion))


class LowRankTest(unittest.TestCase):

    def test_operator_consistency(self):

        # Initialize a random two-body FermionOperator.
        n_qubits = 4
        random_operator = get_fermion_operator(
            random_interaction_operator(n_qubits, seed=28644))

        # Convert to chemist tensor.
        constant, one_body_coefficients, chemist_tensor = (
            get_chemist_two_body_coefficients(random_operator))

        # Build back operator constant and one-body components.
        decomposed_operator = FermionOperator((), constant)
        for p, q in itertools.product(range(n_qubits), repeat=2):
            term = ((p, 1), (q, 0))
            coefficient = one_body_coefficients[p, q]
            decomposed_operator += FermionOperator(term, coefficient)

        # Perform decomposition.
        eigenvalues, one_body_squares, trunc_error = (
            low_rank_two_body_decomposition(chemist_tensor))
        self.assertFalse(trunc_error)

        # Check for exception.
        with self.assertRaises(ValueError):
            eigenvalues, one_body_squares, trunc_error = (
                low_rank_two_body_decomposition(chemist_tensor,
                                                truncation_threshold=1.,
                                                final_rank=1))

        # Build back two-body component.
        for l in range(n_qubits ** 2):
            one_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_squares[l, p, q]
                one_body_operator += FermionOperator(term, coefficient)
            decomposed_operator += eigenvalues[l] * (one_body_operator ** 2)

        # Test for consistency.
        difference = normal_ordered(decomposed_operator - random_operator)
        self.assertAlmostEqual(0., difference.induced_norm())

    def test_rank_reduction(self):

        # Initialize H2.
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        molecule.load()

        # Get molecular Hamiltonian.
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()

        # Get fermion Hamiltonian.
        fermion_hamiltonian = normal_ordered(get_fermion_operator(
            molecular_hamiltonian))

        # Get chemist tensor.
        constant, one_body_coefficients, chemist_tensor = (
            get_chemist_two_body_coefficients(fermion_hamiltonian))
        n_qubits = one_body_coefficients.shape[0]

        # Rank reduce with threshold.
        errors = []
        for truncation_threshold in [1., 0.1, 0.01, 0.001]:

            # Add back one-body terms and constant.
            decomposed_operator = FermionOperator((), constant)
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_coefficients[p, q]
                decomposed_operator += FermionOperator(term, coefficient)

            # Rank reduce.
            eigenvalues, one_body_squares, trunc_error = (
                low_rank_two_body_decomposition(
                    chemist_tensor,
                    truncation_threshold=truncation_threshold))

            # Make sure error is below truncation specification.
            self.assertTrue(trunc_error < truncation_threshold)

            # Reassemble FermionOperator.
            l_max = eigenvalues.size
            for l in range(l_max):
                one_body_operator = FermionOperator()
                for p, q in itertools.product(range(n_qubits), repeat=2):
                    term = ((p, 1), (q, 0))
                    coefficient = one_body_squares[l, p, q]
                    one_body_operator += FermionOperator(term, coefficient)
                decomposed_operator += eigenvalues[l] * (one_body_operator**2)

            # Test for consistency.
            difference = normal_ordered(
                decomposed_operator - fermion_hamiltonian)
            errors += [difference.induced_norm()]
            self.assertTrue(errors[-1] <= trunc_error or
                            abs(errors[-1] - trunc_error) < 1e-6)
        self.assertTrue(errors[3] <= errors[2] <= errors[1] <= errors[0])

        # Rank reduce by setting final rank.
        errors = []
        for final_rank in [4, 6, 8, 10, 12]:

            # Add back one-body terms and constant.
            decomposed_operator = FermionOperator((), constant)
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_coefficients[p, q]
                decomposed_operator += FermionOperator(term, coefficient)

            # Rank reduce.
            eigenvalues, one_body_squares, trunc_error = (
                low_rank_two_body_decomposition(
                    chemist_tensor, final_rank=final_rank))

            # Reassemble FermionOperator.
            l_max = eigenvalues.size
            for l in range(l_max):
                one_body_operator = FermionOperator()
                for p, q in itertools.product(range(n_qubits), repeat=2):
                    term = ((p, 1), (q, 0))
                    coefficient = one_body_squares[l, p, q]
                    one_body_operator += FermionOperator(term, coefficient)
                decomposed_operator += eigenvalues[l] * (one_body_operator**2)

            # Test for consistency.
            difference = normal_ordered(
                decomposed_operator - fermion_hamiltonian)
            errors += [difference.induced_norm()]
        self.assertTrue(errors[3] <= errors[2] <= errors[1] <= errors[0])

    def test_one_body_square_decomposition(self):

        # Initialize a random two-body FermionOperator.
        n_qubits = 4
        random_operator = get_fermion_operator(
            random_interaction_operator(n_qubits, seed=17004))

        # Convert to chemist tensor.
        constant, one_body_coefficients, chemist_tensor = (
            get_chemist_two_body_coefficients(random_operator))

        # Perform decomposition.
        eigenvalues, one_body_squares, trunc_error = (
            low_rank_two_body_decomposition(chemist_tensor))

        # Build back two-body component.
        for l in range(n_qubits ** 2):

            # Get the squared one-body operator.
            one_body_operator = FermionOperator()
            for p, q in itertools.product(range(n_qubits), repeat=2):
                term = ((p, 1), (q, 0))
                coefficient = one_body_squares[l, p, q]
                one_body_operator += FermionOperator(term, coefficient)
            one_body_squared = one_body_operator ** 2

            # Get the squared one-body operator via one-body decomposition.
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
