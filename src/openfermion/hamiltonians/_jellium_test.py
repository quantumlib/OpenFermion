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


import unittest

import numpy

from openfermion.hamiltonians._jellium import *
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import get_sparse_operator, jordan_wigner
from openfermion.utils import count_qubits, eigenspectrum, Grid, is_hermitian


class WignerSeitzRadiusTest(unittest.TestCase):

    def test_wigner_seitz_radius_1d(self):
        wigner_seitz_radius = 3.17
        n_particles = 20
        one_d_test = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, 1)
        self.assertAlmostEqual(
            one_d_test, n_particles * 2. * wigner_seitz_radius)

    def test_wigner_seitz_radius_2d(self):
        wigner_seitz_radius = 0.5
        n_particles = 3
        two_d_test = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, 2) ** 2.
        self.assertAlmostEqual(
            two_d_test, n_particles * numpy.pi * wigner_seitz_radius ** 2.)

    def test_wigner_seitz_radius_3d(self):
        wigner_seitz_radius = 4.6
        n_particles = 37
        three_d_test = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, 3) ** 3.
        self.assertAlmostEqual(
            three_d_test, n_particles * (4. * numpy.pi / 3. *
                                         wigner_seitz_radius ** 3.))

    def test_wigner_seitz_radius_6d(self):
        wigner_seitz_radius = 5.
        n_particles = 42
        six_d_test = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, 6) ** 6
        self.assertAlmostEqual(
            six_d_test, n_particles * (numpy.pi ** 3 / 6 *
                                       wigner_seitz_radius ** 6))

    def test_wigner_seitz_radius_bad_dimension_not_integer(self):
        with self.assertRaises(ValueError):
            _ = wigner_seitz_length_scale(3, 2, dimension=4.2)

    def test_wigner_seitz_radius_bad_dimension_not_positive(self):
        with self.assertRaises(ValueError):
            _ = wigner_seitz_length_scale(3, 2, dimension=0)


class HypercubeGridTest(unittest.TestCase):

    def test_1d_generation(self):
        dim = 1
        orbitals = 4
        wigner_seitz_radius = 7.

        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dim, orbitals, wigner_seitz_radius)
        self.assertEqual(grid.dimensions, 1)
        self.assertEqual(grid.length, (4,))
        self.assertEqual(grid.volume_scale(), orbitals * wigner_seitz_radius)

    def test_generation_away_from_half_filling(self):
        dim = 1
        orbitals = 100
        wigner_seitz_radius = 7.
        filling = 0.2

        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dim, orbitals, wigner_seitz_radius, filling_fraction=filling)
        self.assertEqual(grid.dimensions, 1)
        self.assertEqual(grid.length, (100,))
        self.assertAlmostEqual(grid.volume_scale(),
                               orbitals * wigner_seitz_radius / 2.5)

    def test_generation_with_spin(self):
        dim = 2
        orbitals = 4
        wigner_seitz_radius = 10.
        spinless = False

        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            dim, orbitals, wigner_seitz_radius, spinless=spinless)
        self.assertEqual(grid.dimensions, 2)
        self.assertEqual(grid.length, (4, 4))
        self.assertAlmostEqual(grid.volume_scale(), numpy.pi * 16 * 100.)

    def test_3d_generation_with_rounding(self):
        filling = 0.42
        grid = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
            3, 5, 1., filling_fraction=filling)
        self.assertEqual(grid.dimensions, 3)
        self.assertEqual(grid.length, (5, 5, 5))

        # There are floor(125 * .42) = 52 particles.
        # The volume scale should be 4/3 pi r^3 * the "true" filling fraction.
        self.assertAlmostEqual(grid.volume_scale(),
                               (4. / 3.) * numpy.pi * (5. ** 3) * (52. / 125))

    def test_raise_ValueError_filling_fraction_too_low(self):
        with self.assertRaises(ValueError):
            _ = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
                3, 5, wigner_seitz_radius=10., filling_fraction=0.005)

    def test_raise_ValueError_filling_fraction_too_high(self):
        with self.assertRaises(ValueError):
            _ = hypercube_grid_with_given_wigner_seitz_radius_and_filling(
                1, 4, wigner_seitz_radius=1., filling_fraction=2.)


class JelliumTest(unittest.TestCase):

    def test_kinetic_integration(self):
        # Compute kinetic energy operator in both momentum and position space.
        grid = Grid(dimensions=2, length=2, scale=3.)
        spinless = False
        momentum_kinetic = plane_wave_kinetic(grid, spinless)
        position_kinetic = dual_basis_kinetic(grid, spinless)

        # Confirm they are Hermitian
        momentum_kinetic_operator = get_sparse_operator(momentum_kinetic)
        self.assertTrue(is_hermitian(momentum_kinetic_operator))

        position_kinetic_operator = get_sparse_operator(position_kinetic)
        self.assertTrue(is_hermitian(position_kinetic_operator))

        # Confirm spectral match and hermiticity
        for length in [2, 3, 4]:
            grid = Grid(dimensions=1, length=length, scale=2.1)
            spinless = False

            momentum_kinetic = plane_wave_kinetic(grid, spinless)
            position_kinetic = dual_basis_kinetic(grid, spinless)

            # Confirm they are Hermitian
            momentum_kinetic_operator = get_sparse_operator(momentum_kinetic)
            self.assertTrue(is_hermitian(momentum_kinetic_operator))

            position_kinetic_operator = get_sparse_operator(position_kinetic)
            self.assertTrue(is_hermitian(position_kinetic_operator))

            # Diagonalize and confirm the same energy.
            jw_momentum = jordan_wigner(momentum_kinetic)
            jw_position = jordan_wigner(position_kinetic)
            momentum_spectrum = eigenspectrum(jw_momentum, 2 * length)
            position_spectrum = eigenspectrum(jw_position, 2 * length)

            # Confirm spectra are the same.
            difference = numpy.amax(
                numpy.absolute(momentum_spectrum - position_spectrum))
            self.assertAlmostEqual(difference, 0.)

    def test_potential_integration(self):
        # Compute potential energy operator in momentum and position space.
        for length in [2, 3]:
            grid = Grid(dimensions=2, length=length, scale=2.)
            spinless = True
            momentum_potential = plane_wave_potential(grid, spinless)
            position_potential = dual_basis_potential(grid, spinless)

            # Confirm they are Hermitian
            momentum_potential_operator = (
                get_sparse_operator(momentum_potential))
            self.assertTrue(is_hermitian(momentum_potential_operator))

            position_potential_operator = (
                get_sparse_operator(position_potential))
            self.assertTrue(is_hermitian(position_potential_operator))

            # Diagonalize and confirm the same energy.
            jw_momentum = jordan_wigner(momentum_potential)
            jw_position = jordan_wigner(position_potential)
            momentum_spectrum = eigenspectrum(jw_momentum)
            position_spectrum = eigenspectrum(jw_position)

            # Confirm spectra are the same.
            difference = numpy.amax(
                numpy.absolute(momentum_spectrum - position_spectrum))
            self.assertAlmostEqual(difference, 0.)

    def test_model_integration(self):
        # Compute Hamiltonian in both momentum and position space.
        for length in [2, 3]:
            grid = Grid(dimensions=2, length=length, scale=1.0)
            spinless = True
            momentum_hamiltonian = jellium_model(grid, spinless, True)
            position_hamiltonian = jellium_model(grid, spinless, False)

            # Confirm they are Hermitian
            momentum_hamiltonian_operator = (
                get_sparse_operator(momentum_hamiltonian))
            self.assertTrue(is_hermitian(momentum_hamiltonian_operator))

            position_hamiltonian_operator = (
                get_sparse_operator(position_hamiltonian))
            self.assertTrue(is_hermitian(position_hamiltonian_operator))

            # Diagonalize and confirm the same energy.
            jw_momentum = jordan_wigner(momentum_hamiltonian)
            jw_position = jordan_wigner(position_hamiltonian)
            momentum_spectrum = eigenspectrum(jw_momentum)
            position_spectrum = eigenspectrum(jw_position)

            # Confirm spectra are the same.
            difference = numpy.amax(
                numpy.absolute(momentum_spectrum - position_spectrum))
            self.assertAlmostEqual(difference, 0.)

    def test_model_integration_with_constant(self):
        # Compute Hamiltonian in both momentum and position space.
        length_scale = 0.7
        for length in [2, 3]:
            grid = Grid(dimensions=2, length=length, scale=length_scale)
            spinless = True

            # Include Madelung constant in the momentum but not the position
            # Hamiltonian.
            momentum_hamiltonian = jellium_model(grid, spinless, True,
                                                 include_constant=True)
            position_hamiltonian = jellium_model(grid, spinless, False)

            # Confirm they are Hermitian
            momentum_hamiltonian_operator = (
                get_sparse_operator(momentum_hamiltonian))
            self.assertTrue(is_hermitian(momentum_hamiltonian_operator))

            position_hamiltonian_operator = (
                get_sparse_operator(position_hamiltonian))
            self.assertTrue(is_hermitian(position_hamiltonian_operator))

            # Diagonalize and confirm the same energy.
            jw_momentum = jordan_wigner(momentum_hamiltonian)
            jw_position = jordan_wigner(position_hamiltonian)
            momentum_spectrum = eigenspectrum(jw_momentum)
            position_spectrum = eigenspectrum(jw_position)

            # Confirm momentum spectrum is shifted 2.8372/length_scale higher.
            max_difference = numpy.amax(momentum_spectrum - position_spectrum)
            min_difference = numpy.amax(momentum_spectrum - position_spectrum)
            self.assertAlmostEqual(max_difference, 2.8372 / length_scale)
            self.assertAlmostEqual(min_difference, 2.8372 / length_scale)

    def test_coefficients(self):
        # Test that the coefficients post-JW transform are as claimed in paper.
        grid = Grid(dimensions=2, length=3, scale=2.)
        spinless = 1
        n_orbitals = grid.num_points
        n_qubits = (2 ** (1 - spinless)) * n_orbitals
        volume = grid.volume_scale()

        # Kinetic operator.
        kinetic = dual_basis_kinetic(grid, spinless)
        qubit_kinetic = jordan_wigner(kinetic)

        # Potential operator.
        potential = dual_basis_potential(grid, spinless)
        qubit_potential = jordan_wigner(potential)

        # Check identity.
        identity = tuple()
        kinetic_coefficient = qubit_kinetic.terms[identity]
        potential_coefficient = qubit_potential.terms[identity]

        paper_kinetic_coefficient = 0.
        paper_potential_coefficient = 0.
        for indices in grid.all_points_indices():
            momenta = grid.momentum_vector(indices)
            paper_kinetic_coefficient += float(
                n_qubits) * momenta.dot(momenta) / float(4. * n_orbitals)

            if momenta.any():
                potential_contribution = -numpy.pi * float(n_qubits) / float(
                    2. * momenta.dot(momenta) * volume)
                paper_potential_coefficient += potential_contribution

        self.assertAlmostEqual(
            kinetic_coefficient, paper_kinetic_coefficient)
        self.assertAlmostEqual(
            potential_coefficient, paper_potential_coefficient)

        # Check Zp.
        for p in range(n_qubits):
            zp = ((p, 'Z'),)
            kinetic_coefficient = qubit_kinetic.terms[zp]
            potential_coefficient = qubit_potential.terms[zp]

            paper_kinetic_coefficient = 0.
            paper_potential_coefficient = 0.
            for indices in grid.all_points_indices():
                momenta = grid.momentum_vector(indices)
                paper_kinetic_coefficient -= momenta.dot(
                    momenta) / float(4. * n_orbitals)

                if momenta.any():
                    potential_contribution = numpy.pi / float(
                        momenta.dot(momenta) * volume)
                    paper_potential_coefficient += potential_contribution

            self.assertAlmostEqual(
                kinetic_coefficient, paper_kinetic_coefficient)
            self.assertAlmostEqual(
                potential_coefficient, paper_potential_coefficient)

        # Check Zp Zq.
        if spinless:
            spins = [None]

        for indices_a in grid.all_points_indices():
            for indices_b in grid.all_points_indices():

                potential_coefficient = 0.
                paper_kinetic_coefficient = 0.
                paper_potential_coefficient = 0.

                position_a = grid.position_vector(indices_a)
                position_b = grid.position_vector(indices_b)
                differences = position_b - position_a

                for spin_a in spins:
                    for spin_b in spins:

                        p = grid.orbital_id(indices_a, spin_a)
                        q = grid.orbital_id(indices_b, spin_b)

                        if p == q:
                            continue

                        zpzq = ((min(p, q), 'Z'), (max(p, q), 'Z'))
                        if zpzq in qubit_potential.terms:
                            potential_coefficient = qubit_potential.terms[zpzq]

                        for indices_c in grid.all_points_indices():
                            momenta = grid.momentum_vector(indices_c)

                            if momenta.any():
                                potential_contribution = numpy.pi * numpy.cos(
                                    differences.dot(momenta)) / float(
                                    momenta.dot(momenta) * volume)
                                paper_potential_coefficient += (
                                    potential_contribution)

                        self.assertAlmostEqual(
                            potential_coefficient, paper_potential_coefficient)

    def test_jordan_wigner_dual_basis_jellium(self):
        # Parameters.
        grid = Grid(dimensions=2, length=3, scale=1.)
        spinless = True

        # Compute fermionic Hamiltonian. Include then subtract constant.
        fermion_hamiltonian = dual_basis_jellium_model(
            grid, spinless, include_constant=True)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        qubit_hamiltonian -= QubitOperator((), 2.8372)

        # Compute Jordan-Wigner Hamiltonian.
        test_hamiltonian = jordan_wigner_dual_basis_jellium(grid, spinless)

        # Make sure Hamiltonians are the same.
        self.assertTrue(test_hamiltonian == qubit_hamiltonian)

        # Check number of terms.
        n_qubits = count_qubits(qubit_hamiltonian)
        if spinless:
            paper_n_terms = 1 - .5 * n_qubits + 1.5 * (n_qubits ** 2)

        num_nonzeros = sum(1 for coeff in qubit_hamiltonian.terms.values() if
                           coeff != 0.0)
        self.assertTrue(num_nonzeros <= paper_n_terms)

    def test_jordan_wigner_dual_basis_jellium_constant_shift(self):
        length_scale = 0.6
        grid = Grid(dimensions=2, length=3, scale=length_scale)
        spinless = True

        hamiltonian_without_constant = jordan_wigner_dual_basis_jellium(
            grid, spinless, include_constant=False)
        hamiltonian_with_constant = jordan_wigner_dual_basis_jellium(
            grid, spinless, include_constant=True)

        difference = hamiltonian_with_constant - hamiltonian_without_constant
        expected = QubitOperator('') * (2.8372 / length_scale)

        self.assertTrue(expected == difference)

    def test_plane_wave_energy_cutoff(self):
        grid = Grid(dimensions=1, length=5, scale=1.0)
        spinless = True
        e_cutoff = 20.0

        hamiltonian_1 = jellium_model(grid, spinless, True, False)
        jw_1 = jordan_wigner(hamiltonian_1)
        spectrum_1 = eigenspectrum(jw_1)

        hamiltonian_2 = jellium_model(grid, spinless, True, False, e_cutoff)
        jw_2 = jordan_wigner(hamiltonian_2)
        spectrum_2 = eigenspectrum(jw_2)

        max_diff = numpy.amax(numpy.absolute(spectrum_1 - spectrum_2))
        self.assertGreater(max_diff, 0.)

    def test_plane_wave_period_cutoff(self):
        # TODO: After figuring out the correct formula for period cutoff for
        #     dual basis, change period_cutoff to default, and change
        #     hamiltonian_1 to a real jellium_model for real integration test.

        grid = Grid(dimensions=2, length=2, scale=1.0)
        spinless = True
        period_cutoff = 0.

        hamiltonian_1 = FermionOperator()
        jw_1 = jordan_wigner(hamiltonian_1)
        spectrum_1 = eigenspectrum(jw_1)

        hamiltonian_2 = jellium_model(grid, spinless, True, False, None, True,
                                      period_cutoff)
        jw_2 = jordan_wigner(hamiltonian_2)
        spectrum_2 = eigenspectrum(jw_2)

        max_diff = numpy.amax(numpy.absolute(spectrum_1 - spectrum_2))
        self.assertGreater(max_diff, 0.)

        # TODO: This is only for code coverage. Remove after having real
        #     integration test.
        jellium_model(grid, spinless, True, False, None, True)
        jellium_model(grid, spinless, False, False, None, True)
