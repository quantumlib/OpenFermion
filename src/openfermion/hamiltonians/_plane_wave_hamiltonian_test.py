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

"""Tests for plane_wave_hamiltonian.py"""

import unittest


from openfermion.hamiltonians._plane_wave_hamiltonian import *
from openfermion.transforms import jordan_wigner, get_sparse_operator
from openfermion.utils import (eigenspectrum, Grid, inverse_fourier_transform,
                               is_hermitian)


class PlaneWaveHamiltonianTest(unittest.TestCase):

    def test_plane_wave_hamiltonian_integration(self):
        length_set = [2, 3, 4]
        spinless_set = [True, False]
        length_scale = 1.1
        for geometry in [[('H', (0,)), ('H', (0.8,))],
                         [('H', (0.1,))],
                         [('H', (0.1,))]]:
            for l in length_set:
                for spinless in spinless_set:
                    grid = Grid(dimensions=1, scale=length_scale, length=l)
                    h_plane_wave = plane_wave_hamiltonian(
                        grid, geometry, spinless, True, include_constant=False)
                    h_dual_basis = plane_wave_hamiltonian(
                        grid, geometry, spinless, False, include_constant=False)

                    # Test for Hermiticity
                    plane_wave_operator = get_sparse_operator(h_plane_wave)
                    dual_operator = get_sparse_operator(h_dual_basis)
                    self.assertTrue(is_hermitian((plane_wave_operator)))
                    self.assertTrue(is_hermitian(dual_operator))

                    jw_h_plane_wave = jordan_wigner(h_plane_wave)
                    jw_h_dual_basis = jordan_wigner(h_dual_basis)
                    h_plane_wave_spectrum = eigenspectrum(jw_h_plane_wave)
                    h_dual_basis_spectrum = eigenspectrum(jw_h_dual_basis)

                    max_diff = numpy.amax(
                        h_plane_wave_spectrum - h_dual_basis_spectrum)
                    min_diff = numpy.amin(
                        h_plane_wave_spectrum - h_dual_basis_spectrum)
                    self.assertAlmostEqual(max_diff, 0)
                    self.assertAlmostEqual(min_diff, 0)

    def test_plane_wave_hamiltonian_default_to_jellium_with_no_geometry(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        self.assertTrue(plane_wave_hamiltonian(grid) == jellium_model(grid))

    def test_plane_wave_hamiltonian_bad_geometry(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            plane_wave_hamiltonian(grid, geometry=[('H', (0, 0, 0))])

        with self.assertRaises(ValueError):
            plane_wave_hamiltonian(grid, geometry=[('H', (0, 0, 0))],
                                   include_constant=True)

    def test_plane_wave_hamiltonian_bad_element(self):
        grid = Grid(dimensions=3, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            plane_wave_hamiltonian(grid, geometry=[('Unobtainium',
                                                    (0, 0, 0))])

    def test_jordan_wigner_dual_basis_hamiltonian(self):
        grid = Grid(dimensions=2, length=3, scale=1.)
        spinless_set = [True, False]
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        for spinless in spinless_set:
            fermion_hamiltonian = plane_wave_hamiltonian(
                grid, geometry, spinless, False, include_constant=False)
            qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

            test_hamiltonian = jordan_wigner_dual_basis_hamiltonian(
                grid, geometry, spinless, include_constant=False)
            self.assertTrue(test_hamiltonian == qubit_hamiltonian)

    def test_jordan_wigner_dual_basis_hamiltonian_default_to_jellium(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        self.assertTrue(jordan_wigner_dual_basis_hamiltonian(grid) ==
                        jordan_wigner(jellium_model(grid, plane_wave=False)))

    def test_jordan_wigner_dual_basis_hamiltonian_bad_geometry(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            jordan_wigner_dual_basis_hamiltonian(
                grid, geometry=[('H', (0, 0, 0))])

        with self.assertRaises(ValueError):
            jordan_wigner_dual_basis_hamiltonian(
                grid, geometry=[('H', (0, 0, 0))], include_constant=True)

    def test_jordan_wigner_dual_basis_hamiltonian_bad_element(self):
        grid = Grid(dimensions=3, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            jordan_wigner_dual_basis_hamiltonian(
                grid, geometry=[('Unobtainium', (0, 0, 0))])

    def test_plane_wave_energy_cutoff(self):
        geometry = [('H', (0,)), ('H', (0.8,))]
        grid = Grid(dimensions=1, scale=1.1, length=5)
        e_cutoff = 50.0

        h_1 = plane_wave_hamiltonian(grid, geometry, True, True, False)
        jw_1 = jordan_wigner(h_1)
        spectrum_1 = eigenspectrum(jw_1)

        h_2 = plane_wave_hamiltonian(grid, geometry, True, True, False,
                                     e_cutoff)
        jw_2 = jordan_wigner(h_2)
        spectrum_2 = eigenspectrum(jw_2)

        max_diff = numpy.amax(numpy.absolute(spectrum_1 - spectrum_2))
        self.assertGreater(max_diff, 0.)

    def test_plane_wave_period_cutoff(self):
        # TODO: After figuring out the correct formula for period cutoff for
        #     dual basis, change period_cutoff to default, and change
        #     h_1 to also accept period_cutoff for real integration test.

        geometry = [('H', (0,)), ('H', (0.8,))]
        grid = Grid(dimensions=1, scale=1.1, length=5)
        period_cutoff = 50.0

        h_1 = plane_wave_hamiltonian(grid, geometry, True, True, False, None)
        jw_1 = jordan_wigner(h_1)
        spectrum_1 = eigenspectrum(jw_1)

        h_2 = plane_wave_hamiltonian(grid, geometry, True, True, False, None,
                                     True, period_cutoff)
        jw_2 = jordan_wigner(h_2)
        spectrum_2 = eigenspectrum(jw_2)

        max_diff = numpy.amax(numpy.absolute(spectrum_1 - spectrum_2))
        self.assertGreater(max_diff, 0.)

        # TODO: This is only for code coverage. Remove after having real
        #     integration test.
        plane_wave_hamiltonian(grid, geometry, True, True, False, None, True)
