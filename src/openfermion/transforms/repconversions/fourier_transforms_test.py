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
"""tests for fourier_transforms.py"""

import unittest

from openfermion.utils import Grid
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.transforms.opconversions import normal_ordered
from openfermion.linalg import get_sparse_operator
from openfermion.utils import is_hermitian

from openfermion.transforms.repconversions.fourier_transforms import (
    fourier_transform, inverse_fourier_transform)


class FourierTransformTest(unittest.TestCase):

    def test_fourier_transform(self):
        for length in [2, 3]:
            grid = Grid(dimensions=1, scale=1.5, length=length)
            spinless_set = [True, False]
            geometry = [('H', (0.1,)), ('H', (0.5,))]
            for spinless in spinless_set:
                h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless,
                                                      True)
                h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless,
                                                      False)
                h_plane_wave_t = fourier_transform(h_plane_wave, grid, spinless)

                self.assertEqual(normal_ordered(h_plane_wave_t),
                                 normal_ordered(h_dual_basis))

                # Verify that all 3 are Hermitian
                plane_wave_operator = get_sparse_operator(h_plane_wave)
                dual_operator = get_sparse_operator(h_dual_basis)
                plane_wave_t_operator = get_sparse_operator(h_plane_wave_t)
                self.assertTrue(is_hermitian(plane_wave_operator))
                self.assertTrue(is_hermitian(dual_operator))
                self.assertTrue(is_hermitian(plane_wave_t_operator))

    def test_inverse_fourier_transform_1d(self):
        grid = Grid(dimensions=1, scale=1.5, length=4)
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless,
                                                  True)
            h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless,
                                                  False)
            h_dual_basis_t = inverse_fourier_transform(h_dual_basis, grid,
                                                       spinless)
            self.assertEqual(normal_ordered(h_dual_basis_t),
                             normal_ordered(h_plane_wave))

    def test_inverse_fourier_transform_2d(self):
        grid = Grid(dimensions=2, scale=1.5, length=3)
        spinless = True
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless, True)
        h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless, False)
        h_dual_basis_t = inverse_fourier_transform(h_dual_basis, grid, spinless)
        self.assertEqual(normal_ordered(h_dual_basis_t),
                         normal_ordered(h_plane_wave))