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

"""Tests for Bose-Hubbard model module."""
from __future__ import absolute_import

import unittest

from openfermion.hamiltonians import bose_hubbard


class BoseHubbardTest(unittest.TestCase):

    def setUp(self):
        self.x_dimension = 2
        self.y_dimension = 2
        self.tunneling = 2.
        self.interaction = 1.
        self.chemical_potential = 0.25
        self.dipole = 1.
        self.periodic = 0

    def test_two_by_two(self):

        # Initialize the Hamiltonian.
        hubbard_model = bose_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling,
            self.interaction, self.chemical_potential, self.dipole,
            self.periodic)

        # Check on on-site interaction and chemical-potential terms.
        chem_coeff = -self.interaction/2 - self.chemical_potential
        on_site_coeff = self.interaction/2
        for i in range(4):
            self.assertAlmostEqual(
                hubbard_model.terms[((i, 1), (i, 0))], chem_coeff)
            self.assertAlmostEqual(
                hubbard_model.terms[((i, 1), (i, 0), (i, 1), (i, 0))],
                on_site_coeff)

        # Check right/left hopping terms.
        t_coeff = -self.tunneling
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (1, 0))], t_coeff)
        self.assertAlmostEqual(hubbard_model.terms[((0, 0), (1, 1))], t_coeff)
        self.assertAlmostEqual(hubbard_model.terms[((2, 0), (3, 1))], t_coeff)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (3, 0))], t_coeff)

        # Check top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (2, 0))], t_coeff)
        self.assertAlmostEqual(hubbard_model.terms[((0, 0), (2, 1))], t_coeff)
        self.assertAlmostEqual(hubbard_model.terms[((1, 0), (3, 1))], t_coeff)
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (3, 0))], t_coeff)

        # Check left/right dipole interaction terms.
        d_coeff = self.dipole
        self.assertAlmostEqual(
            hubbard_model.terms[((0, 1), (0, 0), (1, 1), (1, 0))], d_coeff)
        self.assertAlmostEqual(
            hubbard_model.terms[((2, 1), (2, 0), (3, 1), (3, 0))], d_coeff)

        # Check top/bottom interaction terms.
        self.assertAlmostEqual(
            hubbard_model.terms[((0, 1), (0, 0), (2, 1), (2, 0))], d_coeff)
        self.assertAlmostEqual(
            hubbard_model.terms[((1, 1), (1, 0), (3, 1), (3, 0))], d_coeff)

        # Check that there are no other interaction terms.
        self.assertNotIn(((0, 1), (0, 0), (3, 1), (3, 0)),
                         hubbard_model.terms)
        self.assertNotIn(((1, 1), (1, 0), (2, 1), (2, 0)),
                         hubbard_model.terms)

    def test_two_by_two_periodic_rudimentary(self):
        hubbard_model = bose_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling,
            self.interaction, self.chemical_potential, self.dipole,
            periodic=True)

    def test_two_by_three_periodic_rudimentary(self):
        hubbard_model = bose_hubbard(
            2, 3, self.tunneling, self.interaction,
            self.chemical_potential, self.dipole, periodic=True)
        # Check up top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 0), (4, 1))],
                               -self.tunneling)

    def test_three_by_two_periodic_rudimentary(self):
        hubbard_model = bose_hubbard(
            3, 2, self.tunneling, self.interaction,
            self.chemical_potential, self.dipole, periodic=True)
        # Check up top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 0), (2, 1))],
                               -self.tunneling)
