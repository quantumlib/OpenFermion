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

"""Tests for Hubbard model module."""
from __future__ import absolute_import

import unittest

from openfermion.hamiltonians import fermi_hubbard


class FermiHubbardTest(unittest.TestCase):

    def setUp(self):
        self.x_dimension = 2
        self.y_dimension = 2
        self.tunneling = 2.
        self.coulomb = 1.
        self.magnetic_field = 0.5
        self.chemical_potential = 0.25
        self.periodic = 0
        self.spinless = 0

    def test_two_by_two_spinless(self):

        # Initialize the Hamiltonian.
        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            self.periodic, spinless=True)

        # Check on site terms and magnetic field.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (0, 0))], -0.25)
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (1, 0))], -0.25)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (2, 0))], -0.25)
        self.assertAlmostEqual(hubbard_model.terms[((3, 1), (3, 0))], -0.25)

        # Check right/left hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (1, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (0, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((3, 1), (2, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (3, 0))], -2.)

        # Check top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (2, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (0, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((3, 1), (1, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (3, 0))], -2.)

        # Check left/right Coulomb terms.
        self.assertAlmostEqual(
            hubbard_model.terms[((0, 1), (0, 0), (1, 1), (1, 0))], 1.)
        self.assertAlmostEqual(
            hubbard_model.terms[((2, 1), (2, 0), (3, 1), (3, 0))], 1.)

        # Check top/bottom Coulomb terms.
        self.assertAlmostEqual(
            hubbard_model.terms[((0, 1), (0, 0), (2, 1), (2, 0))], 1.)
        self.assertAlmostEqual(
            hubbard_model.terms[((1, 1), (1, 0), (3, 1), (3, 0))], 1.)

        # Check that there are no other Coulomb terms.
        self.assertNotIn(((0, 1), (0, 0), (3, 1), (3, 0)),
                         hubbard_model.terms)
        self.assertNotIn(((1, 1), (1, 0), (2, 1), (2, 0)),
                         hubbard_model.terms)

    def test_two_by_two_spinful(self):
        # Initialize the Hamiltonian.
        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            self.periodic, self.spinless, False)

        # Check up spin on site terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (0, 0))], -.75)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (2, 0))], -.75)
        self.assertAlmostEqual(hubbard_model.terms[((4, 1), (4, 0))], -.75)
        self.assertAlmostEqual(hubbard_model.terms[((6, 1), (6, 0))], -.75)

        # Check down spin on site terms.
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (1, 0))], .25)
        self.assertAlmostEqual(hubbard_model.terms[((3, 1), (3, 0))], .25)
        self.assertAlmostEqual(hubbard_model.terms[((5, 1), (5, 0))], .25)
        self.assertAlmostEqual(hubbard_model.terms[((7, 1), (7, 0))], .25)

        # Check up right/left hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (2, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (0, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((4, 1), (6, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((6, 1), (4, 0))], -2.)

        # Check up top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (4, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((4, 1), (0, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (6, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((6, 1), (2, 0))], -2.)

        # Check down right/left hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (3, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((3, 1), (1, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((5, 1), (7, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((7, 1), (5, 0))], -2.)

        # Check down top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((1, 1), (5, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((5, 1), (1, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((3, 1), (7, 0))], -2.)
        self.assertAlmostEqual(hubbard_model.terms[((7, 1), (3, 0))], -2.)

        # Check on site interaction term.
        self.assertAlmostEqual(hubbard_model.terms[((0, 1), (0, 0),
                                                    (1, 1), (1, 0))], 1.)
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (2, 0),
                                                    (3, 1), (3, 0))], 1.)
        self.assertAlmostEqual(hubbard_model.terms[((4, 1), (4, 0),
                                                    (5, 1), (5, 0))], 1.)
        self.assertAlmostEqual(hubbard_model.terms[((6, 1), (6, 0),
                                                    (7, 1), (7, 0))], 1.)

    def test_two_by_two_spinful_phs(self):
        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            self.periodic, self.spinless, False)

        hubbard_model_phs = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            self.periodic, self.spinless, True)

        # Compute difference between the models with and without
        # spin symmetry.
        difference = hubbard_model_phs - hubbard_model

        # Check constant term in difference.
        self.assertAlmostEqual(difference.terms[()], 1.0)

        # Check up spin on site terms in difference.
        self.assertAlmostEqual(difference.terms[((0, 1), (0, 0))], -.50)
        self.assertAlmostEqual(difference.terms[((2, 1), (2, 0))], -.50)
        self.assertAlmostEqual(difference.terms[((4, 1), (4, 0))], -.50)
        self.assertAlmostEqual(difference.terms[((6, 1), (6, 0))], -.50)

        # Check down spin on site terms in difference.
        self.assertAlmostEqual(difference.terms[((1, 1), (1, 0))], -.50)
        self.assertAlmostEqual(difference.terms[((3, 1), (3, 0))], -.50)
        self.assertAlmostEqual(difference.terms[((5, 1), (5, 0))], -.50)
        self.assertAlmostEqual(difference.terms[((7, 1), (7, 0))], -.50)

    def test_two_by_two_spinful_periodic_rudimentary(self):
        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=True, spinless=False, particle_hole_symmetry=False)

        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=True, spinless=False, particle_hole_symmetry=True)

    def test_two_by_two_spinful_aperiodic_rudimentary(self):
        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=False, spinless=False, particle_hole_symmetry=False)

        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=False, spinless=False, particle_hole_symmetry=True)

    def test_two_by_two_spinless_periodic_rudimentary(self):
        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=True, spinless=True, particle_hole_symmetry=False)

        hubbard_model = fermi_hubbard(
            self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=True, spinless=True, particle_hole_symmetry=True)

    def test_two_by_three_spinless_periodic_rudimentary(self):
        hubbard_model = fermi_hubbard(
            2, 3, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=True, spinless=True, particle_hole_symmetry=False)
        # Check up top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((4, 1), (0, 0))],
                               -self.tunneling)

    def test_three_by_two_spinless_periodic_rudimentary(self):
        hubbard_model = fermi_hubbard(
            3, 2, self.tunneling, self.coulomb,
            self.chemical_potential, self.magnetic_field,
            periodic=True, spinless=True, particle_hole_symmetry=False)
        # Check up top/bottom hopping terms.
        self.assertAlmostEqual(hubbard_model.terms[((2, 1), (0, 0))],
                               -self.tunneling)
