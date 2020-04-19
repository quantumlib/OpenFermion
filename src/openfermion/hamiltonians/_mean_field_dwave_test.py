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

"""Tests for mean-field d-wave model module."""

import unittest

from openfermion.hamiltonians import mean_field_dwave


class MeanfieldDwaveTest(unittest.TestCase):

    def setUp(self):
        self.tunneling = 2.
        self.sc_gap = 2.
        self.chemical_potential = 2.

    def test_two_by_two(self):
        # Test the 2 by 2 model.

        # Initialize the Hamiltonian.
        mean_field_dwave_model = mean_field_dwave(
            2, 2, self.tunneling, self.sc_gap, self.chemical_potential)

        # Check on-site terms.
        for site in range(8):
            self.assertAlmostEqual(
                    mean_field_dwave_model.terms[((site, 1), (site, 0))],
                    -self.chemical_potential)

        # Check up right/left hopping terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((0, 1), (2, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((2, 1), (0, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((4, 1), (6, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((6, 1), (4, 0))],
                               -self.tunneling)

        # Check up top/bottom hopping terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((0, 1), (4, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((4, 1), (0, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((2, 1), (6, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((6, 1), (2, 0))],
                               -self.tunneling)

        # Check down right/left hopping terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((1, 1), (3, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((3, 1), (1, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((5, 1), (7, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((7, 1), (5, 0))],
                               -self.tunneling)

        # Check down top/bottom hopping terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((1, 1), (5, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((5, 1), (1, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((3, 1), (7, 0))],
                               -self.tunneling)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((7, 1), (3, 0))],
                               -self.tunneling)

        # Check right/left pairing terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((0, 1), (3, 1))],
                               -self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((1, 1), (2, 1))],
                               self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((3, 0), (0, 0))],
                               -self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((2, 0), (1, 0))],
                               self.sc_gap / 2)

        self.assertAlmostEqual(mean_field_dwave_model.terms[((4, 1), (7, 1))],
                               -self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((5, 1), (6, 1))],
                               self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((7, 0), (4, 0))],
                               -self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((6, 0), (5, 0))],
                               self.sc_gap / 2)

        # Check top/bottom pairing terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((0, 1), (5, 1))],
                               self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((1, 1), (4, 1))],
                               -self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((5, 0), (0, 0))],
                               self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((4, 0), (1, 0))],
                               -self.sc_gap / 2)

        self.assertAlmostEqual(mean_field_dwave_model.terms[((2, 1), (7, 1))],
                               self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((3, 1), (6, 1))],
                               -self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((7, 0), (2, 0))],
                               self.sc_gap / 2)
        self.assertAlmostEqual(mean_field_dwave_model.terms[((6, 0), (3, 0))],
                               -self.sc_gap / 2)

    def test_two_by_three_spinless_periodic_rudimentary(self):
        mean_field_dwave_model = mean_field_dwave(2, 3,
                                                  self.tunneling, self.sc_gap)
        # Check top/bottom hopping terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((8, 1), (1, 1))],
                               self.sc_gap / 2)

    def test_three_by_two_spinless_periodic_rudimentary(self):
        mean_field_dwave_model = mean_field_dwave(3, 2,
                                                  self.tunneling, self.sc_gap)
        # Check right/left hopping terms.
        self.assertAlmostEqual(mean_field_dwave_model.terms[((4, 1), (1, 1))],
                               -self.sc_gap / 2)
