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
import numpy as np
from scipy.linalg import norm

from .channel_state import amplitude_damping_channel, dephasing_channel, depolarizing_channel


class ChannelTest(unittest.TestCase):
    def setUp(self):
        """Initialize a few density matrices"""
        zero_state = np.array([[1], [0]], dtype=complex)
        one_state = np.array([[0], [1]], dtype=complex)
        one_one_state = np.kron(one_state, one_state)
        zero_zero_state = np.kron(zero_state, zero_state)
        cat_state = 1.0 / np.sqrt(2) * (zero_zero_state + one_one_state)

        self.density_matrix = np.dot(one_one_state, one_one_state.T)
        self.cat_matrix = np.dot(cat_state, cat_state.T)

    def test_amplitude_damping(self):
        """Test amplitude damping on a simple qubit state"""

        # With probability 0
        test_density_matrix = amplitude_damping_channel(self.density_matrix, 0, 1)
        self.assertAlmostEqual(norm(self.density_matrix - test_density_matrix), 0.0)

        test_density_matrix = amplitude_damping_channel(self.density_matrix, 0, 1, transpose=True)
        self.assertAlmostEqual(norm(self.density_matrix - test_density_matrix), 0.0)

        # With probability 1
        correct_density_matrix = np.zeros((4, 4), dtype=complex)
        correct_density_matrix[2, 2] = 1

        test_density_matrix = amplitude_damping_channel(self.density_matrix, 1, 1)

        self.assertAlmostEqual(norm(correct_density_matrix - test_density_matrix), 0.0)

    def test_dephasing(self):
        """Test dephasing on a simple qubit state"""

        # Check for identity on |11> state
        test_density_matrix = dephasing_channel(self.density_matrix, 1, 1)
        self.assertAlmostEqual(norm(self.density_matrix - test_density_matrix), 0.0)

        test_density_matrix = dephasing_channel(self.density_matrix, 1, 1, transpose=True)

        correct_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, -0.5],
                [0.0, 0.0, -0.5, 1.0],
            ]
        )
        self.assertAlmostEqual(norm(correct_matrix - test_density_matrix), 0.0)

        # Check for correct action on cat state
        # With probability = 0
        test_density_matrix = dephasing_channel(self.cat_matrix, 0, 1)
        self.assertAlmostEqual(norm(self.cat_matrix - test_density_matrix), 0.0)
        # With probability = 1

        correct_matrix = np.array(
            [
                [0.50, 0.25, 0.00, 0.00],
                [0.25, 0.25, 0.00, -0.25],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, -0.25, 0.00, 0.50],
            ]
        )
        test_density_matrix = dephasing_channel(self.cat_matrix, 1, 1)
        self.assertAlmostEqual(norm(correct_matrix - test_density_matrix), 0.0)

    def test_depolarizing(self):
        """Test depolarizing on a simple qubit state"""

        # With probability = 0
        test_density_matrix = depolarizing_channel(self.cat_matrix, 0, 1)
        self.assertAlmostEqual(norm(self.cat_matrix - test_density_matrix), 0.0)

        test_density_matrix = depolarizing_channel(self.cat_matrix, 0, 1, transpose=True)
        self.assertAlmostEqual(norm(self.cat_matrix - test_density_matrix), 0.0)

        # With probability 1 on both qubits
        correct_density_matrix = np.array(
            [
                [0.27777778, 0.00000000, 0.00000000, 0.05555556],
                [0.00000000, 0.22222222, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.22222222, 0.00000000],
                [0.05555556, 0.00000000, 0.00000000, 0.27777778],
            ]
        )

        test_density_matrix = depolarizing_channel(self.cat_matrix, 1, 0)
        test_density_matrix = depolarizing_channel(test_density_matrix, 1, 1)

        self.assertAlmostEqual(norm(correct_density_matrix - test_density_matrix), 0.0, places=6)

        # Depolarizing channel should be self-adjoint
        test_density_matrix = depolarizing_channel(self.cat_matrix, 1, 0, transpose=True)
        test_density_matrix = depolarizing_channel(test_density_matrix, 1, 1, transpose=True)

        self.assertAlmostEqual(norm(correct_density_matrix - test_density_matrix), 0.0, places=6)

        # With probability 1 for total depolarization
        correct_density_matrix = np.eye(4) / 4.0
        test_density_matrix = depolarizing_channel(self.cat_matrix, 1, 'All')
        self.assertAlmostEqual(norm(correct_density_matrix - test_density_matrix), 0.0, places=6)

    def test_verification(self):
        """Verify basic sanity checking on inputs"""
        with self.assertRaises(ValueError):
            _ = amplitude_damping_channel(self.density_matrix, 2, 1)

        with self.assertRaises(ValueError):
            _ = amplitude_damping_channel(self.density_matrix, 0.5, 3)

        with self.assertRaises(ValueError):
            bad_density = np.zeros((3, 4))
            _ = amplitude_damping_channel(bad_density, 0.5, 3)
