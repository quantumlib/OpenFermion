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

"""Tests for _verstraete_cirac.py."""

import unittest

from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import (get_sparse_operator,
                                    verstraete_cirac_2d_square)
from openfermion.transforms._verstraete_cirac import (
    coordinates_to_snake_index, snake_index_to_coordinates,
    stabilizer_local_2d_square)
from openfermion.utils import get_ground_state


class VerstraeteCirac2dSquareGroundStateTest(unittest.TestCase):
    """Test that the transform preserves desired ground state properties."""

    def setUp(self):
        self.x_dimension = 2
        self.y_dimension = 3

        # Create a Hamiltonian with nearest-neighbor hopping terms
        self.ferm_op = fermi_hubbard(self.x_dimension, self.y_dimension,
                                     1., 0., 0., 0., False, True)

        # Get the ground energy and ground state
        self.ferm_op_sparse = get_sparse_operator(self.ferm_op)
        self.ferm_op_ground_energy, self.ferm_op_ground_state = (
            get_ground_state(self.ferm_op_sparse))

        # Transform the FermionOperator to a QubitOperator
        self.transformed_op = verstraete_cirac_2d_square(
            self.ferm_op, self.x_dimension, self.y_dimension,
            add_auxiliary_hamiltonian=True, snake=False)

        # Get the ground energy and state of the transformed operator
        self.transformed_sparse = get_sparse_operator(self.transformed_op)
        self.transformed_ground_energy, self.transformed_ground_state = (
            get_ground_state(self.transformed_sparse))

    def test_ground_energy(self):
        """Test that the transformation preserves the ground energy."""
        self.assertAlmostEqual(self.transformed_ground_energy,
                               self.ferm_op_ground_energy)


class VerstraeteCirac2dSquareOperatorLocalityTest(unittest.TestCase):
    """Test that the transform results in local qubit operators."""

    def setUp(self):
        self.x_dimension = 6
        self.y_dimension = 6

        # Create a Hubbard Hamiltonian
        self.ferm_op = fermi_hubbard(self.x_dimension, self.y_dimension,
                                     1.0, 4.0, 0.0, 0.0, False, True)

        # Transform the FermionOperator to a QubitOperator without including
        # the auxiliary Hamiltonian
        self.transformed_op_no_aux = verstraete_cirac_2d_square(
            self.ferm_op, self.x_dimension, self.y_dimension,
            add_auxiliary_hamiltonian=False, snake=False)
        self.transformed_op_no_aux.compress()

        # Transform the FermionOperator to a QubitOperator, including
        # the auxiliary Hamiltonian
        self.transformed_op_aux = verstraete_cirac_2d_square(
            self.ferm_op, self.x_dimension, self.y_dimension,
            add_auxiliary_hamiltonian=True, snake=False)
        self.transformed_op_aux.compress()

    def test_operator_locality_no_aux(self):
        """Test that the operators without the auxiliary Hamiltonian
        are at most 4-local."""
        for term in self.transformed_op_no_aux.terms:
            self.assertTrue(len(term) <= 4)

    def test_operator_locality_aux(self):
        """Test that the operators with the auxiliary Hamiltonian
        are at most 6-local."""
        for term in self.transformed_op_aux.terms:
            self.assertTrue(len(term) <= 6)


class ExceptionTest(unittest.TestCase):
    """Test that exceptions are raised correctly."""

    def test_verstraete_cirac_2d_square(self):
        ferm_op = fermi_hubbard(3, 2, 1., 0., spinless=True)
        with self.assertRaises(NotImplementedError):
            _ = verstraete_cirac_2d_square(ferm_op, 3, 2)

    def test_stabilizer_local_2d_square(self):
        with self.assertRaises(ValueError):
            _ = stabilizer_local_2d_square(0, 2, 4, 4)

    def test_coordinates_to_snake_index(self):
        with self.assertRaises(ValueError):
            _ = coordinates_to_snake_index(4, 4, 4, 5)
        with self.assertRaises(ValueError):
            _ = coordinates_to_snake_index(4, 4, 5, 4)

    def test_snake_index_to_coordinates(self):
        with self.assertRaises(ValueError):
            _ = snake_index_to_coordinates(20, 4, 5)
