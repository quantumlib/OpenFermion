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

"""Test for hamiltonian stabilizer sets."""

import unittest

from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import fermi_hubbard
from openfermion import QubitOperator

from openfermion.measurements import get_hamiltonian_subsets
from openfermion.measurements._hamiltonian_stabilizers_sets import (
    _check_stabilizer_overlap, _check_missing_paulis)


def hubbard_ham():
    """
    Generate Hubbard model Hamiltonian to test functions.

    Args:
        None
        Return:
        hamiltonian: QubitOperator in Jordan-Wigner
        spectrum: List of energies.
    """
    # Set model.
    x_dimension = 2
    y_dimension = 1
    tunneling = 1
    coulomb = 2
    magnetic_field = 0.
    chemical_potential = 0.
    periodic = 1
    spinless = 0

    # Get fermion operator.
    hubbard_model = fermi_hubbard(
        x_dimension, y_dimension, tunneling, coulomb, chemical_potential,
        magnetic_field, periodic, spinless)

    # Get qubit operator under Jordan-Wigner.
    jw_hamiltonian = jordan_wigner(hubbard_model)
    jw_hamiltonian.compress()

    return jw_hamiltonian


class HamiltonianSetTest(unittest.TestCase):
    """Hamiltonian subset test class."""

    def test_function_errors(self):
        """Test error of main function."""
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)
        stab3 = QubitOperator('Z0 Z1 Z2 Z3', 1.0)
        ham = hubbard_ham()

        with self.assertRaises(TypeError):
            get_hamiltonian_subsets(hamiltonian=1.0, stabilizers=[
                                    stab1, stab2, stab3])
        with self.assertRaises(TypeError):
            get_hamiltonian_subsets(hamiltonian=ham, stabilizers=1.0)

    def test_check_stabilizer_overlap(self):
        """Test function _check_stabilizer_overlap."""
        pauli = QubitOperator('Z0 X1', 1.0)
        stab1 = QubitOperator('X2 Y3', -1.0)
        stab2 = QubitOperator('Z0 Z2', -1.0)
        self.assertFalse(_check_stabilizer_overlap(QubitOperator(' '), stab1))
        self.assertFalse(_check_stabilizer_overlap(pauli, stab1))
        self.assertTrue(_check_stabilizer_overlap(pauli, stab2))

    def test_missing_paulis(self):
        """Test _check_missing_paulis function."""
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)
        ham = hubbard_ham()

        ham_subsets, pauli_rest = get_hamiltonian_subsets(
            ham, [stab1, stab2])

        # Will compare the number of Pauli strings in the Hamiltonian,
        # with respect to the sum of the strings in the subsets
        # and in Pauli rest.

        num_paulis_ham = len(ham.terms)
        num_paulis_left = len(pauli_rest.terms)
        aux_set = set()
        for sb in ham_subsets:
            aux_set.update(set(sb.terms.keys()))
        num_paulis_in_subsets = len(aux_set)
        self.assertTrue(num_paulis_ham == (
            num_paulis_in_subsets + num_paulis_left))

    def test_no_missing_paulis(self):
        """Test return when no paulis are missing."""
        op = QubitOperator('Z0 X1 X2', 1.0)
        sub_list = [QubitOperator('Z0 X1 X2', 1.0)]
        self.assertIsInstance(_check_missing_paulis(
            op, sub_list), QubitOperator)
