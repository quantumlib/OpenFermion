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

"""Test for qubit_tapering_from_stabilizer model."""

import unittest

import numpy

from openfermion.hamiltonians import MolecularData
from openfermion.ops import QubitOperator
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.utils import eigenspectrum

from openfermion.utils import reduce_number_of_terms, taper_off_qubits
from openfermion.utils._qubit_tapering_from_stabilizer import\
    StabilizerError


def lih_hamiltonian():
    """
    Generate test Hamiltonian from LiH.

    Args:
        None

    Return:

        hamiltonian: FermionicOperator

        spectrum: List of energies.
    """
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    active_space_start = 1
    active_space_stop = 3
    molecule = MolecularData(geometry, 'sto-3g', 1,
                             description="1.45")
    molecule.load()

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=range(active_space_start),
        active_indices=range(active_space_start, active_space_stop))

    hamiltonian = get_fermion_operator(molecular_hamiltonian)
    spectrum = eigenspectrum(hamiltonian)

    return hamiltonian, spectrum


class TaperingTest(unittest.TestCase):
    """TaperingTest class."""

    def test_function_errors(self):
        """Test error of main function."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)
        with self.assertRaises(TypeError):
            reduce_number_of_terms(operator=1,
                                   stabilizers=stab1 + stab2)
        with self.assertRaises(TypeError):
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=1)
        with self.assertRaises(StabilizerError):
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=stab1 + stab2,
                                   manual_input=True)
        with self.assertRaises(StabilizerError):
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=stab1 + stab2,
                                   manual_input=True,
                                   fixed_positions=[1])
        with self.assertRaises(StabilizerError):
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=stab1 + stab2,
                                   manual_input=True,
                                   fixed_positions=[1, 1])
        with self.assertRaises(StabilizerError):
            # Check Identity as stabilizer error.
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=(stab1 +
                                                QubitOperator(' ', 1.0)))
        with self.assertRaises(StabilizerError):
            # Check complex coefficient stabilzier error.
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=(stab1 +
                                                QubitOperator('Z0', 1.0j)))
        with self.assertRaises(StabilizerError):
            # Check linearly-dependent stabilizer error.
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=(stab1 +
                                                QubitOperator('Z0 Z1 Z2 Z3',
                                                              1.0) +
                                                stab2))
        with self.assertRaises(StabilizerError):
            # Check anti-commuting stabilizer error.
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=(QubitOperator('X0', 1.0) +
                                                QubitOperator('Y0', 1.0)))

    def test_reduce_terms(self):
        """Test reduce_terms function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        red_eigenspectrum = eigenspectrum(
            reduce_number_of_terms(qubit_hamiltonian,
                                   stab1 + stab2))

        self.assertAlmostEqual(spectrum[0], red_eigenspectrum[0])

    def test_tapering_qubits_manual_input_false(self):
        """Test taper_off_qubits function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        tapered_hamiltonian = taper_off_qubits(operator=qubit_hamiltonian,
                                               stabilizers=stab1 + stab2,
                                               manual_input=False,
                                               fixed_positions=[0, 3])
        tapered_spectrum = eigenspectrum(tapered_hamiltonian)

        self.assertAlmostEqual(spectrum[0], tapered_spectrum[0])

    def test_tapering_qubits_manual_input(self):
        """
        Test taper_off_qubits function using LiH Hamiltonian.

        Checks different qubits inputs to remove manually.

        Test the lowest eigenvalue against the full Hamiltonian,
        and the full spectrum between them.
        """
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        tapered_ham_0_3 = taper_off_qubits(qubit_hamiltonian,
                                           stab1 + stab2,
                                           manual_input=True,
                                           fixed_positions=[0, 3])
        tapered_ham_2_1 = taper_off_qubits(qubit_hamiltonian,
                                           stab1 + stab2,
                                           manual_input=True,
                                           fixed_positions=[2, 1])

        tapered_spectrum_0_3 = eigenspectrum(tapered_ham_0_3)
        tapered_spectrum_2_1 = eigenspectrum(tapered_ham_2_1)

        self.assertAlmostEqual(spectrum[0], tapered_spectrum_0_3[0])
        self.assertAlmostEqual(spectrum[0], tapered_spectrum_2_1[0])
        self.assertTrue(numpy.allclose(tapered_spectrum_0_3,
                                       tapered_spectrum_2_1))
