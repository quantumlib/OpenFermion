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
from openfermion.utils import eigenspectrum, count_qubits

from openfermion.utils import reduce_number_of_terms, taper_off_qubits
from openfermion.utils._qubit_tapering_from_stabilizer import (
    StabilizerError, check_commuting_stabilizers, check_stabilizer_linearity,
    fix_single_term, _reduce_terms, _reduce_terms_keep_length, _lookup_term)


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
    molecule = MolecularData(geometry, 'sto-3g', 1, description="1.45")
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
        hamiltonian, _ = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)
        with self.assertRaises(TypeError):
            reduce_number_of_terms(operator=1, stabilizers=stab1 + stab2)
        with self.assertRaises(TypeError):
            reduce_number_of_terms(operator=qubit_hamiltonian, stabilizers=1)
        with self.assertRaises(TypeError):
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=stab1 + stab2,
                                   manual_input=True,
                                   fixed_positions=None)
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
            # Check complex coefficient stabilizer error.
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=(stab1 +
                                                QubitOperator('Z0', 1.0j)))
        with self.assertRaises(StabilizerError):
            # Check linearly-dependent stabilizer error.
            reduce_number_of_terms(
                operator=qubit_hamiltonian,
                stabilizers=(stab1 + QubitOperator('Z0 Z1 Z2 Z3', 1.0) + stab2))
        with self.assertRaises(StabilizerError):
            # Check anti-commuting stabilizer error.
            reduce_number_of_terms(operator=qubit_hamiltonian,
                                   stabilizers=(QubitOperator('X0', 1.0) +
                                                QubitOperator('Y0', 1.0)))
        with self.assertRaises(StabilizerError):
            # Check linearly-dependent stabilizer error.
            _reduce_terms(
                terms=qubit_hamiltonian,
                stabilizer_list=list(stab1 + QubitOperator('Z0 Z1 Z2 Z3', 1.0) +
                                     stab2),
                manual_input=False,
                fixed_positions=[])
        with self.assertRaises(StabilizerError):
            # Check complex coefficient stabilizer error.
            _reduce_terms(terms=qubit_hamiltonian,
                          stabilizer_list=list(stab1 +
                                               QubitOperator('Z0', 1.0j)),
                          manual_input=False,
                          fixed_positions=[])
        with self.assertRaises(StabilizerError):
            # Check linearly-dependent stabilizer error.
            par_qop = QubitOperator('Z0 Z1 Z2 Z3', 1.0)
            _reduce_terms_keep_length(terms=qubit_hamiltonian,
                                      stabilizer_list=[stab1, par_qop, stab2],
                                      manual_input=False,
                                      fixed_positions=[])
        with self.assertRaises(StabilizerError):
            # Check complex coefficient stabilizer error.
            aux_qop = QubitOperator('Z0', 1.0j)
            _reduce_terms_keep_length(terms=qubit_hamiltonian,
                                      stabilizer_list=[stab1, aux_qop],
                                      manual_input=False,
                                      fixed_positions=[])
        with self.assertRaises(StabilizerError):
            # Test check_commuting_stabilizer function
            # Requires a list of QubitOperators one of which
            # has an imaginary term.
            check_commuting_stabilizers(stabilizer_list=[
                QubitOperator('Z0 Z1', 1.0),
                QubitOperator('X0', 1j)
            ],
                                        msg='This test fails.')
        with self.assertRaises(StabilizerError):
            # Test check_stabilizer_linearity function.
            # Requires a list of QUbitOperators one of which is
            # the identity.
            check_stabilizer_linearity(
                [QubitOperator('Z0 Z1', 1.0),
                 QubitOperator(' ', 1.0)],
                msg='This test fails.')

    def test_fix_single_term(self):
        """Test fix_single_term function."""
        stab2 = QubitOperator('Z1 Z3', -1.0)
        test_term = QubitOperator('Z1 Z2')

        fix1 = fix_single_term(test_term, 1, 'Z', 'X', stab2)
        fix2 = fix_single_term(test_term, 0, 'X', 'X', stab2)

        self.assertTrue(fix1 == (test_term * stab2))
        self.assertTrue(fix2 == test_term)

    def test_lookup_term(self):
        """Test for the auxiliar function _lookup_term."""
        # Dummy test where the initial Pauli string is larger than the
        # updated one.
        start_op = list(QubitOperator('Z0 Z1 Z2 Z3').terms.keys())[0]
        updateop1 = QubitOperator('Z0 Z2', -1.0)
        updateop2 = list(QubitOperator('Z0 Z1 Z2 Z3').terms.keys())

        qop = _lookup_term(start_op, [updateop1], updateop2)
        final_op = list(qop.terms.keys())[0]

        self.assertLess(len(final_op), len(start_op))

    def test_reduce_terms(self):
        """Test reduce_terms function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        red_eigenspectrum = eigenspectrum(
            reduce_number_of_terms(qubit_hamiltonian, stab1 + stab2))

        self.assertAlmostEqual(spectrum[0], red_eigenspectrum[0])

    def test_reduce_terms_manual_input(self):
        """Test reduce_terms function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        red_eigenspectrum = eigenspectrum(
            reduce_number_of_terms(qubit_hamiltonian, [stab1, stab2],
                                   manual_input=True,
                                   fixed_positions=[0, 1]))

        self.assertAlmostEqual(spectrum[0], red_eigenspectrum[0])

    def test_reduce_terms_maintain_length(self):
        """Test reduce_terms function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        red_eigenspectrum = eigenspectrum(
            reduce_number_of_terms(qubit_hamiltonian,
                                   stab1 + stab2,
                                   maintain_length=True))

        self.assertAlmostEqual(spectrum[0], red_eigenspectrum[0])

    def test_reduce_terms_auxiliar_functions(self):
        """Test reduce_terms function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_ham = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        red_ham1, _ = _reduce_terms(terms=qubit_ham,
                                    stabilizer_list=[stab1, stab2],
                                    manual_input=False,
                                    fixed_positions=[])
        red_ham2, _ = _reduce_terms_keep_length(terms=qubit_ham,
                                                stabilizer_list=[stab1, stab2],
                                                manual_input=False,
                                                fixed_positions=[])
        red_eigspct1 = eigenspectrum(red_ham1)
        red_eigspct2 = eigenspectrum(red_ham2)

        self.assertAlmostEqual(spectrum[0], red_eigspct1[0])
        self.assertAlmostEqual(spectrum[0], red_eigspct2[0])

    def test_reduce_terms_auxiliar_functions_manual_input(self):
        """Test reduce_terms function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_ham = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        red_ham1, _ = _reduce_terms(terms=qubit_ham,
                                    stabilizer_list=[stab1, stab2],
                                    manual_input=True,
                                    fixed_positions=[0, 1])
        red_ham2, _ = _reduce_terms_keep_length(terms=qubit_ham,
                                                stabilizer_list=[stab1, stab2],
                                                manual_input=True,
                                                fixed_positions=[0, 1])
        red_eigspct1 = eigenspectrum(red_ham1)
        red_eigspct2 = eigenspectrum(red_ham2)

        self.assertAlmostEqual(spectrum[0], red_eigspct1[0])
        self.assertAlmostEqual(spectrum[0], red_eigspct2[0])

    def test_tapering_qubits_manual_input_false(self):
        """Test taper_off_qubits function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        tapered_hamiltonian = taper_off_qubits(operator=qubit_hamiltonian,
                                               stabilizers=[stab1, stab2],
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

        tapered_ham_0_3 = taper_off_qubits(qubit_hamiltonian, [stab1, stab2],
                                           manual_input=True,
                                           fixed_positions=[0, 3])
        tapered_ham_2_1 = taper_off_qubits(qubit_hamiltonian, [stab1, stab2],
                                           manual_input=True,
                                           fixed_positions=[2, 1])

        tapered_spectrum_0_3 = eigenspectrum(tapered_ham_0_3)
        tapered_spectrum_2_1 = eigenspectrum(tapered_ham_2_1)

        self.assertAlmostEqual(spectrum[0], tapered_spectrum_0_3[0])
        self.assertAlmostEqual(spectrum[0], tapered_spectrum_2_1[0])
        self.assertTrue(
            numpy.allclose(tapered_spectrum_0_3, tapered_spectrum_2_1))

    def test_tapering_qubits_remove_positions(self):
        """Test taper_off_qubits function using LiH Hamiltonian."""
        hamiltonian, spectrum = lih_hamiltonian()
        qubit_hamiltonian = jordan_wigner(hamiltonian)
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)

        (tapered_hamiltonian,
         positions) = taper_off_qubits(operator=qubit_hamiltonian,
                                       stabilizers=[stab1, stab2],
                                       manual_input=True,
                                       fixed_positions=[0, 3],
                                       output_tapered_positions=True)

        tapered_spectrum = eigenspectrum(tapered_hamiltonian)

        self.assertAlmostEqual(spectrum[0], tapered_spectrum[0])
        self.assertEqual(positions, [0, 3])

    def test_tappering_stabilizer_more_qubits(self):
        """Test for stabilizer with more qubits than operator."""
        hamiltonian = QubitOperator('Y0 Y1', 1.0)
        stab = QubitOperator('X0 X1 X2', -1.0)

        num_qubits = max(count_qubits(hamiltonian), count_qubits(stab))
        tap_ham = taper_off_qubits(hamiltonian, stab)
        num_qubits_tap = count_qubits(tap_ham)

        self.assertFalse(num_qubits == num_qubits_tap)
