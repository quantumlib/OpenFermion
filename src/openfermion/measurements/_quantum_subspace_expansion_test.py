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

"""Test for quantum subspace expansion functions."""

import unittest

from openfermion.transforms import (jordan_wigner,
                                    bravyi_kitaev, get_fermion_operator,
                                    project_onto_sector)
from openfermion.hamiltonians import MolecularData, fermi_hubbard
from openfermion.ops import QubitOperator
from openfermion.utils import eigenspectrum

from openfermion.measurements import calculate_qse_spectrum
from openfermion.measurements._quantum_subspace_expansion import (
    get_additional_operators)


def hubbard_ham():
    """
    Generate Hubbard model Hamiltonian to test functions.

    Args:
        None
    Return:
        hamiltonian: QubitOperator in Jordan-Wigner
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


def h2_bravyi_kitaev():
    """
    Generate test Hamiltonian from H2.

    Args:
        None

    Return:

        hamiltonian: FermionicOperator
        qop_ham: QubitOperator
    """
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
    molecule = MolecularData(geometry, 'sto-3g', 1,
                             description="0.7414")
    molecule.load()

    molecular_hamiltonian = molecule.get_molecular_hamiltonian()

    hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qop_ham = project_onto_sector(bravyi_kitaev(hamiltonian), [1, 3], [0, 0])

    return hamiltonian, qop_ham


def hubbard_ham_expect_vals():
    """
    Get expectation values.

    Expectation values previously obtained in simulations.

    Returns:
            expct_vals (QubitOperator): Expectation values as QubitOperator.
            expct_vals_ext (QubitOperator): Expectation values of the extended
                Hamiltoinan.
    """
    expct_vals = QubitOperator()
    expct_vals_ext = QubitOperator()

    expct_vals.terms = {((0, 'Y'), (1, 'Z'), (2, 'Y')): 0.8743466545506314,
                        ((0, 'X'), (1, 'Z'), (2, 'X')): 0.8751599153719161,
                        ((1, 'Y'), (2, 'Z'), (3, 'Y')): 0.8728774175805551,
                        ((1, 'X'), (2, 'Z'), (3, 'X')): 0.8732403041510814,
                        (): 1.0,
                        ((1, 'Z'),): 0.0038787944469396707,
                        ((0, 'Z'),): -0.00032851220327859143,
                        ((0, 'Z'), (1, 'Z')): -0.44347605363496034,
                        ((3, 'Z'),): -0.0011170198007229848,
                        ((2, 'Z'),): 0.0011778143202714753,
                        ((2, 'Z'), (3, 'Z')): -0.4439960440736147
                        }

    expct_vals_ext.terms = {((0, 'Y'), (1, 'Z'), (2, 'Y')): 0.8743466545506314,
                            ((0, 'X'), (1, 'Z'), (2, 'X')): 0.8751599153719161,
                            ((1, 'Y'), (2, 'Z'), (3, 'Y')): 0.8728774175805551,
                            ((1, 'X'), (2, 'Z'), (3, 'X')): 0.8732403041510814,
                            (): 1.0,
                            ((0, 'Z'), (1, 'Z')): -0.44347605363496034,
                            ((2, 'Z'), (3, 'Z')): -0.4439960440736147,
                            ((0, 'X'), (2, 'X'),
                                (3, 'Z')): -0.8742992712768001,
                            ((0, 'Y'), (2, 'Y'),
                                (3, 'Z')): -0.874981704992061,
                            ((0, 'Z'), (1, 'X'),
                                (3, 'X')): -0.8769719627156427,
                            ((0, 'Z'), (1, 'Y'),
                                (3, 'Y')): -0.8749501758400109,
                            ((0, 'Z'), (1, 'Z'),
                                (2, 'Z'), (3, 'Z')): 0.97538832027736,
                            ((1, 'Z'), (3, 'Z')): -0.9821867799949147,
                            ((0, 'Z'), (3, 'Z')): 0.4412051987768667,
                            ((1, 'Z'), (2, 'Z')): 0.44311225284211214,
                            ((0, 'Z'), (2, 'Z')): -0.9812982051266501,
                            ((1, 'Z'),): 0.0038787944469396707,
                            ((0, 'Z'),): -0.00032851220327859143,
                            ((3, 'Z'),): -0.0011170198007229848,
                            ((2, 'Z'),): 0.0011778143202714753}

    return expct_vals, expct_vals_ext


def h2_expect_vals():
    """Expectation values for H2 LR-QSE test."""
    expct_vals = QubitOperator()
    expct_vals.terms = {(): 1.0,
                        ((0, 'Y'), (1, 'Y')): -0.21958330202868692,
                        ((0, 'X'), (1, 'X')): -0.21992539710797498,
                        ((0, 'Z'), (1, 'Z')): -0.9787248504287553,
                        ((0, 'Y'), (1, 'X')): 0.0009958622913323499,
                        ((0, 'X'), (1, 'Y')): 1.960326404420033e-05,
                        ((0, 'Y'),): 0.005290597167661589,
                        ((0, 'X'),): 0.0044508645519099055,
                        ((0, 'Y'), (1, 'Z')): 0.005374734672752549,
                        ((0, 'X'), (1, 'Z')): 0.002254414818511139,
                        ((0, 'Z'), (1, 'Y')): -0.0018175385756145346,
                        ((1, 'X'),): 0.003192646549690809,
                        ((0, 'Z'), (1, 'X')): -0.0019412082018775102,
                        ((1, 'Y'),): 0.0006835832299994119,
                        ((0, 'Z'),): -0.9577234581517733,
                        ((1, 'Z'),): 0.961162783855539}

    return expct_vals


class QSETest(unittest.TestCase):
    """Quantum subspace expansion test class."""

    def test_function_errors(self):
        """Test error of main function."""
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)
        stab3 = QubitOperator('Z0 Z1 Z2 Z3', 1.0)
        ham = hubbard_ham()
        (expct_vals, expct_vals_ext) = hubbard_ham_expect_vals()

        with self.assertRaises(TypeError):
            get_additional_operators(hamiltonian=1.0, expansion_operators=[
                stab1, stab2, stab3])
        with self.assertRaises(TypeError):
            get_additional_operators(hamiltonian=ham, expansion_operators=1.0)
        with self.assertRaises(TypeError):
            calculate_qse_spectrum(hamiltonian=1.0, expansion_operators=[
                stab1, stab2, stab3], expectation_values=expct_vals_ext)
        with self.assertRaises(TypeError):
            calculate_qse_spectrum(hamiltonian=ham, expansion_operators=1.0,
                                   expectation_values=expct_vals_ext)
        with self.assertRaises(TypeError):
            calculate_qse_spectrum(hamiltonian=ham, expansion_operators=[
                stab1, stab2, stab3], expectation_values=2.0)
        with self.assertRaises(ValueError):
            calculate_qse_spectrum(hamiltonian=ham, expansion_operators=[
                stab1, stab2, stab3], expectation_values=expct_vals)

    def test_qubitoperator_to_list(self):
        """Test QubitOperators of expansion are set as list."""
        stab_qop = (QubitOperator('Z0 Z2', -1.0) +
                    QubitOperator('Z1 Z3', -1.0) +
                    QubitOperator('Z0 Z1 Z2 Z3', 1.0))
        ham = hubbard_ham()
        (expct_vals, expct_vals_ext) = hubbard_ham_expect_vals()

        get_additional_operators(hamiltonian=ham, expansion_operators=stab_qop)
        calculate_qse_spectrum(hamiltonian=ham, expansion_operators=stab_qop,
                               expectation_values=expct_vals_ext)

    def test_symmetry_qse(self):
        """Function to test SQSE."""
        stab1 = QubitOperator('Z0 Z2', -1.0)
        stab2 = QubitOperator('Z1 Z3', -1.0)
        stab3 = QubitOperator('Z0 Z1 Z2 Z3', 1.0)
        ham = hubbard_ham()
        (expct_vals, expct_vals_ext) = hubbard_ham_expect_vals()

        # We will compare the noisy result with respect
        # to the ground state energy of the Hubbard model.
        hub_gs = eigenspectrum(ham)[0]
        noisy_gs = (ham * expct_vals).terms[()]
        sqse_gs = calculate_qse_spectrum(hamiltonian=ham, expansion_operators=[
            stab1, stab2, stab3], expectation_values=expct_vals_ext)[0]

        # Check that the greatest absolute value of the ground state is given
        # by the eigenspectrum of the Hamiltonian.
        self.assertGreater(abs(hub_gs), abs(noisy_gs))
        self.assertGreater(abs(hub_gs), abs(sqse_gs))
        # Check that the Symmetry-QSE ground state energy is greater than
        # the noisy one in absolute value.
        self.assertGreater(abs(sqse_gs), abs(noisy_gs))

    def test_qse(self):
        """Function to test LR-QSE."""
        lr_expansion = [QubitOperator(p + str(q), 1.0) for q in range(2)
                        for p in ['X', 'Y', 'Z']]
        lr_expansion.append(QubitOperator(' ', 1.0))
        ferop, qop_ham = h2_bravyi_kitaev()
        expct_vals = h2_expect_vals()

        spectrum = eigenspectrum(ferop)
        noisy_gs = (expct_vals * qop_ham).terms[()]
        qse_spectrum = calculate_qse_spectrum(
            qop_ham, lr_expansion, expct_vals)

        # The number of eigenvalues of the qse is equal to the number of
        # expantion operators.
        self.assertTrue(len(qse_spectrum) == len(lr_expansion))
        # Check if the highest excited state is an approximation
        # to true highest eigenstate.
        self.assertGreater(abs(spectrum[-1]), abs(qse_spectrum[-1]))
        self.assertAlmostEqual(abs(spectrum[-1]), abs(qse_spectrum[-1]),
                               places=1)
        # Check if it also mitigates.
        self.assertGreater(abs(spectrum[0]), abs(noisy_gs))
        self.assertGreater(abs(spectrum[0]), abs(qse_spectrum[0]))
        self.assertGreater(abs(qse_spectrum[0]), abs(noisy_gs))
