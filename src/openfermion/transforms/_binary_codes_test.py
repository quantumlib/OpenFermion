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

from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import (binary_code_transform, bravyi_kitaev,
                                    get_fermion_operator, jordan_wigner)
from openfermion.utils import eigenspectrum

from openfermion.transforms._binary_codes import (
        bravyi_kitaev_code,
        checksum_code,
        interleaved_code,
        jordan_wigner_code,
        parity_code,
        weight_one_segment_code,
        weight_one_binary_addressing_code,
        weight_one_segment_code,
        weight_two_segment_code,
        linearize_decoder)


def lih_hamiltonian():
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
    ground_state_energy = eigenspectrum(hamiltonian)[0]
    return hamiltonian, ground_state_energy


class LinearizeDecoderTest(unittest.TestCase):

    def test_linearize(self):
        a = linearize_decoder([[0, 1, 1], [1, 0, 0]])
        self.assertListEqual([str(a[0]), str(a[1])], ['[W1] + [W2]', '[W0]'])


class CodeTransformTest(unittest.TestCase):
    def test_tranform_function(self):
        ferm_op = FermionOperator('2')
        n_modes = 5
        qubit_op = binary_code_transform(ferm_op, parity_code(n_modes))
        correct_op = QubitOperator(((1, 'Z'), (2, 'X'), (3, 'X'), (4, 'X')),
                                   0.5) + \
                     QubitOperator(((2, 'Y'), (3, 'X'), (4, 'X')), 0.5j)
        self.assertTrue(qubit_op == correct_op)
        ferm_op = FermionOperator('2^')
        n_modes = 5
        qubit_op = binary_code_transform(ferm_op, parity_code(n_modes))
        correct_op = QubitOperator(((1, 'Z'), (2, 'X'), (3, 'X'), (4, 'X')),
                                   0.5) \
                     + QubitOperator(((2, 'Y'), (3, 'X'), (4, 'X')), -0.5j)
        self.assertTrue(qubit_op == correct_op)

        ferm_op = FermionOperator('5^')
        op2 = QubitOperator('Z0 Z1 Z2 Z3 Z4 X5', 0.5) \
              - QubitOperator('Z0 Z1 Z2 Z3 Z4 Y5', 0.5j)
        op1 = binary_code_transform(ferm_op, jordan_wigner_code(6))
        self.assertTrue(op1 == op2)

    def test_checksum_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = checksum_code(4, 0)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_jordan_wigner(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = jordan_wigner_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])
        self.assertDictEqual(qubit_hamiltonian.terms,
                             jordan_wigner(hamiltonian).terms)

    def test_bravyi_kitaev(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = bravyi_kitaev_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])
        qubit_spectrum = eigenspectrum(qubit_hamiltonian)
        fenwick_spectrum = eigenspectrum(bravyi_kitaev(hamiltonian))
        for eigen_idx, eigenvalue in enumerate(qubit_spectrum):
            self.assertAlmostEqual(eigenvalue, fenwick_spectrum[eigen_idx])

    def test_parity_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = parity_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_weight_one_binary_addressing_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = interleaved_code(8) * (
                2 * weight_one_binary_addressing_code(2))
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_weight_one_segment_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = interleaved_code(6) * (2 * weight_one_segment_code())
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_weight_two_segment_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = weight_two_segment_code()
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_interleaved_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = interleaved_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])
