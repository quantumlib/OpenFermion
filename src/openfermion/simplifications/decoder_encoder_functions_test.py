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

from decoder_encoder_functions import *
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import binary_code_transform
from openfermion.transforms import get_fermion_operator
from openfermion.utils import eigenspectrum


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


class CodeTransformTest(unittest.TestCase):
    def test_bravyi_kitaev(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = bravyi_kitaev_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_jordan_wigner(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = jordan_wigner_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_parity_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = parity_code(4)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

    def test_checksum_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = checksum_code(4, 0)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])

        code = checksum_code(4, 1)
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(eigenspectrum(hamiltonian)[1],
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
        
    def test_weight_one_segment_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = interleaved_code(6)*(2*weight_one_segment_code())
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])
        
    def test_weight_two_segment_code(self):
        hamiltonian, gs_energy = lih_hamiltonian()
        code = interleaved_code(8)*(2*weight_two_binary_addressing_code())
        qubit_hamiltonian = binary_code_transform(hamiltonian, code)
        self.assertAlmostEqual(gs_energy,
                               eigenspectrum(qubit_hamiltonian)[0])
