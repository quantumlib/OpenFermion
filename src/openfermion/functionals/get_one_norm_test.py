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
"""Tests for get_one_norm."""

import os
import unittest
from openfermion import get_one_norm, MolecularData, jordan_wigner
from openfermion.config import DATA_DIRECTORY

class test_get_one_norm(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(DATA_DIRECTORY,
                                     "H1-Li1_sto-3g_singlet_1.45.hdf5")
        self.molecule = MolecularData(filename=self.filename)
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()
        self.qubit_hamiltonian = jordan_wigner(self.molecular_hamiltonian)

    def test_one_norm_from_molecule(self):
        self.assertAlmostEqual(self.qubit_hamiltonian.induced_norm(),
                               get_one_norm(self.molecule))

    def test_one_norm_from_ints(self):
        ints = (self.molecule.nuclear_repulsion,
                self.molecule.one_body_integrals,
                self.molecule.two_body_integrals)
        self.assertAlmostEqual(self.qubit_hamiltonian.induced_norm(),
                               get_one_norm(ints))

    def test_one_norm_woconst(self):
        one_norm_woconst = (self.qubit_hamiltonian.induced_norm() -
                            abs(self.qubit_hamiltonian.constant))
        ints = (self.molecule.nuclear_repulsion,
                self.molecule.one_body_integrals,
                self.molecule.two_body_integrals)
        self.assertAlmostEqual(one_norm_woconst,
                               get_one_norm(self.molecule,
                                            return_constant=False))
        self.assertAlmostEqual(one_norm_woconst,
                               get_one_norm(ints,return_constant=False))
