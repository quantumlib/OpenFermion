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
"""Tests for interaction_operator.py."""
import os
import unittest

import numpy

from openfermion.chem.molecular_data import MolecularData
from openfermion.config import THIS_DIRECTORY
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.ops.representations import DOCIHamiltonian

class DOCIHamiltonianTest(unittest.TestCase):

    def setUp(self):
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.filename = os.path.join(THIS_DIRECTORY, 'data',
                                     'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      filename=self.filename)
        self.molecule.load()

    def test_from_integrals_to_qubit(self):
        hamiltonian = jordan_wigner(self.molecule.get_molecular_hamiltonian())
        doci_hamiltonian = DOCIHamiltonian.from_integrals(
            constant=self.molecule.nuclear_repulsion,
            one_body_integrals=self.molecule.one_body_integrals,
            two_body_integrals=self.molecule.two_body_integrals
        ).qubit_operator

        hamiltonian_matrix = get_sparse_operator(hamiltonian).toarray()
        doci_hamiltonian_matrix = get_sparse_operator(doci_hamiltonian).toarray()
        diagonal = numpy.real(numpy.diag(hamiltonian_matrix))
        doci_diagonal = numpy.real(numpy.diag(doci_hamiltonian_matrix))
        position_of_doci_diag_in_h = [0]*len(doci_diagonal)
        for idx, doci_eigval in enumerate(doci_diagonal):
            closest_in_diagonal = None
            for idx2, eig in enumerate(diagonal):
                if closest_in_diagonal is None or abs(eig - doci_eigval) < abs(closest_in_diagonal - doci_eigval):
                    closest_in_diagonal = eig
                    position_of_doci_diag_in_h[idx] = idx2
            assert abs(closest_in_diagonal - doci_eigval) < 1e-8, "Value "+str(doci_eigval)+" of the DOCI Hamiltonian diagonal did not appear in the diagonal of the full Hamiltonian. The closest value was "+str(closest_in_diagonal)

        sub_matrix = hamiltonian_matrix[numpy.ix_(position_of_doci_diag_in_h, position_of_doci_diag_in_h)]
        assert numpy.allclose(doci_hamiltonian_matrix, sub_matrix), "The coupling between the DOCI states in the DOCI Hamiltonian should be identical to that between these states in the full Hamiltonian bur the DOCI hamiltonian matrix\n"+str(doci_hamiltonian_matrix)+"\ndoes not match the corresponding sub-matrix of the full Hamiltonian\n"+str(sub_matrix)
