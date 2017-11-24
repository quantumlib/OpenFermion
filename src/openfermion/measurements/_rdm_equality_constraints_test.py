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

"""Tests for _rdm_equality_constraints.py"""
import unittest
import os

from openfermion.config import THIS_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_interaction_operator
from openfermion.measurements import (one_body_fermion_constraints,
                                      two_body_fermion_constraints)


class FermionConstraintsTest(unittest.TestCase):

    def setUp(self):

        # Setup.
        n_atoms = 2
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data',
                                'H2_sto-3g_singlet_0.7414')
        molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        molecule.load()
        self.n_fermions = molecule.n_electrons
        self.n_orbitals = molecule.n_qubits

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        self.fci_rdm = molecule.get_molecular_rdm(use_fci=1)

    def test_one_body_constraints(self):
        for constraint in one_body_fermion_constraints(
                self.n_orbitals, self.n_fermions):
            interaction_operator = get_interaction_operator(
                constraint, self.n_orbitals)
            constraint_value = self.fci_rdm.expectation(interaction_operator)
            self.assertAlmostEqual(constraint_value, 0.)
            for term, coefficient in constraint.terms.items():
                if len(term) == 2:
                    self.assertTrue(term[0][1])
                    self.assertFalse(term[1][1])
                else:
                    self.assertEqual(term, ())

    def test_two_body_constraints(self):
        for constraint in two_body_fermion_constraints(
                self.n_orbitals, self.n_fermions):
            interaction_operator = get_interaction_operator(
                constraint, self.n_orbitals)
            constraint_value = self.fci_rdm.expectation(interaction_operator)
            self.assertAlmostEqual(constraint_value, 0.)
            for term, coefficient in constraint.terms.items():
                if len(term) == 2:
                    self.assertTrue(term[0][1])
                    self.assertFalse(term[1][1])
                elif len(term) == 4:
                    self.assertTrue(term[0][1])
                    self.assertTrue(term[1][1])
                    self.assertFalse(term[2][1])
                    self.assertFalse(term[3][1])
                else:
                    self.assertEqual(term, ())
