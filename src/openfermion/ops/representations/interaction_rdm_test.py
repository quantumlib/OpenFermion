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
"""Tests for interaction_rdms.py."""

import os
import unittest
import numpy

from openfermion.config import EQ_TOLERANCE, DATA_DIRECTORY
from openfermion.chem import MolecularData
from openfermion.ops.operators import QubitOperator
from openfermion.ops.representations.interaction_operator import InteractionOperator
from openfermion.ops.representations.interaction_rdm import InteractionRDM
from openfermion.ops.representations.interaction_rdm import InteractionRDMError
from openfermion.ops.representations.polynomial_tensor import general_basis_change
from openfermion.transforms.opconversions import jordan_wigner


class InteractionRDMTest(unittest.TestCase):
    def setUp(self):
        geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(DATA_DIRECTORY, 'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(geometry, basis, multiplicity, filename=filename)
        self.molecule.load()
        self.cisd_energy = self.molecule.cisd_energy
        self.rdm = self.molecule.get_molecular_rdm()
        self.hamiltonian = self.molecule.get_molecular_hamiltonian()

    def test_get_qubit_expectations(self):
        qubit_operator = jordan_wigner(self.hamiltonian)
        qubit_expectations = self.rdm.get_qubit_expectations(qubit_operator)

        test_energy = 0.0
        for qubit_term in qubit_expectations.terms:
            term_coefficient = qubit_operator.terms[qubit_term]
            test_energy += term_coefficient * qubit_expectations.terms[qubit_term]
        self.assertLess(abs(test_energy - self.cisd_energy), EQ_TOLERANCE)

    def test_get_qubit_expectations_nonmolecular_term(self):
        with self.assertRaises(InteractionRDMError):
            self.rdm.get_qubit_expectations(QubitOperator('X1 X2 X3 X4 Y6'))

    def test_get_qubit_expectations_through_expectation_method(self):
        qubit_operator = jordan_wigner(self.hamiltonian)
        test_energy = self.rdm.expectation(qubit_operator)

        self.assertLess(abs(test_energy - self.cisd_energy), EQ_TOLERANCE)

    def test_get_molecular_operator_expectation(self):
        expectation = self.rdm.expectation(self.hamiltonian)
        self.assertAlmostEqual(expectation, self.cisd_energy, places=7)

    def test_expectation_bad_type(self):
        with self.assertRaises(InteractionRDMError):
            self.rdm.expectation(12)

    def test_addition(self):
        rdm2 = self.rdm + self.rdm
        self.assertTrue(numpy.array_equal(rdm2.one_body_tensor, rdm2.n_body_tensors[(1, 0)]))
        self.assertTrue(numpy.array_equal(rdm2.two_body_tensor, rdm2.n_body_tensors[(1, 1, 0, 0)]))

    def test_rdm_setters(self):
        temp_rdm = self.molecule.get_molecular_rdm()
        one_body_tensor_test = numpy.eye(4)
        temp_rdm.one_body_tensor = one_body_tensor_test
        self.assertTrue(numpy.array_equal(temp_rdm.n_body_tensors[(1, 0)], one_body_tensor_test))
        two_body_tensor_test = numpy.zeros([4, 4, 4, 4])
        temp_rdm.two_body_tensor = two_body_tensor_test
        self.assertTrue(
            numpy.array_equal(temp_rdm.n_body_tensors[(1, 1, 0, 0)], two_body_tensor_test)
        )

    def test_rotate_basis(self):
        """Test RDM basis set rotation."""
        # Get molecular RDM
        temp_rdm = self.molecule.get_molecular_rdm()
        temp_one_body = temp_rdm.one_body_tensor
        temp_two_body = temp_rdm.two_body_tensor
        n_orbitals = temp_rdm.n_body_tensors[(1, 0)].shape[0]

        # Generate random rotation matrix U
        x = numpy.random.rand(n_orbitals, n_orbitals) + 1j * numpy.random.rand(
            n_orbitals, n_orbitals
        )
        u, _ = numpy.linalg.qr(x)  # QR decomposition guarantees u is unitary

        # Compute reference RDM
        expected_one_body = general_basis_change(temp_one_body, u, (0, 1))
        expected_two_body = general_basis_change(temp_two_body, u, (0, 0, 1, 1))

        # Rotate RDM
        temp_rdm.rotate_basis(u, transpose=False)

        # Compare
        self.assertTrue(numpy.allclose(temp_rdm.one_body_tensor, expected_one_body))
        self.assertTrue(numpy.allclose(temp_rdm.two_body_tensor, expected_two_body))

    def test_rotate_basis_transpose(self):
        """Test RDM basis set rotation with transpose=True"""
        # Get molecular RDM
        temp_rdm = self.molecule.get_molecular_rdm()
        temp_one_body = temp_rdm.one_body_tensor
        temp_two_body = temp_rdm.two_body_tensor
        n_orbitals = temp_rdm.n_body_tensors[(1, 0)].shape[0]

        # Generate random rotation matrix U
        x = numpy.random.rand(n_orbitals, n_orbitals) + 1j * numpy.random.rand(
            n_orbitals, n_orbitals
        )
        u, _ = numpy.linalg.qr(x)  # QR decomposition guarantees u is unitary

        # Compute reference RDM
        expected_one_body = general_basis_change(temp_one_body, u, (0, 1))
        expected_two_body = general_basis_change(temp_two_body, u, (0, 0, 1, 1))

        # Rotate RDM
        temp_rdm.rotate_basis(u.T, transpose=True)

        # Compare
        self.assertTrue(numpy.allclose(temp_rdm.one_body_tensor, expected_one_body))
        self.assertTrue(numpy.allclose(temp_rdm.two_body_tensor, expected_two_body))

    def test_expectation_rotation_invariance(self):
        """Test invariance of expectation value under basis rotations."""

        n_orbitals = 3

        # Generate random complex tensors for RDM
        one_body_rdm = numpy.random.rand(n_orbitals, n_orbitals) + 1j * numpy.random.rand(
            n_orbitals, n_orbitals
        )
        two_body_rdm = numpy.random.rand(
            n_orbitals, n_orbitals, n_orbitals, n_orbitals
        ) + 1j * numpy.random.rand(n_orbitals, n_orbitals, n_orbitals, n_orbitals)
        rdm = InteractionRDM(one_body_rdm, two_body_rdm)

        # Generate a random complex Operator
        one_body_op = numpy.random.rand(n_orbitals, n_orbitals) + 1j * numpy.random.rand(
            n_orbitals, n_orbitals
        )
        two_body_op = numpy.random.rand(
            n_orbitals, n_orbitals, n_orbitals, n_orbitals
        ) + 1j * numpy.random.rand(n_orbitals, n_orbitals, n_orbitals, n_orbitals)
        constant = 0.5
        op = InteractionOperator(constant, one_body_op, two_body_op)

        # Generate a random complex unitary matrix U
        x = numpy.random.rand(n_orbitals, n_orbitals) + 1j * numpy.random.rand(
            n_orbitals, n_orbitals
        )
        u, _ = numpy.linalg.qr(x)  # QR decomposition guarantees u is unitary

        # Calculate expectation value in original basis
        exp_before = rdm.expectation(op)

        # Rotate both RDM and Operator with U
        rdm.rotate_basis(u, transpose=False)
        op.rotate_basis(u, transpose=False)

        # Calculate expectation value in rotated basis
        exp_after = rdm.expectation(op)

        # Assert invariance
        self.assertTrue(numpy.allclose(exp_before, exp_after))
