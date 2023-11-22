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
"""Tests for doci_hamiltonian.py."""
import sys
import os
import unittest

import numpy

from openfermion.config import EQ_TOLERANCE
from openfermion.chem.molecular_data import MolecularData
from openfermion.config import DATA_DIRECTORY
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.ops.representations.doci_hamiltonian import (
    DOCIHamiltonian,
    get_tensors_from_doci,
    get_projected_integrals_from_doci,
    get_doci_from_integrals,
)
from openfermion import get_fermion_operator, InteractionOperator, normal_ordered

numpy.set_printoptions(linewidth=2000, threshold=sys.maxsize)


class IntegralTransformsTest(unittest.TestCase):
    def setUp(self):
        self.geometry = [('H', (0.0, 0.0, 0.0)), ('Li', (0.0, 0.0, 1.45))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.filename = os.path.join(DATA_DIRECTORY, 'H1-Li1_sto-3g_singlet_1.45')
        self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity, filename=self.filename
        )
        self.molecule.load()

    def test_integrals_self_inverse(self):
        hc, hr1, hr2 = get_doci_from_integrals(
            self.molecule.one_body_integrals, self.molecule.two_body_integrals
        )
        doci = DOCIHamiltonian(0, hc, hr1, hr2)
        proj_one_body, proj_two_body = doci.get_projected_integrals()
        hc_test, hr1_test, hr2_test = get_doci_from_integrals(proj_one_body, proj_two_body)
        self.assertTrue(numpy.allclose(hc, hc_test))
        self.assertTrue(numpy.allclose(hr1, hr1_test))
        self.assertTrue(numpy.allclose(hr2, hr2_test))

    def test_fermionic_hamiltonian_from_integrals(self):
        constant = self.molecule.nuclear_repulsion
        doci_constant = constant
        hc, hr1, hr2 = get_doci_from_integrals(
            self.molecule.one_body_integrals, self.molecule.two_body_integrals
        )

        doci = DOCIHamiltonian(doci_constant, hc, hr1, hr2)

        doci_qubit_op = doci.qubit_operator
        doci_mat = get_sparse_operator(doci_qubit_op).toarray()
        # doci_eigvals, doci_eigvecs = numpy.linalg.eigh(doci_mat)
        doci_eigvals, _ = numpy.linalg.eigh(doci_mat)

        tensors = doci.n_body_tensors
        one_body_tensors, two_body_tensors = tensors[(1, 0)], tensors[(1, 1, 0, 0)]
        fermion_op1 = get_fermion_operator(
            InteractionOperator(constant, one_body_tensors, 0.0 * two_body_tensors)
        )

        fermion_op2 = get_fermion_operator(
            InteractionOperator(0, 0 * one_body_tensors, 0.5 * two_body_tensors)
        )

        import openfermion as of

        fermion_op1_jw = of.transforms.jordan_wigner(fermion_op1)
        fermion_op2_jw = of.transforms.jordan_wigner(fermion_op2)

        fermion_op_jw = fermion_op1_jw + fermion_op2_jw
        # fermion_eigvals, fermion_eigvecs = numpy.linalg.eigh(
        #    get_sparse_operator(fermion_op_jw).toarray())
        fermion_eigvals, _ = numpy.linalg.eigh(get_sparse_operator(fermion_op_jw).toarray())

        for eigval in doci_eigvals:
            assert any(
                abs(fermion_eigvals - eigval) < 1e-6
            ), "The DOCI spectrum should \
            have been contained in the spectrum of the fermionic operators"

        fermion_diagonal = get_sparse_operator(fermion_op_jw).toarray().diagonal()
        qubit_diagonal = doci_mat.diagonal()
        assert numpy.isclose(fermion_diagonal[0], qubit_diagonal[0]) and numpy.isclose(
            fermion_diagonal[-1], qubit_diagonal[-1]
        ), "The first and last elements of hte qubit and fermionic diagonal of \
        the Hamiltonian maxtrix should be the same as the vaccum should be \
        mapped to the computational all zero state and the completely filled \
        state should be mapped to the all one state"

        print(fermion_diagonal)
        print(qubit_diagonal)

    def test_integrals_to_doci(self):
        one_body_integrals = self.molecule.one_body_integrals
        two_body_integrals = self.molecule.two_body_integrals
        hc, hr1, hr2 = get_doci_from_integrals(one_body_integrals, two_body_integrals)
        n_qubits = one_body_integrals.shape[0]
        self.assertEqual(hc.shape, (n_qubits,))
        self.assertEqual(hr1.shape, (n_qubits, n_qubits))
        self.assertEqual(hr2.shape, (n_qubits, n_qubits))

        for p in range(2):
            self.assertEqual(
                hc[p] + hr2[p, p], 2 * one_body_integrals[p, p] + two_body_integrals[p, p, p, p]
            )
            for q in range(2):
                if p != q:
                    self.assertEqual(hr1[p, q], two_body_integrals[p, p, q, q])
                    self.assertEqual(
                        hr2[p, q],
                        2 * two_body_integrals[p, q, q, p] - two_body_integrals[p, q, p, q],
                    )


class DOCIHamiltonianTest(unittest.TestCase):
    def setUp(self):
        self.geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.filename = os.path.join(DATA_DIRECTORY, 'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity, filename=self.filename
        )
        self.molecule.load()

    def test_n_body_tensor_errors(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        with self.assertRaises(TypeError):
            doci_hamiltonian.n_body_tensors = 0
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((0, 0), (0, 0))]
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((0, 0), (0, 0), (0, 0), (0, 0))]
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((1, 1), (0, 0))]
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((0, 1), (2, 1), (3, 0), (8, 0))]

    def test_errors_operations(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian2 = DOCIHamiltonian.zero(n_qubits=3)
        with self.assertRaises(TypeError):
            doci_hamiltonian += 'a'
        with self.assertRaises(TypeError):
            doci_hamiltonian -= 'a'
        with self.assertRaises(TypeError):
            doci_hamiltonian *= 'a'
        with self.assertRaises(TypeError):
            doci_hamiltonian /= 'a'
        with self.assertRaises(TypeError):
            doci_hamiltonian += doci_hamiltonian2
        with self.assertRaises(TypeError):
            doci_hamiltonian -= doci_hamiltonian2

    def test_adding_constants(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian += 2
        self.assertAlmostEqual(doci_hamiltonian.constant, 2)
        doci_hamiltonian -= 3
        self.assertAlmostEqual(doci_hamiltonian.constant, -1)

    def test_basic_operations(self):
        doci_hamiltonian1 = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian2 = DOCIHamiltonian.from_integrals(
            constant=self.molecule.nuclear_repulsion,
            one_body_integrals=self.molecule.one_body_integrals,
            two_body_integrals=self.molecule.two_body_integrals,
        )
        self.assertTrue(doci_hamiltonian2 == doci_hamiltonian1 + doci_hamiltonian2)
        self.assertTrue(doci_hamiltonian1 - doci_hamiltonian2 == doci_hamiltonian2 / -1)
        self.assertTrue(doci_hamiltonian2 * 0 == doci_hamiltonian1)

    def test_error(self):
        doci_hamiltonian = DOCIHamiltonian.from_integrals(
            constant=self.molecule.nuclear_repulsion,
            one_body_integrals=self.molecule.one_body_integrals,
            two_body_integrals=self.molecule.two_body_integrals,
        )
        with self.assertRaises(TypeError):
            doci_hamiltonian[((1, 0), (0, 1))] = 1
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((1, 0),)]
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((1, 1), (0, 0))]
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((0, 1), (1, 1), (0, 0), (2, 0))]

    def test_getting_setting_constant(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian.constant = 1
        self.assertEqual(doci_hamiltonian[()], 1)

    def test_getting_setting_1body(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian.hc[0] = 2
        doci_hamiltonian.hc[1] = 4
        self.assertEqual(doci_hamiltonian[((0, 1), (0, 0))], 1)
        self.assertEqual(doci_hamiltonian[((1, 1), (1, 0))], 1)
        self.assertEqual(doci_hamiltonian[((2, 1), (2, 0))], 2)
        self.assertEqual(doci_hamiltonian[((3, 1), (3, 0))], 2)

    def test_getting_setting_hr2(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian.hr2[0, 0] = 2
        doci_hamiltonian.hr2[1, 1] = 4
        self.assertEqual(doci_hamiltonian[((0, 1), (1, 1), (1, 0), (0, 0))], 1)
        self.assertEqual(doci_hamiltonian[((1, 1), (0, 1), (0, 0), (1, 0))], 1)
        self.assertEqual(doci_hamiltonian[((2, 1), (3, 1), (3, 0), (2, 0))], 2)
        self.assertEqual(doci_hamiltonian[((3, 1), (2, 1), (2, 0), (3, 0))], 2)

        doci_hamiltonian.hr2[0, 1] = 2
        self.assertEqual(doci_hamiltonian[((0, 1), (2, 1), (2, 0), (0, 0))], 1)
        self.assertEqual(doci_hamiltonian[((0, 1), (3, 1), (3, 0), (0, 0))], 1)
        self.assertEqual(doci_hamiltonian[((1, 1), (2, 1), (2, 0), (1, 0))], 1)
        self.assertEqual(doci_hamiltonian[((1, 1), (3, 1), (3, 0), (1, 0))], 1)

    def test_getting_setting_hr1(self):
        doci_hamiltonian = DOCIHamiltonian.zero(n_qubits=2)
        doci_hamiltonian.hr1[0, 1] = 2
        self.assertEqual(doci_hamiltonian[(0, 1), (1, 1), (3, 0), (2, 0)], 1)
        self.assertEqual(doci_hamiltonian[(1, 1), (0, 1), (2, 0), (3, 0)], 1)

    def test_from_integrals_to_qubit(self):
        hamiltonian = jordan_wigner(self.molecule.get_molecular_hamiltonian())
        doci_hamiltonian = DOCIHamiltonian.from_integrals(
            constant=self.molecule.nuclear_repulsion,
            one_body_integrals=self.molecule.one_body_integrals,
            two_body_integrals=self.molecule.two_body_integrals,
        ).qubit_operator

        hamiltonian_matrix = get_sparse_operator(hamiltonian).toarray()
        doci_hamiltonian_matrix = get_sparse_operator(doci_hamiltonian).toarray()
        diagonal = numpy.real(numpy.diag(hamiltonian_matrix))
        doci_diagonal = numpy.real(numpy.diag(doci_hamiltonian_matrix))
        position_of_doci_diag_in_h = [0] * len(doci_diagonal)
        for idx, doci_eigval in enumerate(doci_diagonal):
            closest_in_diagonal = None
            for idx2, eig in enumerate(diagonal):
                if closest_in_diagonal is None or abs(eig - doci_eigval) < abs(
                    closest_in_diagonal - doci_eigval
                ):
                    closest_in_diagonal = eig
                    position_of_doci_diag_in_h[idx] = idx2
            assert abs(closest_in_diagonal - doci_eigval) < EQ_TOLERANCE, (
                "Value "
                + str(doci_eigval)
                + " of the DOCI Hamiltonian "
                + "diagonal did not appear in the diagonal of the full "
                + "Hamiltonian. The closest value was "
                + str(closest_in_diagonal)
            )

        sub_matrix = hamiltonian_matrix[
            numpy.ix_(position_of_doci_diag_in_h, position_of_doci_diag_in_h)
        ]
        assert numpy.allclose(doci_hamiltonian_matrix, sub_matrix), (
            "The coupling between the DOCI states in the DOCI Hamiltonian "
            + "should be identical to that between these states in the full "
            + "Hamiltonian but the DOCI hamiltonian matrix\n"
            + str(doci_hamiltonian_matrix)
            + "\ndoes not match the corresponding sub-matrix of the full "
            + "Hamiltonian\n"
            + str(sub_matrix)
        )
