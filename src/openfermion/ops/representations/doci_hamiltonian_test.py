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
import os
import unittest

import numpy

from openfermion.config import EQ_TOLERANCE
from openfermion.chem.molecular_data import MolecularData
from openfermion.config import DATA_DIRECTORY
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.ops.representations.doci_hamiltonian import(
    DOCIHamiltonian, _HR1, _HR2, _HC, get_tensors_from_doci,
    get_projected_integrals_from_doci, get_doci_from_integrals)

class HOpsTest(unittest.TestCase):

    def setUp(self):
        self.n_body_tensors = {
            (): 0,
            (1, 0): numpy.zeros((4, 4)),
            (1, 1, 0, 0): numpy.zeros((4, 4, 4, 4))
        }

    def test_hr1_init_set_get(self):
        hr1_mat = numpy.zeros((2, 2))
        hr1 = _HR1(hr1_mat, self.n_body_tensors)
        hr1[0, 1] = 2
        self.assertEqual(hr1[0, 1], 2)
        self.assertEqual(hr1._n_body_tensors[(1, 1, 0, 0)][0, 1, 3, 2], 1)
        self.assertEqual(hr1._n_body_tensors[(1, 1, 0, 0)][1, 0, 2, 3], 1)

    def test_hr1_ops(self):
        hr1a_mat = numpy.zeros((2, 2))
        hr1a = _HR1(hr1a_mat, self.n_body_tensors)

        hr1b_mat = numpy.zeros((2, 2))
        hr1b = _HR1(hr1b_mat, self.n_body_tensors)

        hr1a[0, 1] = 2
        hr1c = hr1a + hr1b
        self.assertEqual(hr1c[0, 1], 2)

        hr1c = hr1b - hr1a
        self.assertEqual(hr1c[0, 1], -2)

        hr1c = hr1a * 2
        self.assertEqual(hr1c[0, 1], 4)

        hr1c = hr1a / 2
        self.assertAlmostEqual(hr1c[0, 1], 1)

    def test_hr1_shape(self):
        hr1_mat = numpy.zeros((2, 2))
        hr1 = _HR1(hr1_mat, self.n_body_tensors)
        self.assertEqual(len(hr1.shape), 2)
        self.assertEqual(hr1.shape[0], 2)

    def test_hr1_raises_errors(self):
        hr1_mat = numpy.zeros((2, 2))
        hr1 = _HR1(hr1_mat, self.n_body_tensors)
        with self.assertRaises(IndexError):
            _ = hr1[1, 2, 3]
        with self.assertRaises(IndexError):
            hr1[1, 2, 3] = 2
        with self.assertRaises(IndexError):
            hr1[1, 1] = 2
        with self.assertRaises(IndexError):
            _ = hr1[1, 1]

    def test_hr2_init_set_get(self):
        hr2_mat = numpy.zeros((2, 2))
        hr2 = _HR2(hr2_mat, self.n_body_tensors)
        hr2[0, 0] = 2
        self.assertEqual(hr2[0, 0], 2)
        self.assertEqual(hr2._n_body_tensors[(1, 1, 0, 0)][0, 0, 0, 0], 1)
        self.assertEqual(hr2._n_body_tensors[(1, 1, 0, 0)][1, 1, 1, 1], 1)
        hr2[0, 1] = 4
        self.assertEqual(hr2[0, 1], 4)
        self.assertEqual(hr2._n_body_tensors[(1, 1, 0, 0)][0, 3, 3, 0], 2)
        self.assertEqual(hr2._n_body_tensors[(1, 1, 0, 0)][1, 2, 2, 1], 2)
        self.assertEqual(hr2._n_body_tensors[(1, 1, 0, 0)][0, 2, 2, 0], 2)
        self.assertEqual(hr2._n_body_tensors[(1, 1, 0, 0)][1, 3, 3, 1], 2)

    def test_hr2_ops(self):
        hr2a_mat = numpy.zeros((2, 2))
        hr2a = _HR2(hr2a_mat, self.n_body_tensors)

        hr2b_mat = numpy.zeros((2, 2))
        hr2b = _HR2(hr2b_mat, self.n_body_tensors)

        hr2a[0, 1] = 2
        hr2c = hr2a + hr2b
        self.assertEqual(hr2c[0, 1], 2)

        hr2c = hr2b - hr2a
        self.assertEqual(hr2c[0, 1], -2)

        hr2c = hr2a * 2
        self.assertEqual(hr2c[0, 1], 4)

        hr2c = hr2a / 2
        self.assertAlmostEqual(hr2c[0, 1], 1)

    def test_hr2_shape(self):
        hr2_mat = numpy.zeros((2, 2))
        hr2 = _HR2(hr2_mat, self.n_body_tensors)
        self.assertEqual(len(hr2.shape), 2)
        self.assertEqual(hr2.shape[0], 2)

    def test_hr2_raises_errors(self):
        hr2_mat = numpy.zeros((2, 2))
        hr2 = _HR2(hr2_mat, self.n_body_tensors)
        with self.assertRaises(IndexError):
            _ = hr2[1, 2, 3]
        with self.assertRaises(IndexError):
            hr2[1, 2, 3] = 2

    def test_hc_init_set_get(self):
        hc_mat = numpy.zeros((2))
        hc = _HC(hc_mat, self.n_body_tensors)
        hc[0] = 2
        self.assertEqual(hc[0], 2)
        self.assertEqual(hc[(0, )], 2)
        self.assertEqual(hc._n_body_tensors[(1, 0)][0, 0], 1)
        hc[(0, )] = 4
        self.assertEqual(hc[0], 4)
        self.assertEqual(hc[(0, )], 4)
        self.assertEqual(hc._n_body_tensors[(1, 0)][0, 0], 2)

    def test_hc_raises_errors(self):
        hc_mat = numpy.zeros((2))
        hc = _HC(hc_mat, self.n_body_tensors)
        with self.assertRaises(IndexError):
            _ = hc[0, 0]
        with self.assertRaises(IndexError):
            hc[0, 0] = 1

    def test_hc_ops(self):
        hca_mat = numpy.zeros((2))
        hca = _HC(hca_mat, self.n_body_tensors)

        hcb_mat = numpy.zeros((2))
        hcb = _HC(hcb_mat, self.n_body_tensors)

        hca[0] = 2
        hcc = hca + hcb
        self.assertEqual(hcc[0], 2)

        hcc = hcb - hca
        self.assertEqual(hcc[0], -2)

        hcc = hca * 2
        self.assertEqual(hcc[0], 4)

        hcc = hca / 2
        self.assertAlmostEqual(hcc[0], 1)


class IntegralTransformsTest(unittest.TestCase):

    def setUp(self):
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.filename = os.path.join(DATA_DIRECTORY,
                                     'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      filename=self.filename)
        self.molecule.load()

    def test_integrals_self_inverse(self):
        hc, hr1, hr2 = get_doci_from_integrals(
            self.molecule.one_body_integrals,
            self.molecule.two_body_integrals)
        proj_one_body, proj_two_body = get_projected_integrals_from_doci(hc,
                                                                         hr1,
                                                                         hr2)
        hc_test, hr1_test, hr2_test = get_doci_from_integrals(
            proj_one_body, proj_two_body)
        self.assertTrue(numpy.allclose(hc, hc_test))
        self.assertTrue(numpy.allclose(hr1, hr1_test))
        print(hr2)
        print(hr2_test)
        self.assertTrue(numpy.allclose(hr2, hr2_test))

    def test_integrals_to_doci(self):
        one_body_integrals = self.molecule.one_body_integrals
        two_body_integrals = self.molecule.two_body_integrals
        hc, hr1, hr2 = get_doci_from_integrals(one_body_integrals, two_body_integrals)
        self.assertEqual(hc.shape[0], 2)
        self.assertEqual(hr1.shape[0], 2)
        self.assertEqual(hr2.shape[0], 2)

        for p in range(2):
            self.assertEqual(hc[p] + hr2[p, p],
                             2 * one_body_integrals[p, p] +
                             two_body_integrals[p, p, p, p])
            for q in range(2):
                if p != q:
                    self.assertEqual(hr1[p, q], two_body_integrals[p, p, q, q])
                    self.assertEqual(hr2[p, q], 2*two_body_integrals[p, q, q, p] -
                                     two_body_integrals[p, q, p, q])



class DOCIHamiltonianTest(unittest.TestCase):

    def setUp(self):
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.filename = os.path.join(DATA_DIRECTORY,
                                     'H2_sto-3g_singlet_0.7414')
        self.molecule = MolecularData(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      filename=self.filename)
        self.molecule.load()

    def test_error(self):
        doci_hamiltonian = DOCIHamiltonian.from_integrals(
            constant=self.molecule.nuclear_repulsion,
            one_body_integrals=self.molecule.one_body_integrals,
            two_body_integrals=self.molecule.two_body_integrals)
        with self.assertRaises(TypeError):
            doci_hamiltonian[((1, 0), (0, 1))] = 1
        with self.assertRaises(IndexError):
            _ = doci_hamiltonian[((1, 0), )]
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
        hamiltonian = jordan_wigner(
            self.molecule.get_molecular_hamiltonian())
        doci_hamiltonian = DOCIHamiltonian.from_integrals(
            constant=self.molecule.nuclear_repulsion,
            one_body_integrals=self.molecule.one_body_integrals,
            two_body_integrals=self.molecule.two_body_integrals
        ).qubit_operator

        hamiltonian_matrix = get_sparse_operator(hamiltonian).toarray()
        doci_hamiltonian_matrix = get_sparse_operator(
            doci_hamiltonian).toarray()
        diagonal = numpy.real(numpy.diag(hamiltonian_matrix))
        doci_diagonal = numpy.real(numpy.diag(doci_hamiltonian_matrix))
        position_of_doci_diag_in_h = [0]*len(doci_diagonal)
        for idx, doci_eigval in enumerate(doci_diagonal):
            closest_in_diagonal = None
            for idx2, eig in enumerate(diagonal):
                if closest_in_diagonal is None or abs(eig - doci_eigval) < abs(
                        closest_in_diagonal - doci_eigval):
                    closest_in_diagonal = eig
                    position_of_doci_diag_in_h[idx] = idx2
            assert abs(closest_in_diagonal - doci_eigval) < EQ_TOLERANCE, (
                "Value " + str(doci_eigval) + " of the DOCI Hamiltonian " +
                "diagonal did not appear in the diagonal of the full " +
                "Hamiltonian. The closest value was "+str(closest_in_diagonal))

        sub_matrix = hamiltonian_matrix[numpy.ix_(position_of_doci_diag_in_h,
                                                  position_of_doci_diag_in_h)]
        assert numpy.allclose(doci_hamiltonian_matrix, sub_matrix), (
            "The coupling between the DOCI states in the DOCI Hamiltonian " +
            "should be identical to that between these states in the full " +
            "Hamiltonian bur the DOCI hamiltonian matrix\n" +
            str(doci_hamiltonian_matrix) +
            "\ndoes not match the corresponding sub-matrix of the full " +
            "Hamiltonian\n"+str(sub_matrix))
