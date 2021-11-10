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
"""Tests for bravyi_kitaev.py."""

import unittest

import numpy
import sympy

from openfermion.ops.operators import (FermionOperator, MajoranaOperator,
                                       QubitOperator)

from openfermion.transforms.opconversions import (jordan_wigner, bravyi_kitaev,
                                                  get_fermion_operator,
                                                  normal_ordered)

from openfermion.transforms import (get_interaction_operator)
from openfermion.testing.testing_utils import (random_interaction_operator)
from openfermion.utils.operator_utils import (count_qubits)
from openfermion.linalg import eigenspectrum
from openfermion.hamiltonians import number_operator


class BravyiKitaevTransformTest(unittest.TestCase):

    def test_bravyi_kitaev_transform(self):
        # Check that the QubitOperators are two-term.
        lowering = bravyi_kitaev(FermionOperator(((3, 0),)))
        raising = bravyi_kitaev(FermionOperator(((3, 1),)))
        self.assertEqual(len(raising.terms), 2)
        self.assertEqual(len(lowering.terms), 2)

        #  Test the locality invariant for N=2^d qubits
        # (c_j majorana is always log2N+1 local on qubits)
        n_qubits = 16
        invariant = numpy.log2(n_qubits) + 1
        for index in range(n_qubits):
            operator = bravyi_kitaev(FermionOperator(((index, 0),)), n_qubits)
            qubit_terms = operator.terms.items()  # Get the majorana terms.

            for item in qubit_terms:
                coeff = item[1]

                #  Identify the c majorana terms by real
                #  coefficients and check their length.
                if not isinstance(coeff, complex):
                    self.assertEqual(len(item[0]), invariant)

        #  Hardcoded coefficient test on 16 qubits
        lowering = bravyi_kitaev(FermionOperator(((9, 0),)), n_qubits)
        raising = bravyi_kitaev(FermionOperator(((9, 1),)), n_qubits)

        correct_operators_c = ((7, 'Z'), (8, 'Z'), (9, 'X'), (11, 'X'), (15,
                                                                         'X'))
        correct_operators_d = ((7, 'Z'), (9, 'Y'), (11, 'X'), (15, 'X'))

        self.assertEqual(lowering.terms[correct_operators_c], 0.5)
        self.assertEqual(lowering.terms[correct_operators_d], 0.5j)
        self.assertEqual(raising.terms[correct_operators_d], -0.5j)
        self.assertEqual(raising.terms[correct_operators_c], 0.5)

    def test_bravyi_kitaev_transform_sympy(self):
        # Check that the QubitOperators are two-term.
        coeff = sympy.Symbol('x')

        #  Hardcoded coefficient test on 16 qubits
        n_qubits = 16
        lowering = bravyi_kitaev(FermionOperator(((9, 0),)) * coeff, n_qubits)
        raising = bravyi_kitaev(FermionOperator(((9, 1),)) * coeff, n_qubits)
        sum_lr = bravyi_kitaev(
            FermionOperator(((9, 0),)) * coeff + FermionOperator(
                ((9, 1),)) * coeff, n_qubits)

        correct_operators_c = ((7, 'Z'), (8, 'Z'), (9, 'X'), (11, 'X'), (15,
                                                                         'X'))
        correct_operators_d = ((7, 'Z'), (9, 'Y'), (11, 'X'), (15, 'X'))

        self.assertEqual(lowering.terms[correct_operators_c], 0.5 * coeff)
        self.assertEqual(lowering.terms[correct_operators_d], 0.5j * coeff)
        self.assertEqual(raising.terms[correct_operators_d], -0.5j * coeff)
        self.assertEqual(raising.terms[correct_operators_c], 0.5 * coeff)
        self.assertEqual(len(sum_lr.terms), 1)
        sum_lr_correct = QubitOperator(correct_operators_c, coeff)
        self.assertEqual(sum_lr, sum_lr_correct)

    def test_bk_identity(self):
        self.assertTrue(bravyi_kitaev(FermionOperator(())) == QubitOperator(()))

    def test_bk_n_qubits_too_small(self):
        with self.assertRaises(ValueError):
            bravyi_kitaev(FermionOperator('2^ 3^ 5 0'), n_qubits=4)
        with self.assertRaises(ValueError):
            bravyi_kitaev(MajoranaOperator((2, 3, 9, 0)), n_qubits=4)
        with self.assertRaises(ValueError):
            bravyi_kitaev(get_interaction_operator(
                FermionOperator('2^ 3^ 5 0')),
                          n_qubits=4)

    def test_bk_jw_number_operator(self):
        # Check if number operator has the same spectrum in both
        # BK and JW representations
        n = number_operator(1, 0)
        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_n)
        bk_spectrum = eigenspectrum(bk_n)

        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_number_operators(self):
        # Check if a number operator has the same spectrum in both
        # JW and BK representations
        n_qubits = 2
        n1 = number_operator(n_qubits, 0)
        n2 = number_operator(n_qubits, 1)
        n = n1 + n2

        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_n)
        bk_spectrum = eigenspectrum(bk_n)

        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_number_operator_scaled(self):
        # Check if number operator has the same spectrum in both
        # JW and BK representations
        n_qubits = 1
        n = number_operator(n_qubits, 0, coefficient=2)  # eigenspectrum (0,2)
        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_n)
        bk_spectrum = eigenspectrum(bk_n)

        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_hopping_operator(self):
        # Check if the spectrum fits for a single hoppping operator
        ho = FermionOperator(((1, 1), (4, 0))) + FermionOperator(
            ((4, 1), (1, 0)))
        jw_ho = jordan_wigner(ho)
        bk_ho = bravyi_kitaev(ho)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_ho)
        bk_spectrum = eigenspectrum(bk_ho)

        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_majoranas(self):
        # Check if the Majorana operators have the same spectrum
        # irrespectively of the transform.

        a = FermionOperator(((1, 0),))
        a_dag = FermionOperator(((1, 1),))

        c = a + a_dag
        d = 1j * (a_dag - a)

        c_spins = [jordan_wigner(c), bravyi_kitaev(c)]
        d_spins = [jordan_wigner(d), bravyi_kitaev(d)]

        c_spectrum = [eigenspectrum(c_spins[0]), eigenspectrum(c_spins[1])]
        d_spectrum = [eigenspectrum(d_spins[0]), eigenspectrum(d_spins[1])]

        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(c_spectrum[0] - c_spectrum[1])))
        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(d_spectrum[0] - d_spectrum[1])))

    def test_bk_jw_integration(self):
        # This is a legacy test, which was a minimal failing example when
        # optimization for hermitian operators was used.

        # Minimal failing example:
        fo = FermionOperator(((3, 1),))

        jw = jordan_wigner(fo)
        bk = bravyi_kitaev(fo)

        jw_spectrum = eigenspectrum(jw)
        bk_spectrum = eigenspectrum(bk)

        self.assertAlmostEqual(
            0., numpy.amax(numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_integration_original(self):
        # This is a legacy test, which was an example proposed by Ryan,
        # failing when optimization for hermitian operators was used.
        fermion_operator = FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)),
                                           -4.3)
        fermion_operator += FermionOperator(((3, 1), (1, 0)), 8.17)
        fermion_operator += 3.2 * FermionOperator()

        # Map to qubits and compare matrix versions.
        jw_qubit_operator = jordan_wigner(fermion_operator)
        bk_qubit_operator = bravyi_kitaev(fermion_operator)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_qubit_operator)
        bk_spectrum = eigenspectrum(bk_qubit_operator)
        self.assertAlmostEqual(0.,
                               numpy.amax(
                                   numpy.absolute(jw_spectrum - bk_spectrum)),
                               places=5)

    def test_bk_bad_type(self):
        with self.assertRaises(TypeError):
            bravyi_kitaev(QubitOperator())


def test_bravyi_kitaev_majorana_op_consistent():
    op = (MajoranaOperator((1, 3, 4), 0.5) + MajoranaOperator(
        (3, 7, 8, 9, 10, 12), 1.8) + MajoranaOperator((0, 4)))
    assert bravyi_kitaev(op) == bravyi_kitaev(get_fermion_operator(op))


class BravyiKitaevInterOpTest(unittest.TestCase):

    test_range = 8

    def one_op(self, a, b):
        return FermionOperator(((a, 1), (b, 0)))

    def two_op(self, a, b):
        return self.one_op(a, b) + self.one_op(b, a)

    def coulomb_exchange_operator(self, a, b):
        return FermionOperator(
            ((a, 1), (b, 1), (a, 0), (b, 0))) + FermionOperator(
                ((a, 1), (b, 1), (b, 0), (a, 0)))

    def number_excitation_operator(self, a, b, c):
        return normal_ordered(
            FermionOperator(((a, 1), (b, 1), (b, 0),
                             (c, 0))) + FermionOperator(((b, 1), (c, 1), (a, 0),
                                                         (b, 0))))

    def four_op(self, a, b, c, d):
        return normal_ordered(FermionOperator(((a, 1), (b, 1), (c, 0), (d, 0))))

    def test_case_one_body_op_success(self):
        # Case A: Simplest class of operators
        # (Number operators and Excitation operators)
        for i in range(self.test_range):
            for j in range(i):
                ham = self.two_op(i, j) + self.two_op(j, i)
                n_qubits = count_qubits(ham)

                opf_ham = bravyi_kitaev(ham, n_qubits)
                custom = bravyi_kitaev(get_interaction_operator(ham))

                assert custom == opf_ham

    def test_coulomb_and_exchange_ops_success(self):
        # Case B: Coulomb and exchange operators
        for i in range(self.test_range):
            for j in range(i):
                ham = self.coulomb_exchange_operator(i, j)

                opf_ham = bravyi_kitaev(ham)
                custom = bravyi_kitaev(get_interaction_operator(ham))

                print(opf_ham)
                print(custom)

                assert custom == opf_ham

    def test_number_excitation_op_success(self):
        # Case C: Number-excitation operator
        for i in range(self.test_range):
            for j in range(self.test_range):
                if i != j:
                    for k in range(self.test_range):
                        if k not in (i, j):
                            ham = self.number_excitation_operator(i, j, k)

                            opf_ham = bravyi_kitaev(ham)
                            custom = bravyi_kitaev(
                                get_interaction_operator(ham))

                            assert custom == opf_ham

    def test_double_excitation_op_success(self):
        # Case D: Double-excitation operator
        for i in range(self.test_range):
            for j in range(self.test_range):
                for k in range(self.test_range):
                    for l in range(self.test_range):
                        if len({i, j, k, l}) == 4:
                            print(i, j, k, l)
                            ham = self.four_op(i, j, k, l) + self.four_op(
                                k, l, i, j)
                            n_qubits = count_qubits(ham)

                            opf_ham = bravyi_kitaev(ham, n_qubits)
                            custom = bravyi_kitaev(
                                get_interaction_operator(ham))

                            assert custom == opf_ham

    def test_consistency_for_complex_numbers(self):
        """Test consistency with JW for FermionOperators."""
        # Random interaction operator
        n_qubits = 8
        iop = random_interaction_operator(n_qubits, real=False)
        op1 = bravyi_kitaev(iop)
        op2 = bravyi_kitaev(get_fermion_operator(iop))
        self.assertEqual(op1, op2)
