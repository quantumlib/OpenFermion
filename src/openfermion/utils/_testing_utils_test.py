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

import itertools
import fractions
import unittest

import numpy

from openfermion.ops import QubitOperator
from openfermion.transforms import get_fermion_operator
from openfermion.utils import count_qubits, is_hermitian
from openfermion.utils._testing_utils import (
        EqualsTester,
        haar_random_vector,
        random_antisymmetric_matrix,
        random_diagonal_coulomb_hamiltonian,
        random_hermitian_matrix,
        random_interaction_operator,
        random_quadratic_hamiltonian,
        random_qubit_operator,
        random_unitary_matrix)


def test_random_qubit_operator():
    op = random_qubit_operator(
        n_qubits=20,
        max_num_terms=20,
        max_many_body_order=20
    )

    assert isinstance(op, QubitOperator)
    assert op.many_body_order() <= 20
    assert len(op.terms) <= 20
    assert count_qubits(op) <= 20



class EqualsTesterTest(unittest.TestCase):

    def test_add_equality_group_correct(self):
        eq = EqualsTester(self)

        eq.add_equality_group(fractions.Fraction(1, 1))

        eq.add_equality_group(fractions.Fraction(1, 2),
                              fractions.Fraction(2, 4))

        eq.add_equality_group(
            fractions.Fraction(2, 3),
            fractions.Fraction(12, 18), fractions.Fraction(14, 21))

        eq.add_equality_group(2, 2.0, fractions.Fraction(2, 1))

        eq.add_equality_group([1, 2, 3], [1, 2, 3])

        eq.add_equality_group({'b': 3, 'a': 2}, {'a': 2, 'b': 3})

        eq.add_equality_group('unrelated')

    def test_assert_add_equality_pair(self):
        eq = EqualsTester(self)

        with self.assertRaises(AssertionError):
            eq.make_equality_pair(object)

        eq.make_equality_pair(lambda: 1)
        eq.make_equality_pair(lambda: 2)
        eq.add_equality_group(3)

        with self.assertRaises(AssertionError):
            eq.add_equality_group(1)
        with self.assertRaises(AssertionError):
            eq.make_equality_pair(lambda: 1)
        with self.assertRaises(AssertionError):
            eq.make_equality_pair(lambda: 3)

    def test_add_equality_group_not_equivalent(self):
        eq = EqualsTester(self)
        with self.assertRaises(AssertionError):
            eq.add_equality_group(1, 2)

    def test_add_equality_group_not_disjoint(self):
        eq = EqualsTester(self)
        eq.add_equality_group(1)
        with self.assertRaises(AssertionError):
            eq.add_equality_group(1)

    def test_add_equality_group_bad_hash(self):
        class KeyHash(object):
            def __init__(self, k, h):
                self._k = k
                self._h = h

            def __eq__(self, other):
                return isinstance(other, KeyHash) and self._k == other._k

            def __ne__(self, other):
                return not self == other

            def __hash__(self):
                return self._h

        eq = EqualsTester(self)
        eq.add_equality_group(KeyHash('a', 5), KeyHash('a', 5))
        eq.add_equality_group(KeyHash('b', 5))
        with self.assertRaises(AssertionError):
            eq.add_equality_group(KeyHash('c', 2), KeyHash('c', 3))

    def test_add_equality_group_exception_hash(self):
        class FailHash(object):
            def __hash__(self):
                raise ValueError('injected failure')

        eq = EqualsTester(self)
        with self.assertRaises(ValueError):
            eq.add_equality_group(FailHash())

    def test_can_fail_when_forgot_type_check(self):
        eq = EqualsTester(self)

        class NoTypeCheckEqualImplementation(object):
            def __init__(self):
                self.x = 1

            def __eq__(self, other):
                return self.x == other.x

            def __ne__(self, other):
                return not self == other

            def __hash__(self):
                return hash(self.x)

        with self.assertRaises(AttributeError):
            eq.add_equality_group(NoTypeCheckEqualImplementation())

    def test_fails_hash_is_default_and_inconsistent(self):
        eq = EqualsTester(self)

        class DefaultHashImplementation(object):
            __hash__ = object.__hash__

            def __init__(self):
                self.x = 1

            def __eq__(self, other):
                if not isinstance(other, type(self)):
                    return NotImplemented
                return self.x == other.x

            def __ne__(self, other):
                return not self == other

        with self.assertRaises(AssertionError):
            eq.make_equality_pair(DefaultHashImplementation)

    def test_fails_when_ne_is_inconsistent(self):
        eq = EqualsTester(self)

        class InconsistentNeImplementation(object):
            def __init__(self):
                self.x = 1

            def __eq__(self, other):
                if not isinstance(other, type(self)):
                    return NotImplemented
                return self.x == other.x

            def __ne__(self, other):
                return NotImplemented

            def __hash__(self):
                return hash(self.x)

        with self.assertRaises(AssertionError):
            eq.make_equality_pair(InconsistentNeImplementation)

    def test_fails_when_not_reflexive(self):
        eq = EqualsTester(self)

        class NotReflexiveImplementation(object):
            def __init__(self):
                self.x = 1

            def __eq__(self, other):
                if other is not self:
                    return NotImplemented
                return False

            def __ne__(self, other):
                return not self == other

        with self.assertRaises(AssertionError):
            eq.add_equality_group(NotReflexiveImplementation())

    def test_fails_when_not_commutative(self):
        eq = EqualsTester(self)

        class NotCommutativeImplementation(object):
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                if not isinstance(other, type(self)):
                    return NotImplemented
                return self.x <= other.x

            def __ne__(self, other):
                return not self == other

        with self.assertRaises(AssertionError):
            eq.add_equality_group(NotCommutativeImplementation(0),
                                  NotCommutativeImplementation(1))

        with self.assertRaises(AssertionError):
            eq.add_equality_group(NotCommutativeImplementation(1),
                                  NotCommutativeImplementation(0))


class RandomInteractionOperatorTest(unittest.TestCase):

    def test_hermiticity(self):
        n_orbitals = 5

        # Real, no spin
        iop = random_interaction_operator(n_orbitals, real=True)
        ferm_op = get_fermion_operator(iop)
        self.assertTrue(is_hermitian(ferm_op))

        # Real, spin
        iop = random_interaction_operator(
                n_orbitals, expand_spin=True, real=True)
        ferm_op = get_fermion_operator(iop)
        self.assertTrue(is_hermitian(ferm_op))

        # Complex, no spin
        iop = random_interaction_operator(n_orbitals, real=False)
        ferm_op = get_fermion_operator(iop)
        self.assertTrue(is_hermitian(ferm_op))

        # Complex, spin
        iop = random_interaction_operator(
                n_orbitals, expand_spin=True, real=False)
        ferm_op = get_fermion_operator(iop)
        self.assertTrue(is_hermitian(ferm_op))

    def test_symmetry(self):
        n_orbitals = 5

        # Real.
        iop = random_interaction_operator(n_orbitals, expand_spin=False,
                                          real=True)
        ferm_op = get_fermion_operator(iop)
        self.assertTrue(is_hermitian(ferm_op))
        two_body_coefficients = iop.two_body_tensor
        for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[r, q, p, s])

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[p, s, r, q])

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[s, r, q, p])

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[q, p, s, r])

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[r, s, p, q])

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[s, p, q, r])

            self.assertAlmostEqual(two_body_coefficients[p, q, r, s],
                                   two_body_coefficients[q, r, s, p])


class HaarRandomVectorTest(unittest.TestCase):

    def test_vector_norm(self):
        n = 15
        seed = 8317
        vector = haar_random_vector(n, seed)
        norm = vector.dot(numpy.conjugate(vector))
        self.assertAlmostEqual(1. + 0.j, norm)


class RandomSeedingTest(unittest.TestCase):

    def test_random_operators_are_reproducible(self):
        op1 = random_diagonal_coulomb_hamiltonian(5, seed=5947)
        op2 = random_diagonal_coulomb_hamiltonian(5, seed=5947)
        numpy.testing.assert_allclose(op1.one_body, op2.one_body)
        numpy.testing.assert_allclose(op1.two_body, op2.two_body)

        op1 = random_interaction_operator(5, seed=8911)
        op2 = random_interaction_operator(5, seed=8911)
        numpy.testing.assert_allclose(op1.one_body_tensor, op2.one_body_tensor)
        numpy.testing.assert_allclose(op1.two_body_tensor, op2.two_body_tensor)

        op1 = random_quadratic_hamiltonian(5, seed=17711)
        op2 = random_quadratic_hamiltonian(5, seed=17711)
        numpy.testing.assert_allclose(op1.combined_hermitian_part,
                                      op2.combined_hermitian_part)
        numpy.testing.assert_allclose(op1.antisymmetric_part,
                                      op2.antisymmetric_part)

        op1 = random_antisymmetric_matrix(5, seed=24074)
        op2 = random_antisymmetric_matrix(5, seed=24074)
        numpy.testing.assert_allclose(op1, op2)

        op1 = random_hermitian_matrix(5, seed=56753)
        op2 = random_hermitian_matrix(5, seed=56753)
        numpy.testing.assert_allclose(op1, op2)

        op1 = random_unitary_matrix(5, seed=56486)
        op2 = random_unitary_matrix(5, seed=56486)
        numpy.testing.assert_allclose(op1, op2)
