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

"""Functions useful for tests."""

import collections
import itertools

import numpy
from scipy.linalg import qr

from openfermion.ops import (DiagonalCoulombHamiltonian,
                             InteractionOperator,
                             QuadraticHamiltonian)


def random_antisymmetric_matrix(n, real=False):
    """Generate a random n x n antisymmetric matrix."""
    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    antisymmetric_mat = rand_mat - rand_mat.T
    return antisymmetric_mat


def random_diagonal_coulomb_hamiltonian(n_qubits, real=False):
    """Generate a random instance of DiagonalCoulombHamiltonian.

    Args:
        n_qubits: The number of qubits
        real: Whether to use only real numbers in the one-body term
    """
    one_body = random_hermitian_matrix(n_qubits, real=real)
    two_body = random_hermitian_matrix(n_qubits, real=True)
    constant = numpy.random.randn()
    return DiagonalCoulombHamiltonian(one_body, two_body, constant)


def random_hermitian_matrix(n, real=False):
    """Generate a random n x n Hermitian matrix."""
    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    hermitian_mat = rand_mat + rand_mat.T.conj()
    return hermitian_mat


def random_interaction_operator(n_qubits, real=True):
    """Generate a random instance of InteractionOperator."""
    if real:
        dtype = float
    else:
        dtype = complex

    # The constant has to be real
    constant = numpy.random.randn()

    # The one-body tensor is a random Hermitian matrix
    one_body_coefficients = random_hermitian_matrix(n_qubits, real)

    # Generate random two-body coefficients
    two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                         n_qubits, n_qubits), dtype)
    # Generate "diagonal" terms, which are necessarily real
    for p, q in itertools.combinations(range(n_qubits), 2):
        coeff = numpy.random.randn()
        two_body_coefficients[p, q, p, q] = coeff
        two_body_coefficients[p, q, q, p] = -coeff
        two_body_coefficients[q, p, q, p] = coeff
    # Generate the rest of the terms
    for (p, q), (r, s) in itertools.combinations(
            itertools.combinations(range(n_qubits), 2),
            2):
        coeff = numpy.random.randn()
        if not real:
            coeff += 1.j * numpy.random.randn()
        two_body_coefficients[p, q, r, s] = coeff
        two_body_coefficients[p, q, s, r] = -coeff
        two_body_coefficients[q, p, r, s] = -coeff
        two_body_coefficients[q, p, s, r] = coeff

        two_body_coefficients[s, r, q, p] = coeff.conjugate()
        two_body_coefficients[s, r, p, q] = -coeff.conjugate()
        two_body_coefficients[r, s, q, p] = -coeff.conjugate()
        two_body_coefficients[r, s, p, q] = coeff.conjugate()

    # Create the InteractionOperator and return
    interaction_operator = InteractionOperator(
        constant, one_body_coefficients, two_body_coefficients)

    return interaction_operator


def random_quadratic_hamiltonian(n_qubits,
                                 conserves_particle_number=False,
                                 real=False):
    """Generate a random instance of QuadraticHamiltonian.

    Args:
        n_qubits(int): the number of qubits
        conserves_particle_number(bool): whether the returned Hamiltonian
            should conserve particle number
        real(bool): whether to use only real numbers

    Returns:
        QuadraticHamiltonian
    """
    constant = numpy.random.randn()
    chemical_potential = numpy.random.randn()
    hermitian_mat = random_hermitian_matrix(n_qubits, real)
    if conserves_particle_number:
        antisymmetric_mat = None
    else:
        antisymmetric_mat = random_antisymmetric_matrix(n_qubits, real)
    return QuadraticHamiltonian(hermitian_mat, antisymmetric_mat,
                                constant, chemical_potential)


def random_unitary_matrix(n, real=False):
    """Obtain a random n x n unitary matrix."""
    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    Q, R = qr(rand_mat)
    return Q


class EqualsTester(object):
    """Tests equality against user-provided disjoint equivalence groups."""

    def __init__(self, test_case):
        self.groups = [(_ClassUnknownToSubjects(),)]
        self.test_case = test_case

    def add_equality_group(self, *group_items):
        """Tries to add a disjoint equivalence group to the equality tester.
        This methods asserts that items within the group must all be equal to
        each other, but not equal to any items in other groups that have been
        or will be added.

        Args:
            *group_items: The items making up the equivalence group.

        Raises:
            AssertError: Items within the group are not equal to each other, or
                items in another group are equal to items within the new group,
                or the items violate the equals-implies-same-hash rule.
        """

        self.test_case.assertIsNotNone(group_items)

        # Check that group items are equivalent to each other.
        for v1, v2 in itertools.product(group_items, repeat=2):
            # Binary operators should always work.
            self.test_case.assertTrue(v1 == v2)
            self.test_case.assertTrue(not v1 != v2)

            # __eq__ and __ne__ should both be correct or not implemented.
            self.test_case.assertTrue(
                hasattr(v1, '__eq__') == hasattr(v1, '__ne__'))
            # Careful: python2 int doesn't have __eq__ or __ne__.
            if hasattr(v1, '__eq__'):
                eq = v1.__eq__(v2)
                ne = v1.__ne__(v2)
                self.test_case.assertIn(
                    (eq, ne),
                    [(True, False),
                     (NotImplemented, False),
                     (NotImplemented, NotImplemented)])

        # Check that this group's items don't overlap with other groups.
        for other_group in self.groups:
            for v1, v2 in itertools.product(group_items, other_group):
                # Binary operators should always work.
                self.test_case.assertTrue(not v1 == v2)
                self.test_case.assertTrue(v1 != v2)

                # __eq__ and __ne__ should both be correct or not implemented.
                self.test_case.assertTrue(
                    hasattr(v1, '__eq__') == hasattr(v1, '__ne__'))
                # Careful: python2 int doesn't have __eq__ or __ne__.
                if hasattr(v1, '__eq__'):
                    eq = v1.__eq__(v2)
                    ne = v1.__ne__(v2)
                    self.test_case.assertIn(
                        (eq, ne),
                        [(False, True),
                         (NotImplemented, True),
                         (NotImplemented, NotImplemented)])

        # Check that group items hash to the same thing, or are all unhashable.
        hashes = [hash(v) if isinstance(v, collections.Hashable) else None
                  for v in group_items]
        if len(set(hashes)) > 1:
            examples = ((v1, h1, v2, h2)
                        for v1, h1 in zip(group_items, hashes)
                        for v2, h2 in zip(group_items, hashes)
                        if h1 != h2)
            example = next(examples)
            raise AssertionError(
                'Items in the same group produced different hashes. '
                'Example: hash({}) is {} but hash({}) is {}.'.format(*example))

        # Remember this group, to enable disjoint checks vs later groups.
        self.groups.append(group_items)

    def make_equality_pair(self, factory):
        """Tries to add a disjoint (item, item) group to the equality tester.
        Uses the factory method to produce two different objects containing
        equal items. Asserts that the two object are equal, but not equal to
        any items in other groups that have been or will be added. Adds the
        pair as a group.

        Args:
            factory (Callable[[], Any]): A method for producing independent
                copies of an item.

        Raises:
            AssertError: The factory produces items not equal to each other, or
                items in another group are equal to items from the factory, or
                the items violate the equal-implies-same-hash rule.
        """
        self.add_equality_group(factory(), factory())


class _ClassUnknownToSubjects(object):
    """Equality methods should be able to deal with the unexpected."""

    def __eq__(self, other):
        return isinstance(other, _ClassUnknownToSubjects)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(_ClassUnknownToSubjects)
