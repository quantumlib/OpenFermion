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
                             QuadraticHamiltonian,
                             QubitOperator)


def random_qubit_operator(n_qubits=16,
                          max_num_terms=16,
                          max_many_body_order=16,
                          seed=None):
    prng = numpy.random.RandomState(seed)
    op = QubitOperator()
    num_terms = prng.randint(1, max_num_terms+1)
    for _ in range(num_terms):
        many_body_order = prng.randint(max_many_body_order+1)
        term = []
        for _ in range(many_body_order):
            index = prng.randint(n_qubits)
            action = prng.choice(('X', 'Y', 'Z'))
            term.append((index, action))
        coefficient = prng.randn()
        op += QubitOperator(term, coefficient)
    return op


def haar_random_vector(n, seed=None):
    """Generate an n dimensional Haar randomd vector."""
    if seed is not None:
        numpy.random.seed(seed)
    vector = numpy.random.randn(n).astype(complex)
    vector += 1.j * numpy.random.randn(n).astype(complex)
    normalization = numpy.sqrt(vector.dot(numpy.conjugate(vector)))
    return vector / normalization


def random_antisymmetric_matrix(n, real=False, seed=None):
    """Generate a random n x n antisymmetric matrix."""
    if seed is not None:
        numpy.random.seed(seed)

    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    antisymmetric_mat = rand_mat - rand_mat.T
    return antisymmetric_mat


def random_diagonal_coulomb_hamiltonian(n_qubits, real=False, seed=None):
    """Generate a random instance of DiagonalCoulombHamiltonian.

    Args:
        n_qubits: The number of qubits
        real: Whether to use only real numbers in the one-body term
    """
    if seed is not None:
        numpy.random.seed(seed)

    one_body = random_hermitian_matrix(n_qubits, real=real)
    two_body = random_hermitian_matrix(n_qubits, real=True)
    constant = numpy.random.randn()
    return DiagonalCoulombHamiltonian(one_body, two_body, constant)


def random_hermitian_matrix(n, real=False, seed=None):
    """Generate a random n x n Hermitian matrix."""
    if seed is not None:
        numpy.random.seed(seed)

    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    hermitian_mat = rand_mat + rand_mat.T.conj()
    return hermitian_mat


def random_interaction_operator(
        n_orbitals, expand_spin=False, real=True, seed=None):
    """Generate a random instance of InteractionOperator.

    Args:
        n_orbitals: The number of orbitals.
        expand_spin: Whether to expand each orbital symmetrically into two
            spin orbitals. Note that if this option is set to True, then
            the total number of orbitals will be doubled.
        real: Whether to use only real numbers.
        seed: A random number generator seed.
    """
    if seed is not None:
        numpy.random.seed(seed)

    if real:
        dtype = float
    else:
        dtype = complex

    # The constant has to be real.
    constant = numpy.random.randn()

    # The one-body tensor is a random Hermitian matrix.
    one_body_coefficients = random_hermitian_matrix(n_orbitals, real)

    # Generate random two-body coefficients.
    two_body_coefficients = numpy.zeros((n_orbitals, n_orbitals,
                                         n_orbitals, n_orbitals), dtype)
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        coeff = numpy.random.randn()
        if not real and len(set([p,q,r,s])) >= 3:
            coeff += 1.j * numpy.random.randn()

        # Four point symmetry.
        two_body_coefficients[p, q, r, s] = coeff
        two_body_coefficients[q, p, s, r] = coeff
        two_body_coefficients[s, r, q, p] = coeff.conjugate()
        two_body_coefficients[r, s, p, q] = coeff.conjugate()

        # Eight point symmetry.
        if real:
            two_body_coefficients[r, q, p, s] = coeff
            two_body_coefficients[p, s, r, q] = coeff
            two_body_coefficients[s, p, q, r] = coeff
            two_body_coefficients[q, r, s, p] = coeff

    # If requested, expand to spin orbitals.
    if expand_spin:
        n_spin_orbitals = 2 * n_orbitals

        # Expand one-body tensor.
        one_body_coefficients = numpy.kron(one_body_coefficients, numpy.eye(2))

        # Expand two-body tensor.
        new_two_body_coefficients = numpy.zeros((
            n_spin_orbitals, n_spin_orbitals,
            n_spin_orbitals, n_spin_orbitals), dtype=complex)
        for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
            coefficient = two_body_coefficients[p, q, r, s]

            # Mixed spin.
            new_two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                coefficient)
            new_two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                coefficient)

            # Same spin.
            new_two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = coefficient
            new_two_body_coefficients[2 * p + 1, 2 * q + 1,
                                      2 * r + 1, 2 * s + 1] = coefficient
        two_body_coefficients = new_two_body_coefficients

    # Create the InteractionOperator.
    interaction_operator = InteractionOperator(
        constant, one_body_coefficients, two_body_coefficients)

    return interaction_operator


def random_quadratic_hamiltonian(n_orbitals,
                                 conserves_particle_number=False,
                                 real=False,
                                 expand_spin=False,
                                 seed=None):
    """Generate a random instance of QuadraticHamiltonian.

    Args:
        n_orbitals(int): the number of orbitals
        conserves_particle_number(bool): whether the returned Hamiltonian
            should conserve particle number
        real(bool): whether to use only real numbers
        expand_spin: Whether to expand each orbital symmetrically into two
            spin orbitals. Note that if this option is set to True, then
            the total number of orbitals will be doubled.

    Returns:
        QuadraticHamiltonian
    """
    if seed is not None:
        numpy.random.seed(seed)

    constant = numpy.random.randn()
    chemical_potential = numpy.random.randn()
    hermitian_mat = random_hermitian_matrix(n_orbitals, real)

    if conserves_particle_number:
        antisymmetric_mat = None
    else:
        antisymmetric_mat = random_antisymmetric_matrix(n_orbitals, real)

    if expand_spin:
        hermitian_mat = numpy.kron(hermitian_mat, numpy.eye(2))
        if antisymmetric_mat is not None:
            antisymmetric_mat = numpy.kron(antisymmetric_mat, numpy.eye(2))

    return QuadraticHamiltonian(hermitian_mat, antisymmetric_mat,
                                constant, chemical_potential)


def random_unitary_matrix(n, real=False, seed=None):
    """Obtain a random n x n unitary matrix."""
    if seed is not None:
        numpy.random.seed(seed)

    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    Q, _ = qr(rand_mat)
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


def module_importable(module):
    """Without importing it, returns whether python module is importable.

    Args:
        module (string): Name of module.

    Returns:
        bool

    """
    import sys
    if sys.version_info >= (3, 4):
        from importlib import util
        plug_spec = util.find_spec(module)
    else:
        import pkgutil
        plug_spec = pkgutil.find_loader(module)
    if plug_spec is None:
        return False
    else:
        return True
