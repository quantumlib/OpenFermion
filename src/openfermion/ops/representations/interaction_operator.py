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
"""Class and functions to store interaction operators."""
import itertools

import numpy

from openfermion.ops.representations.polynomial_tensor import PolynomialTensor
from openfermion.config import EQ_TOLERANCE


class InteractionOperatorError(Exception):
    pass


class InteractionOperator(PolynomialTensor):
    r"""Class for storing 'interaction operators' which are defined to be
    fermionic operators consisting of one-body and two-body terms which
    conserve particle number and spin. The most common examples of data that
    will use this structure are molecular Hamiltonians. In principle,
    everything stored in this class could also be represented using the more
    general FermionOperator class. However, this class is able to exploit
    specific properties of how fermions interact to enable more numerically
    efficient manipulation of the data. Note that the operators stored in this
    class take the form:

        $$
            constant + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s.
        $$

    Attributes:
        one_body_tensor: The coefficients of the one-body terms
        ($h_{p, q}$). This is an n_qubits x n_qubits
        numpy array of floats.
        two_body_tensor: The coefficients of the two-body terms
            ($h_{p, q, r, s}$).
            This is an n_qubits x n_qubits x n_qubits x
            n_qubits numpy array of floats.
    """

    def __init__(self, constant, one_body_tensor, two_body_tensor):
        """
        Initialize the InteractionOperator class.

        Args:
            constant: A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
            one_body_tensor: The coefficients of the one-body terms
                ($h_{p,q}$).
               This is an n_qubits x n_qubits numpy array of floats.
            two_body_tensor: The coefficients of the two-body terms
                ($h_{p, q, r, s}$).
                This is an n_qubits x n_qubits x n_qubits x
                n_qubits numpy array of floats.
        """
        # Make sure nonzero elements are only for normal ordered terms.
        super(InteractionOperator, self).__init__({
            (): constant,
            (1, 0): one_body_tensor,
            (1, 1, 0, 0): two_body_tensor
        })

    @property
    def one_body_tensor(self):
        """The value of the one-body tensor."""
        return self.n_body_tensors[1, 0]

    @one_body_tensor.setter
    def one_body_tensor(self, value):
        """Set the value of the one-body tensor."""
        self.n_body_tensors[1, 0] = value

    @property
    def two_body_tensor(self):
        """The value of the two-body tensor."""
        return self.n_body_tensors[1, 1, 0, 0]

    @two_body_tensor.setter
    def two_body_tensor(self, value):
        """Set the value of the two-body tensor."""
        self.n_body_tensors[1, 1, 0, 0] = value

    def unique_iter(self, complex_valued=False):
        """
        Iterate all terms that are not in the same symmetry group.

        Four point symmetry:
            1. pq = qp.
            2. pqrs = srqp = qpsr = rspq.
        Eight point symmetry:
            1. pq = qp.
            2. pqrs = rqps = psrq = srqp = qpsr = rspq = spqr = qrsp.

        Args:
            complex_valued (bool):
                Whether the operator has complex coefficients.
        Yields:
            tuple[int]
        """
        # Constant.
        if self.constant:
            yield ()

        # One-body terms.
        for p in range(self.n_qubits):
            for q in range(p + 1):
                if self.one_body_tensor[p, q]:
                    yield (p, 1), (q, 0)

        # Two-body terms.
        seen = set()
        for quad in itertools.product(range(self.n_qubits), repeat=4):
            if self.two_body_tensor[quad] and quad not in seen:
                seen |= set(_symmetric_two_body_terms(quad, complex_valued))
                yield tuple(zip(quad, (1, 1, 0, 0)))

    @classmethod
    def zero(cls, n_qubits):
        return cls(0, numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128),
                   numpy.zeros((n_qubits,) * 4, dtype=numpy.complex128))

    def projected(self, indices, exact=False):
        projected_n_body_tensors = self.projected_n_body_tensors(indices, exact)
        return type(self)(*(projected_n_body_tensors[key]
                            for key in [(), (1, 0), (1, 1, 0, 0)]))

    def with_function_applied_elementwise(self, func):
        return type(self)(*(
            func(tensor) for tensor in
            [self.constant, self.one_body_tensor, self.two_body_tensor]))


def _symmetric_two_body_terms(quad, complex_valued):
    p, q, r, s = quad
    yield p, q, r, s
    yield q, p, s, r
    yield s, r, q, p
    yield r, s, p, q
    if not complex_valued:
        yield p, s, r, q
        yield q, r, s, p
        yield s, p, q, r
        yield r, q, p, s


def get_tensors_from_integrals(one_body_integrals, two_body_integrals):
    '''Converts one and two-body integrals into tensor form

    Arguments:
        one_body_integrals [numpy array] -- the one-body integrals
            of the given Hamiltonian
        two_body_integrals [numpy array] -- the two-body integrals
            of the given Hamiltonian
    '''

    n_qubits = 2 * one_body_integrals.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
    two_body_coefficients = numpy.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s] /
                                                2.)
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s] /
                                                2.)

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s] /
                                                2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s] /
                                                2.)

    # Truncate.
    one_body_coefficients[
        numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    return one_body_coefficients, two_body_coefficients


def get_active_space_integrals(one_body_integrals,
                               two_body_integrals,
                               occupied_indices=None,
                               active_indices=None):
    """Restricts a molecule at a spatial orbital level to an active space

    This active space may be defined by a list of active indices and
        doubly occupied indices. Note that one_body_integrals and
        two_body_integrals must be defined
        n an orthonormal basis set.

    Args:
        one_body_integrals: One-body integrals of the target Hamiltonian
        two_body_integrals: Two-body integrals of the target Hamiltonian
        occupied_indices: A list of spatial orbital indices
            indicating which orbitals should be considered doubly occupied.
        active_indices: A list of spatial orbital indices indicating
            which orbitals should be considered active.

    Returns:
        tuple: Tuple with the following entries:

        **core_constant**: Adjustment to constant shift in Hamiltonian
        from integrating out core orbitals

        **one_body_integrals_new**: one-electron integrals over active
        space.

        **two_body_integrals_new**: two-electron integrals over active
        space.
    """
    # Fix data type for a few edge cases
    occupied_indices = [] if occupied_indices is None else occupied_indices
    if (len(active_indices) < 1):
        raise ValueError('Some active indices required for reduction.')

    # Determine core constant
    core_constant = 0.0
    for i in occupied_indices:
        core_constant += 2 * one_body_integrals[i, i]
        for j in occupied_indices:
            core_constant += (2 * two_body_integrals[i, j, j, i] -
                              two_body_integrals[i, j, i, j])

    # Modified one electron integrals
    one_body_integrals_new = numpy.copy(one_body_integrals)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                one_body_integrals_new[u, v] += (
                    2 * two_body_integrals[i, u, v, i] -
                    two_body_integrals[i, u, i, v])

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_new[numpy.ix_(active_indices, active_indices)],
            two_body_integrals[numpy.ix_(active_indices, active_indices,
                                         active_indices, active_indices)])
