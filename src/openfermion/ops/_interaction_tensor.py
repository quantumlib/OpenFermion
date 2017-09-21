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

"""Base class for representation for InteractionOperator and InteractionRDM."""
from __future__ import absolute_import

import copy
import numpy

from openfermion.config import *


class InteractionTensorError(Exception):
    pass


def one_body_basis_change(one_body_tensor, rotation_matrix):
    """Change the basis of an 1-body interaction tensor such as the 1-RDM.

    M' = R^T.M.R where R is the rotation matrix, M is the 1-body tensor
    and M' is the transformed 1-body tensor.

    Args:
        one_body_tensor: A square numpy array or matrix containing information
            about a 1-body interaction tensor such as the 1-RDM.
        rotation_matrix: A square numpy array or matrix having dimensions of
            n_qubits by n_qubits. Assumed to be real and invertible.

    Returns:
        transformed_one_body_tensor: one_body_tensor in the rotated basis.
    """
    # If operator acts on spin degrees of freedom, enlarge rotation matrix.
    n_orbitals = rotation_matrix.shape[0]
    if one_body_tensor.shape[0] == 2 * n_orbitals:
        rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))

    # Effect transformation and return.
    transformed_one_body_tensor = numpy.einsum('qp, qr, rs',
                                               rotation_matrix,
                                               one_body_tensor,
                                               rotation_matrix)
    return transformed_one_body_tensor


def two_body_basis_change(two_body_tensor, rotation_matrix):
    """Change the basis of 2-body interaction tensor such as 2-RDM.

    Procedure we use is an N^5 transformation which can be expressed as
    (pq|rs) = \sum_a R^p_a
      (\sum_b R^q_b (\sum_c R^r_c (\sum_d R^s_d (ab|cd)))).

    Args:
        two_body_tensor: a square rank 4 interaction tensor.
        rotation_matrix: A square numpy array or matrix having dimensions of
            n_qubits by n_qubits. Assumed to be real and invertible.

    Returns:
        transformed_two_body_tensor: two_body_tensor matrix in rotated basis.
    """
    # If operator acts on spin degrees of freedom, enlarge rotation matrix.
    n_orbitals = rotation_matrix.shape[0]
    if two_body_tensor.shape[0] == 2 * n_orbitals:
        rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))

    # Effect transformation and return.
    two_body_tensor = numpy.einsum('prsq', two_body_tensor)
    first_sum = numpy.einsum('ds, abcd', rotation_matrix, two_body_tensor)
    second_sum = numpy.einsum('cr, abcs', rotation_matrix, first_sum)
    third_sum = numpy.einsum('bq, abrs', rotation_matrix, second_sum)
    transformed_two_body_tensor = numpy.einsum('ap, aqrs',
                                               rotation_matrix, third_sum)
    transformed_two_body_tensor = numpy.einsum('psqr',
                                               transformed_two_body_tensor)
    return transformed_two_body_tensor


class InteractionTensor(object):
    """Class for storing data about the interactions between orbitals. Because
    electrons interact pairwise, in second-quantization, all Hamiltonian terms
    have either the form of a^\dagger_p a_q or a^\dagger_p a^\dagger_q a_r a_s.
    The first of these terms is associated with the one-body Hamiltonian and
    1-RDM and its information is stored in one_body_tensor. The second of these
    terms is associated with the two-body Hamiltonian and 2-RDM and its
    information is stored in two_body_tensor. Much of the functionality of this
    class is redudant with FermionOperator but enables much more efficient
    numerical computations in many cases, such as basis rotations.

    Attributes:
        n_qubits: The number of qubits on which the tensor
            acts.
        constant: A constant term in the operator given as a float.
            For instance, the nuclear repulsion energy.
        one_body_tensor: The coefficients of the 2D matrix terms. This is an
            n_qubits x n_qubits numpy array of floats. For instance, the one
            body term of MolecularOperator.
        two_body_tensor: The coefficients of the 4D matrix terms. This is an
            n_qubits x n_qubits x n_qubits x n_qubits numpy array offloats.
            For instance, the two body term of MolecularOperator.
    """

    def __init__(self, constant, one_body_tensor, two_body_tensor):
        """Initialize the InteractionTensor class.

        Args:
            constant: A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
            one_body_tensor: The coefficients of the 2D matrix terms. This
                is an n_qubits x n_qubits numpy array of floats. For
                instance, the one body term of MolecularOperator.
            two_body_tensor: The coefficients of the 4D matrix terms. This is
                an n_qubits x n_qubits x n_qubits x n_qubits numpy array of
                floats. For instance, the two body term of MolecularOperator.
        """
        if constant is None:
            constant = 0.0
        self.constant = constant
        self.one_body_tensor = one_body_tensor
        self.two_body_tensor = two_body_tensor
        self.n_qubits = self.one_body_tensor.shape[0]

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            Ints giving indices of tensor. Either p,q or p,q,r,s.

        Raises:
            ValueError: args must be of length 2 or 4.
            ValueError: args must be of length 0, 2 or 4.
        """
        if len(args) == 4:
            p, q, r, s = args
            return self.two_body_tensor[p, q, r, s]
        elif len(args) == 2:
            p, q = args
            return self.one_body_tensor[p, q]
        elif not len(args):
            return self.constant
        else:
            raise ValueError('args must be of length 0, 2, or 4.')

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            Ints giving indices of tensor. Either p,q or p,q,r,s.

        Raises:
            ValueError: args must be of length 2 or 4.
        """
        if len(args) == 4:
            p, q, r, s = args
            self.two_body_tensor[p, q, r, s] = value
        elif len(args) == 2:
            p, q = args
            self.one_body_tensor[p, q] = value
        elif not len(args):
            self.constant = value
        else:
            raise ValueError('args must be of length 0, 2, or 4.')

    def __eq__(self, molecular_tensor):
        constant_diff = abs(molecular_tensor.constant - self.constant)
        diff = max(constant_diff,
                   numpy.amax(
                       numpy.absolute(self.one_body_tensor -
                                      molecular_tensor.one_body_tensor)),
                   numpy.amax(
                       numpy.absolute(self.two_body_tensor -
                                      molecular_tensor.two_body_tensor)))
        return diff < EQ_TOLERANCE

    def __neq__(self, molecular_tensor):
        return not (self == molecular_tensor)

    def __iadd__(self, addend):
        if not issubclass(type(addend), InteractionTensor):
            raise TypeError('Invalid type.')

        if self.n_qubits != addend.n_qubits:
            raise TypeError('Invalid tensor shape.')

        self.constant += addend.constant
        self.one_body_tensor = numpy.add(self.one_body_tensor,
                                         addend.one_body_tensor)
        self.two_body_tensor = numpy.add(self.two_body_tensor,
                                         addend.two_body_tensor)
        return self

    def __add__(self, addend):
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __neg__(self):
        return InteractionTensor(-self.constant,
                                 numpy.negative(self.one_body_tensor),
                                 numpy.negative(self.two_body_tensor))

    def __isub__(self, subtrahend):
        if not issubclass(type(subtrahend), InteractionTensor):
            raise TypeError('Invalid type.')

        if self.n_qubits != subtrahend.n_qubits:
            raise TypeError('Invalid tensor shape.')

        self.constant -= subtrahend.constant
        self.one_body_tensor = numpy.subtract(self.one_body_tensor,
                                              subtrahend.one_body_tensor)
        self.two_body_tensor = numpy.subtract(self.two_body_tensor,
                                              subtrahend.two_body_tensor)
        return self

    def __sub__(self, subtrahend):
        r = copy.deepcopy(self)
        r -= subtrahend
        return r

    def __imul__(self, multiplier):
        if not issubclass(type(multiplier), InteractionTensor):
            raise TypeError('Invalid type.')

        if self.n_qubits != multiplier.n_qubits:
            raise TypeError('Invalid tensor shape.')

        self.constant *= multiplier.constant
        self.one_body_tensor = numpy.multiply(self.one_body_tensor,
                                              multiplier.one_body_tensor)
        self.two_body_tensor = numpy.multiply(self.two_body_tensor,
                                              multiplier.two_body_tensor)
        return self

    def __mul__(self, multiplier):
        product = copy.deepcopy(self)
        product *= multiplier
        return product

    def __iter__(self):
        """Iterate over non-zero elements of InteractionTensor."""
        # Constant.
        if self.constant:
            yield []

        # 1-body elements.
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                if self.one_body_tensor[p, q]:
                    yield [p, q]

        # 2-body elements.
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        if self.two_body_tensor[p, q, r, s]:
                            yield [p, q, r, s]

    def __str__(self):
        """Print out the non-zero elements of InteractionTensor."""
        string = ''
        for key in self:
            if len(key) == 0:
                string += '[] {}\n'.format(self[key])
            elif len(key) == 2:
                string += '[{} {}] {}\n'.format(key[0], key[1], self[key])
            elif len(key) == 4:
                string += '[{} {} {} {}] {}\n'.format(key[0], key[1], key[2],
                                                      key[3], self[key])
        return string if string else '0'

    def rotate_basis(self, rotation_matrix):
        """
        Rotate the orbital basis of the InteractionTensor.

        Args:
            rotation_matrix: A square numpy array or matrix having
                dimensions of n_qubits by n_qubits. Assumed to be real and
                invertible.
        """
        self.one_body_tensor = one_body_basis_change(
            self.one_body_tensor, rotation_matrix)
        self.two_body_tensor = two_body_basis_change(
            self.two_body_tensor, rotation_matrix)

    def __repr__(self):
        return str(self)
