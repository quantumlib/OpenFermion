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
import itertools
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
        n_qubits(int): The number of qubits on which the tensor
            acts.
        constant(float): A constant term in the operator given as a float.
            For instance, the nuclear repulsion energy.
        n_body_tensors(dict): A dictionary storing the tensors describing
            n-body interactions. For instance, n_body_tensors[2] is a
            n_qubits x n_qubits x n_qubits x n_qubits numpy array of floats.
    """

    def __init__(self, constant, *args, n_body_tensors=None, n_qubits=None):
        """Initialize the InteractionTensor class.

        Args:
            constant(float): A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
            n_body_tensors(dict): A dictionary storing the tensors describing
                n-body interactions. For instance, n_body_tensors[2] is a
                n_qubits x n_qubits x n_qubits x n_qubits numpy array of floats.
        """
        # initialize constant
        if constant is not None:
            self.constant = constant
        else:
            self.constant = 0.

        # initialize n_body_tensors
        if args:
            # tensors were passed in as arguments
            self.n_body_tensors = dict()
            for i in range(len(args)):
                self.n_body_tensors[i + 1] = args[i]
        elif n_body_tensors:
            # tensors were passed in directly as dictionary
            self.n_body_tensors = n_body_tensors
        else:
            # no tensors were given
            self.n_body_tensors = dict()

        # initialize n_qubits
        if n_qubits:
            self.n_qubits = n_qubits
        elif 1 in self.n_body_tensors:
            self.n_qubits = self.n_body_tensors[1].shape[0]
        else:
            raise ValueError("Could not determine n_qubits.")

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            Ints giving indices of tensor. Should have even length.

        Raises:
            ValueError: args must be of even length.
        """
        args = tuple(args)
        if not len(args):
            return self.constant
        elif len(args) % 2 == 0:
            return self.n_body_tensors[len(args) // 2][args]
        else:
            raise ValueError('args must be of even length.')

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            Ints giving indices of tensor.

        Raises:
            ValueError: args must be of even length.
        """
        args = tuple(args)
        if not len(args):
            self.constant = value
        elif len(args) % 2 == 0:
            self.n_body_tensors[len(args) // 2][args] = value
        else:
            raise ValueError('args must be of even length.')

    def __eq__(self, other_operator):
        if self.n_body_tensors.keys() != other_operator.n_body_tensors.keys():
            return False
        diff = abs(other_operator.constant - self.constant)
        for n in self.n_body_tensors.keys():
            self_tensor = self.n_body_tensors[n]
            other_tensor = other_operator.n_body_tensors[n]
            discrepancy = numpy.amax(
                              numpy.absolute(self_tensor - other_tensor))
            diff = max(diff, discrepancy)
        return diff < EQ_TOLERANCE

    def __neq__(self, other_operator):
        return not (self == other_operator)

    def __iadd__(self, addend):
        if not issubclass(type(addend), InteractionTensor):
            raise TypeError('Invalid type.')

        if self.n_qubits != addend.n_qubits:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != addend.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

        self.constant += addend.constant
        for n in self.n_body_tensors.keys():
            self.n_body_tensors[n] = numpy.add(self.n_body_tensors[n],
                                               addend.n_body_tensors[n])
        return self

    def __add__(self, addend):
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __neg__(self):
        neg_n_body_tensors = dict()
        for n in self.n_body_tensors.keys():
            neg_n_body_tensors[n] = numpy.negative(self.n_body_tensors[n])
        return InteractionTensor(-self.constant,
                                 n_body_tensors=neg_n_body_tensors)

    def __isub__(self, subtrahend):
        if not issubclass(type(subtrahend), InteractionTensor):
            raise TypeError('Invalid type.')

        if self.n_qubits != subtrahend.n_qubits:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != subtrahend.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

        self.constant -= subtrahend.constant
        for n in self.n_body_tensors.keys():
            self.n_body_tensors[n] = numpy.subtract(
                    self.n_body_tensors[n], subtrahend.n_body_tensors[n])
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

        if self.n_body_tensors.keys() != multiplier.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

        self.constant *= multiplier.constant
        for n in self.n_body_tensors.keys():
            self.n_body_tensors[n] = numpy.multiply(
                    self.n_body_tensors[n], multiplier.n_body_tensors[n])
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

        # n-body elements
        for n, n_body_tensor in self.n_body_tensors.items():
            for index in itertools.product(range(self.n_qubits), repeat=2 * n):
                if n_body_tensor[index]:
                    yield list(index)

    def __str__(self):
        """Print out the non-zero elements of InteractionTensor."""
        string = ''
        for key in self:
            if len(key) == 0:
                string += '[] {}\n'.format(self[key])
            else:
                string += '[{}] {}\n'.format(' '.join([str(i) for i in key]),
                                             self[key])
        return string if string else '0'

    def rotate_basis(self, rotation_matrix):
        """
        Rotate the orbital basis of the InteractionTensor.

        Args:
            rotation_matrix: A square numpy array or matrix having
                dimensions of n_qubits by n_qubits. Assumed to be real and
                invertible.
        """
        self.n_body_tensors[1] = one_body_basis_change(
            self.n_body_tensors[1], rotation_matrix)
        self.n_body_tensors[2] = two_body_basis_change(
            self.n_body_tensors[2], rotation_matrix)

    def __repr__(self):
        return str(self)
