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

"""Base class for representating operators that are polynomials in the
fermionic ladder operators."""
from __future__ import absolute_import

import copy
import itertools
import numpy

from openfermion.config import *


class PolynomialTensorError(Exception):
    pass


def one_body_basis_change(one_body_tensor, rotation_matrix):
    """Change the basis of an 1-body interaction tensor such as the 1-RDM.

    M' = R^T.M.R where R is the rotation matrix, M is the 1-body tensor
    and M' is the transformed 1-body tensor.

    Args:
        one_body_tensor: A square numpy array or matrix containing information
            about a 1-body interaction tensor such as the 1-RDM.
        rotation_matrix: A square numpy array or matrix having dimensions of
            n_sites by n_sites. Assumed to be real and invertible.

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
            n_sites by n_sites. Assumed to be real and invertible.

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


class PolynomialTensor(object):
    """Class for storing tensor representations of operators that correspond
    with multilinear polynomials in the fermionic ladder operators.
    For instance, in a quadratic Hamiltonian (degree 2 polynomial) which
    conserves particle number, there are only terms of the form
    a^\dagger_p a_q, and the coefficients can be stored in an
    n_sites x n_sites matrix. Higher order terms would be described with
    tensors of higher dimension. Note that each tensor must have an even
    number of dimensions, since parity is conserved.
    Much of the functionality of this class is redudant with FermionOperator
    but enables much more efficient numerical computations in many cases,
    such as basis rotations.

    Attributes:
        n_sites(int): The number of sites on which the tensor acts.
        constant(complex or float): A constant term in the operator given
        as a complex number. For instance, the nuclear repulsion energy.
        n_body_tensors(dict): A dictionary storing the tensors describing
            n-body interactions. The keys are tuples that indicate the
            type of tensor. For instance, n_body_tensors[(1, 0)] would
            be an (n_sites x n_sites x n_sites x n_sites) numpy array,
            and it could represent the coefficients of terms of the form
            a^\dagger_i a_j, whereas n_body_tensors[(0, 1)] would be
            an array of the same shape, but instead representing terms
            of the form a_i a^\dagger_j.
    """

    def __init__(self, constant, n_body_tensors):
        """Initialize the PolynomialTensor class.

        Args:
            n_body_tensors(dict): A dictionary storing the tensors describing
                n-body interactions.
        """
        if constant is None:
            constant = 0.
        self.constant = constant
        self.n_body_tensors = n_body_tensors
        self.n_sites = list(n_body_tensors.values())[0].shape[0]

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: Tuples indicating which coefficient to get. For instance,
                    my_tensor[(6, 1), (8, 1), (2, 0)]
                returns
                    my_tensor.n_body_tensors[1, 1, 0][6, 8, 2]
        """
        if len(args) == 0:
            return self.constant
        else:
            index = tuple([operator[0] for operator in args])
            key = tuple([operator[1] for operator in args])
            return self.n_body_tensors[key][index]

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            Ints giving indices of tensor.

        Raises:
            ValueError: args must be of even length.
        """
        if len(args) == 0:
            self.constant = value
        else:
            key = tuple([operator[1] for operator in args])
            index = tuple([operator[0] for operator in args])
            self.n_body_tensors[key][index] = value

    def __eq__(self, other_operator):
        if self.n_body_tensors.keys() != other_operator.n_body_tensors.keys():
            return False
        diff = abs(other_operator.constant - self.constant)
        for key in self.n_body_tensors:
            self_tensor = self.n_body_tensors[key]
            other_tensor = other_operator.n_body_tensors[key]
            discrepancy = numpy.amax(
                              numpy.absolute(self_tensor - other_tensor))
            diff = max(diff, discrepancy)
        return diff < EQ_TOLERANCE

    def __neq__(self, other_operator):
        return not (self == other_operator)

    def __iadd__(self, addend):
        if not issubclass(type(addend), PolynomialTensor):
            raise TypeError('Invalid type.')

        if self.n_sites != addend.n_sites:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != addend.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

        self.constant += addend.constant
        for key in self.n_body_tensors:
            self.n_body_tensors[key] = numpy.add(self.n_body_tensors[key],
                                                 addend.n_body_tensors[key])
        return self

    def __add__(self, addend):
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __neg__(self):
        neg_n_body_tensors = dict()
        for key in self.n_body_tensors:
            neg_n_body_tensors[key] = numpy.negative(self.n_body_tensors[key])
        return PolynomialTensor(-self.constant,
                                n_body_tensors=neg_n_body_tensors)

    def __isub__(self, subtrahend):
        if not issubclass(type(subtrahend), PolynomialTensor):
            raise TypeError('Invalid type.')

        if self.n_sites != subtrahend.n_sites:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != subtrahend.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

        self.constant -= subtrahend.constant
        for key in self.n_body_tensors:
            self.n_body_tensors[key] = numpy.subtract(
                    self.n_body_tensors[key], subtrahend.n_body_tensors[key])
        return self

    def __sub__(self, subtrahend):
        r = copy.deepcopy(self)
        r -= subtrahend
        return r

    def __imul__(self, multiplier):
        if not issubclass(type(multiplier), PolynomialTensor):
            raise TypeError('Invalid type.')

        if self.n_sites != multiplier.n_sites:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != multiplier.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

        self.constant *= multiplier.constant
        for key in self.n_body_tensors:
            self.n_body_tensors[key] = numpy.multiply(
                    self.n_body_tensors[key], multiplier.n_body_tensors[key])
        return self

    def __mul__(self, multiplier):
        product = copy.deepcopy(self)
        product *= multiplier
        return product

    def __iter__(self):
        """Iterate over non-zero elements of PolynomialTensor."""
        # Constant.
        if self.constant:
            yield ()

        # n-body elements
        for key, n_body_tensor in self.n_body_tensors.items():
            for index in itertools.product(
                    range(self.n_sites), repeat=len(key)):
                if n_body_tensor[index]:
                    yield tuple(zip(index, key))

    def __str__(self):
        """Print out the non-zero elements of PolynomialTensor."""
        string = ''
        for key in self:
            if len(key) == 0:
                string += '() {}\n'.format(self[key])
            else:
                string += '{} {}\n'.format(key, self[key])
        return string if string else '0'

    def rotate_basis(self, rotation_matrix):
        """
        Rotate the orbital basis of the PolynomialTensor.

        Args:
            rotation_matrix: A square numpy array or matrix having
                dimensions of n_sites by n_sites. Assumed to be real and
                invertible.
        """
        self.n_body_tensors[1, 0] = one_body_basis_change(
            self.n_body_tensors[1, 0], rotation_matrix)
        self.n_body_tensors[1, 1, 0, 0] = two_body_basis_change(
            self.n_body_tensors[1, 1, 0, 0], rotation_matrix)

    def __repr__(self):
        return str(self)
