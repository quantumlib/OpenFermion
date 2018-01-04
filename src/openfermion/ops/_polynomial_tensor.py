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


def general_basis_change(general_tensor, rotation_matrix, key):
    """Change the basis of an general interaction tensor.

    M'^{p_1p_2...p_n} = R^{p_1}_{a_1} R^{p_2}_{a_2} ...
                        R^{p_n}_{a_n} M^{a_1a_2...a_n} R^{p_n}_{a_n}^T ...
                        R^{p_2}_{a_2}^T R_{p_1}_{a_1}^T

    where R is the rotation matrix, M is the general tensor, M' is the
    transformed general tensor, and a_k and p_k are indices. The formula uses
    the Einstein notation (implicit sum over repeated indices).

    In case R is complex, the k-th R in the above formula need to be conjugated
    if key has a 1 in the k-th place (meaning that the corresponding operator
    is a creation operator).

    Args:
        general_tensor: A square numpy array or matrix containing information
            about a general interaction tensor.
        rotation_matrix: A square numpy array or matrix having dimensions of
            n_qubits by n_qubits. Assumed to be unitary.
        key: A tuple indicating the type of general_tensor. Assumed to be
            non-empty. For example, a tensor storing coefficients of
            :math:`a^\dagger_p a_q` would have a key of (1, 0) whereas a tensor
            storing coefficients of :math:`a^\dagger_p a_q a_r a^\dagger_s`
            would have a key of (1, 0, 0, 1).

    Returns:
        transformed_general_tensor: general_tensor in the rotated basis.
    """
    # If operator acts on spin degrees of freedom, enlarge rotation matrix.
    n_orbitals = rotation_matrix.shape[0]
    if general_tensor.shape[0] == 2 * n_orbitals:
        rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))

    order = len(key)
    if order > 26:
        raise ValueError('Order exceeds maximum order supported (26).')

    # Do the basis change through a single call of numpy.einsum. For example,
    # for the (1, 1, 0, 0) tensor, the call is:
    #     numpy.einsum('abcd,aA,bB,cC,dD',
    #                  general_tensor,
    #                  rotation_matrix.conj(),
    #                  rotation_matrix.conj(),
    #                  rotation_matrix,
    #                  rotation_matrix)

    # The 'abcd' part of the subscripts
    subscripts_first = ''.join(chr(ord('a') + i) for i in range(order))

    # The 'Aa,Bb,Cc,Dd' part of the subscripts
    subscripts_rest = ','.join(chr(ord('a') + i) +
                               chr(ord('A') + i) for i in range(order))

    subscripts = subscripts_first + ',' + subscripts_rest

    # The list of rotation matrices, conjugated as necessary.
    rotation_matrices = [rotation_matrix.conj() if x else
                         rotation_matrix for x in key]

    # "optimize = True" does greedy optimization, which will be enough here.
    transformed_general_tensor = numpy.einsum(subscripts,
                                              general_tensor,
                                              *rotation_matrices,
                                              optimize=True)
    return transformed_general_tensor


class PolynomialTensor(object):
    """Class for storing tensor representations of operators that correspond
    with multilinear polynomials in the fermionic ladder operators.
    For instance, in a quadratic Hamiltonian (degree 2 polynomial) which
    conserves particle number, there are only terms of the form
    a^\dagger_p a_q, and the coefficients can be stored in an
    n_qubits x n_qubits matrix. Higher order terms would be described with
    tensors of higher dimension. Note that each tensor must have an even
    number of dimensions, since parity is conserved.
    Much of the functionality of this class is redudant with FermionOperator
    but enables much more efficient numerical computations in many cases,
    such as basis rotations.

    Attributes:
        n_qubits(int): The number of sites on which the tensor acts.
        n_body_tensors(dict): A dictionary storing the tensors describing
            n-body interactions. The keys are tuples that indicate the
            type of tensor. For instance, n_body_tensors[(1, 0)] would
            be an (n_qubits x n_qubits x n_qubits x n_qubits) numpy array,
            and it could represent the coefficients of terms of the form
            a^\dagger_i a_j, whereas n_body_tensors[(0, 1)] would be
            an array of the same shape, but instead representing terms
            of the form a_i a^\dagger_j.
    """

    def __init__(self, n_body_tensors):
        """Initialize the PolynomialTensor class.

        Args:
            n_body_tensors(dict): A dictionary storing the tensors describing
                n-body interactions.
        """
        self.n_body_tensors = n_body_tensors

        # Set n_qubits
        key_iterator = iter(n_body_tensors.keys())
        key = next(key_iterator)
        if key == ():
            key = next(key_iterator)
        self.n_qubits = n_body_tensors[key].shape[0]

    @property
    def constant(self):
        """Get the value of the constant term."""
        return self.n_body_tensors[()]

    @constant.setter
    def constant(self, value):
        """Set the value of the constant term."""
        self.n_body_tensors[()] = value

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: Tuples indicating which coefficient to get. For instance,
                `my_tensor[(6, 1), (8, 1), (2, 0)]`
                returns
                `my_tensor.n_body_tensors[1, 1, 0][6, 8, 2]`
        """
        if len(args) == 0:
            return self.n_body_tensors[()]
        else:
            index = tuple([operator[0] for operator in args])
            key = tuple([operator[1] for operator in args])
            return self.n_body_tensors[key][index]

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            args: Tuples indicating which coefficient to set.
        """
        if len(args) == 0:
            self.n_body_tensors[()] = value
        else:
            key = tuple([operator[1] for operator in args])
            index = tuple([operator[0] for operator in args])
            self.n_body_tensors[key][index] = value

    def __eq__(self, other_operator):
        if self.n_qubits != other_operator.n_qubits:
            return False
        if self.n_body_tensors.keys() != other_operator.n_body_tensors.keys():
            return False
        diff = 0.
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

        if self.n_qubits != addend.n_qubits:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != addend.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

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
        return PolynomialTensor(neg_n_body_tensors)

    def __isub__(self, subtrahend):
        if not issubclass(type(subtrahend), PolynomialTensor):
            raise TypeError('Invalid type.')

        if self.n_qubits != subtrahend.n_qubits:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != subtrahend.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

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

        if self.n_qubits != multiplier.n_qubits:
            raise TypeError('Invalid tensor shape.')

        if self.n_body_tensors.keys() != multiplier.n_body_tensors.keys():
            raise TypeError('Invalid tensor type.')

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
        def sort_key(key):
            """This determines how the keys to n_body_tensors
            should be sorted."""
            # Interpret key as an integer written in binary
            if key == ():
                return 0, 0
            else:
                key_int = int(''.join(map(str, key)))
                return len(key), key_int

        for key in sorted(self.n_body_tensors.keys(), key=sort_key):
            if key == ():
                yield ()
            else:
                n_body_tensor = self.n_body_tensors[key]
                for index in itertools.product(
                        range(self.n_qubits), repeat=len(key)):
                    if n_body_tensor[index]:
                        yield tuple(zip(index, key))

    def __str__(self):
        """Print out the non-zero elements of PolynomialTensor."""
        strings = []
        for key in self:
            strings.append('{} {}\n'.format(key, self[key]))
        return ''.join(strings) if strings else '0'

    def rotate_basis(self, rotation_matrix):
        """
        Rotate the orbital basis of the PolynomialTensor.

        Args:
            rotation_matrix: A square numpy array or matrix having
                dimensions of n_qubits by n_qubits. Assumed to be real and
                invertible.
        """
        for key in self.n_body_tensors:
            if key == ():
                pass
            else:
                self.n_body_tensors[key] = general_basis_change(
                    self.n_body_tensors[key], rotation_matrix, key)

    def __repr__(self):
        return str(self)
