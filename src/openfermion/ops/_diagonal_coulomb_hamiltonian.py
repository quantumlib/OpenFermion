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
"""Class for electronic structure Hamiltonians with a diagonal Coulomb term"""

import copy

import numpy


class DiagonalCoulombHamiltonian:
    r"""Class for storing Hamiltonians of the form

    .. math::

        \sum_{p, q} T_{pq} a^\dagger_p a_q +
        \sum_{p, q} V_{pq} a^\dagger_p a_p a^\dagger_q a_q +
        \text{constant}

    where

        - :math:`T` is a Hermitian matrix.
        - :math:`V` is a real symmetric matrix.

    Attributes:
        one_body(ndarray): The Hermitian matrix :math:`T`.
        two_body(ndarray): The real symmetric matrix :math:`V`.
        constant(float): The constant.
    """
    def __init__(self, one_body, two_body, constant=0.):
        if two_body.dtype != numpy.float:
            raise ValueError('Two-body tensor has invalid dtype. Expected {} '
                             'but was {}'.format(numpy.float, two_body.dtype))
        if not numpy.allclose(two_body, two_body.T):
            raise ValueError('Two-body tensor must be symmetric.')
        if not numpy.allclose(one_body, one_body.T.conj()):
            raise ValueError('One-body tensor must be Hermitian.')

        # Move the diagonal of two_body to one_body
        diag_indices = numpy.diag_indices(one_body.shape[0])
        one_body[diag_indices] += two_body[diag_indices]
        numpy.fill_diagonal(two_body, 0.)

        self.one_body = one_body
        self.two_body = two_body
        self.constant = constant

    def __imul__(self, multiplier):
        self.one_body *= multiplier
        self.two_body *= multiplier
        self.constant *= multiplier
        return self

    def __mul__(self, multiplier):
        product = copy.deepcopy(self)
        product *= multiplier
        return product

    def __rmul__(self, multiplier):
        product = copy.deepcopy(self)
        product *= multiplier
        return product

    def __itruediv__(self, dividend):
        self.one_body /= dividend
        self.two_body /= dividend
        self.constant /= dividend
        return self

    def __truediv__(self, dividend):
        quotient = copy.deepcopy(self)
        quotient /= dividend
        return quotient

    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)
