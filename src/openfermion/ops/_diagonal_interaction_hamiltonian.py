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

class DiagonalInteractionHamiltonian:
    """Class for storing Hamiltonians of the form

    .. math::

        \sum_{p, q} T_{pq} a^\dagger_p a_q +
        \sum_{p, q} V_{pq} a^\dagger_p a_p a^\dagger_q a_q +
        \\text{constant}

    where
    
        - :math:`T` is a Hermitian matrix.
        - :math:`V` is a real symmetric matrix.

    Attributes:
        kinetic_term(ndarray): The Hermitian matrix :math:`T`.
        potential_term(ndarray): The real symmetric matrix :math:`V`.
        constant(float): The constant.
    """

    def __init__(self, kinetic_term, potential_term, constant=0.):
        self.kinetic_term = kinetic_term
        self.potential_term = potential_term
        self.constant = constant
