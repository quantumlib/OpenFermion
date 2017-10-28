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

"""Class and functions to store and manipulate Hamiltonians that are quadratic
in the fermionic ladder operators."""
from __future__ import absolute_import

import numpy

from openfermion.config import EQ_TOLERANCE
from openfermion.ops import PolynomialTensor


class QuadraticHamiltonianError(Exception):
    pass


class QuadraticHamiltonian(PolynomialTensor):
    """Class for storing Hamiltonians that are quadratic in the fermionic
    ladder operators. The operators stored in this class take the form::

        \sum_{p, q} (M_{pq} - \mu \delta_{pq}) a^\dagger_p a_q
        + 1 / 2 \sum_{p, q} (A_{pq} a^\dagger_p a^\dagger_q + h.c.)
        + constant

    where
        M is a Hermitian n_qubits x n_qubits matrix.
        A is an antisymmetric n_qubits x n_qubits matrix.
        \mu is a float representing the chemical potential term.
        \delta_{pq} is the Kronecker delta symbol.

    We separate the chemical potential \mu from M so that we can use it
    to adjust the expectation value of the total number of particles.

    Attributes:
        n_qubits: An int giving the number of qubits.
        constant: A constant term in the operator given as a float.
    """

    def __init__(self, constant, hermitian_part,
                 antisymmetric_part=None, chemical_potential=None):
        """
        Initialize the QuadraticHamiltonian class.

        Args:
            constant(float): A constant term in the operator.
            hermitian_part(ndarray): The matrix M, which represents the
                coefficients of the particle-number-conserving terms.
                This is an n_qubits x n_qubits numpy array of complex numbers.
            antisymmetric_part(ndarray): The matrix A, which represents the
                coefficients of the non-particle-number-conserving terms.
                This is an n_qubits x n_qubits numpy array of complex numbers.
            chemical_potential(float): The chemical potential \mu.
        """
        n_qubits = hermitian_part.shape[0]

        # Initialize combined Hermitian part
        if not chemical_potential:
            combined_hermitian_part = hermitian_part
        else:
            combined_hermitian_part = (
                    hermitian_part -
                    chemical_potential * numpy.eye(n_qubits))

        # Initialize antisymmetric part
        if antisymmetric_part is None:
            antisymmetric_part = numpy.zeros((n_qubits, n_qubits), complex)

        super(QuadraticHamiltonian, self).__init__(
                constant,
                {(1, 0): combined_hermitian_part,
                 (1, 1): .5 * antisymmetric_part,
                 (0, 0): -.5 * antisymmetric_part.conj()})
        self.chemical_potential = chemical_potential

    def combined_hermitian_part(self):
        """Return the Hermitian part including the chemical potential."""
        return self.n_body_tensors[1, 0]

    def hermitian_part(self):
        """Return the Hermitian part not including the chemical potential."""
        hermitian_part = self.combined_hermitian_part()
        if self.chemical_potential:
            hermitian_part += (self.chemical_potential *
                               numpy.eye(self.n_qubits))
        return hermitian_part

    def antisymmetric_part(self):
        """Return the antisymmetric part."""
        return 2. * self.n_body_tensors[1, 1]

    def conserves_particle_number(self):
        """Return whether this Hamiltonian conserves particle number."""
        discrepancy = numpy.max(numpy.abs(self.antisymmetric_part()))
        return discrepancy < EQ_TOLERANCE
