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
import numpy
from scipy.linalg import qr

from openfermion.ops import InteractionOperator, QuadraticHamiltonian


def random_unitary_matrix(n, real=False):
    """Obtain a random n x n unitary matrix."""
    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    Q, R = qr(rand_mat)
    return Q


def random_hermitian_matrix(n, real=False):
    """Generate a random n x n Hermitian matrix."""
    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    hermitian_mat = rand_mat + rand_mat.T.conj()
    return hermitian_mat


def random_antisymmetric_matrix(n, real=False):
    """Generate a random n x n antisymmetric matrix."""
    if real:
        rand_mat = numpy.random.randn(n, n)
    else:
        rand_mat = numpy.random.randn(n, n) + 1.j * numpy.random.randn(n, n)
    antisymmetric_mat = rand_mat - rand_mat.T
    return antisymmetric_mat


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
    return QuadraticHamiltonian(constant, hermitian_mat,
                                antisymmetric_mat, chemical_potential)


def random_interaction_operator(n_qubits):
    """Generate a random instance of InteractionOperator."""

    # Initialize.
    constant = numpy.random.randn()
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits), float)
    two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                         n_qubits, n_qubits), float)

    # Randomly generate the one-body and two-body integrals.
    for p in range(n_qubits):
        for q in range(n_qubits):

            # One-body terms.
            if (p <= p) and (p % 2 == q % 2):
                one_body_coefficients[p, q] = numpy.random.randn()
                one_body_coefficients[q, p] = one_body_coefficients[p, q]

            # Keep looping.
            for r in range(n_qubits):
                for s in range(n_qubits):

                    # Skip zero terms.
                    if (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        unique_indices = len(set([p, q, r, s]))

                        # srqp srpq sprq spqr sqpr sqrp
                        # rsqp rspq rpsq rpqs rqps rqsp.
                        if unique_indices == 4:
                            if min(r, s) <= min(p, q):
                                continue

                        # qqpp.
                        elif unique_indices == 2:
                            if q < p:
                                continue

                    # Add the two-body coefficients.
                    two_body_coefficients[p, q, r, s] = numpy.random.randn()
                    two_body_coefficients[s, r, q, p] = two_body_coefficients[
                        p, q, r, s]

    # Build the molecular operator and return.
    molecular_operator = InteractionOperator(
        constant, one_body_coefficients, two_body_coefficients)
    return molecular_operator
