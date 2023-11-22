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
"""A Trotter algorithm using the low rank decomposition strategy."""

from typing import cast, Optional, Sequence, TYPE_CHECKING, Tuple

import numpy

import cirq

import openfermion.ops as ops

# import openfermion.circuits.gates as gates
from openfermion.circuits.gates import rot11, rot111

# import openfermion.circuits.primitives as primitives
from openfermion.circuits.primitives import swap_network, bogoliubov_transform
from openfermion.circuits.low_rank import (
    low_rank_two_body_decomposition,
    prepare_one_body_squared_evolution,
)
from openfermion.circuits.trotter.trotter_algorithm import (
    Hamiltonian,
    TrotterStep,
    TrotterAlgorithm,
)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


class LowRankTrotterAlgorithm(TrotterAlgorithm):
    r"""A Trotter algorithm using the low rank decomposition strategy.

    This algorithm simulates an InteractionOperator with real coefficients.
    The one-body terms are simulated in their diagonal basis; the basis change
    is accomplished with a Bogoliubov transformation.
    To simulate the two-body terms, the two-body tensor is decomposed into
    singular components and possibly truncating. Then, each singular component
    is simulated in the appropriate basis using a (non-fermionic) swap network.
    The general idea is based on expressing the two-body operator as
    $\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s =
    \sum_{j=0}^{J-1} \lambda_j (\sum_{pq} g_{jpq} a^\dagger_p a_q)^2$
    One can then diagonalize the squared one-body component as
    math:`\sum_{pq} g_{pqj} a^\dagger_p a_q =
    R_j (\sum_{p} f_{pj} n_p) R_j^\dagger`
    Then, a 'low rank' Trotter step of the two-body tensor can be simulated as
    $\prod_{j=0}^{J-1}
    R_j e^{-i \lambda_j \sum_{pq} f_{pj} f_{qj} n_p n_q} R_j^\dagger$.
    The $R_j$ are Bogoliubov transformations, and one can
    use a swap network to simulate the diagonal $n_p n_q$ terms.
    The value of J is either fully the square of the number of qubits,
    which would imply no truncation, or it is specified by the user,
    or it is chosen so that
    $\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x$
    where x is a truncation threshold specified by user.
    """

    supported_types = {ops.InteractionOperator}

    def __init__(
        self,
        truncation_threshold: Optional[float] = 1e-8,
        final_rank: Optional[int] = None,
        spin_basis=True,
    ) -> None:
        """
        Args:
            truncation_threshold: The value of x from the docstring of
                this class.
            final_rank: If provided, this specifies the value of J at which to
                truncate.
            spin_basis: Whether the Hamiltonian is given in the spin orbital
                (rather than spatial orbital) basis.
        """
        self.truncation_threshold = truncation_threshold
        self.final_rank = final_rank
        self.spin_basis = spin_basis

    def asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return AsymmetricLowRankTrotterStep(
            hamiltonian, self.truncation_threshold, self.final_rank, self.spin_basis
        )

    def controlled_asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return ControlledAsymmetricLowRankTrotterStep(
            hamiltonian, self.truncation_threshold, self.final_rank, self.spin_basis
        )


LOW_RANK = LowRankTrotterAlgorithm()


class LowRankTrotterStep(TrotterStep):
    def __init__(
        self,
        hamiltonian: 'ops.InteractionOperator',
        truncation_threshold: Optional[float] = 1e-8,
        final_rank: Optional[int] = None,
        spin_basis=True,
    ) -> None:
        self.truncation_threshold = truncation_threshold
        self.final_rank = final_rank

        # Perform the low rank decomposition of two-body operator.
        (
            self.eigenvalues,
            self.one_body_squares,
            one_body_correction,
            _,
        ) = low_rank_two_body_decomposition(
            hamiltonian.two_body_tensor,
            truncation_threshold=self.truncation_threshold,
            final_rank=self.final_rank,
            spin_basis=spin_basis,
        )

        # Get scaled density-density terms and basis transformation matrices.
        self.scaled_density_density_matrices = []  # type: List[numpy.ndarray]
        self.basis_change_matrices = []  # type: List[numpy.ndarray]
        for j in range(len(self.eigenvalues)):
            density_density_matrix, basis_change_matrix = prepare_one_body_squared_evolution(
                self.one_body_squares[j]
            )
            self.scaled_density_density_matrices.append(
                numpy.real(self.eigenvalues[j] * density_density_matrix)
            )
            self.basis_change_matrices.append(basis_change_matrix)

        # Get transformation matrix and orbital energies for one-body terms
        one_body_coefficients = hamiltonian.one_body_tensor + one_body_correction
        quad_ham = ops.QuadraticHamiltonian(one_body_coefficients)
        (
            self.one_body_energies,
            self.one_body_basis_change_matrix,
            _,
        ) = quad_ham.diagonalizing_bogoliubov_transform()

        super().__init__(hamiltonian)


class AsymmetricLowRankTrotterStep(LowRankTrotterStep):
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> cirq.OP_TREE:
        n_qubits = len(qubits)

        # Change to the basis in which the one-body term is diagonal
        yield bogoliubov_transform(qubits, self.one_body_basis_change_matrix.T.conj())

        # Simulate the one-body terms.
        for p in range(n_qubits):
            yield cirq.rz(rads=-self.one_body_energies[p] * time).on(qubits[p])

        # Simulate each singular vector of the two-body terms.
        prior_basis_matrix = self.one_body_basis_change_matrix

        for j in range(len(self.eigenvalues)):
            # Get the two-body coefficients and basis change matrix.
            two_body_coefficients = self.scaled_density_density_matrices[j]
            basis_change_matrix = self.basis_change_matrices[j]

            # Merge previous basis change matrix with the inverse of the
            # current one
            merged_basis_change_matrix = numpy.dot(prior_basis_matrix, basis_change_matrix.T.conj())
            yield bogoliubov_transform(qubits, merged_basis_change_matrix)

            # Simulate the off-diagonal two-body terms.
            yield swap_network(
                qubits,
                lambda p, q, a, b: rot11(rads=-2 * two_body_coefficients[p, q] * time).on(a, b),
            )
            qubits = qubits[::-1]

            # Simulate the diagonal two-body terms.
            for p in range(n_qubits):
                yield cirq.rz(rads=-two_body_coefficients[p, p] * time).on(qubits[p])

            # Update prior basis change matrix
            prior_basis_matrix = basis_change_matrix

        # Undo final basis transformation
        yield bogoliubov_transform(qubits, prior_basis_matrix)

    def step_qubit_permutation(
        self, qubits: Sequence[cirq.Qid], control_qubit: Optional[cirq.Qid] = None
    ) -> Tuple[Sequence[cirq.Qid], Optional[cirq.Qid]]:
        # A Trotter step reverses the qubit ordering when the number of
        # eigenvalues is odd
        if len(self.eigenvalues) & 1:
            return qubits[::-1], None
        else:
            return qubits, None

    def finish(
        self,
        qubits: Sequence[cirq.Qid],
        n_steps: int,
        control_qubit: Optional[cirq.Qid] = None,
        omit_final_swaps: bool = False,
    ) -> cirq.OP_TREE:
        if not omit_final_swaps:
            # If the number of swap networks was odd, swap the qubits back
            if n_steps & 1 and len(self.eigenvalues) & 1:
                yield swap_network(qubits)


class ControlledAsymmetricLowRankTrotterStep(LowRankTrotterStep):
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> cirq.OP_TREE:
        if not isinstance(control_qubit, cirq.Qid):
            raise TypeError('Control qudit must be specified.')

        n_qubits = len(qubits)

        # Change to the basis in which the one-body term is diagonal
        yield bogoliubov_transform(qubits, self.one_body_basis_change_matrix.T.conj())

        # Simulate the one-body terms.
        for p in range(n_qubits):
            yield rot11(rads=-self.one_body_energies[p] * time).on(control_qubit, qubits[p])

        # Simulate each singular vector of the two-body terms.
        prior_basis_matrix = self.one_body_basis_change_matrix

        for j in range(len(self.eigenvalues)):
            # Get the two-body coefficients and basis change matrix.
            two_body_coefficients = self.scaled_density_density_matrices[j]
            basis_change_matrix = self.basis_change_matrices[j]

            # Merge previous basis change matrix with the inverse of the
            # current one
            merged_basis_change_matrix = numpy.dot(prior_basis_matrix, basis_change_matrix.T.conj())
            yield bogoliubov_transform(qubits, merged_basis_change_matrix)

            # Simulate the off-diagonal two-body terms.
            yield swap_network(
                qubits,
                lambda p, q, a, b: rot111(-2 * two_body_coefficients[p, q] * time).on(
                    cast(cirq.Qid, control_qubit), a, b
                ),
            )
            qubits = qubits[::-1]

            # Simulate the diagonal two-body terms.
            yield (
                rot11(rads=-two_body_coefficients[k, k] * time).on(control_qubit, qubits[k])
                for k in range(n_qubits)
            )

            # Update prior basis change matrix.
            prior_basis_matrix = basis_change_matrix

        # Undo final basis transformation.
        yield bogoliubov_transform(qubits, prior_basis_matrix)

        # Apply phase from constant term
        yield cirq.rz(rads=-self.hamiltonian.constant * time).on(control_qubit)

    def step_qubit_permutation(
        self, qubits: Sequence[cirq.Qid], control_qubit: Optional[cirq.Qid] = None
    ) -> Tuple[Sequence[cirq.Qid], Optional[cirq.Qid]]:
        # A Trotter step reverses the qubit ordering when the number of
        # eigenvalues is odd
        if len(self.eigenvalues) & 1:
            return qubits[::-1], control_qubit
        else:
            return qubits, control_qubit

    def finish(
        self,
        qubits: Sequence[cirq.Qid],
        n_steps: int,
        control_qubit: Optional[cirq.Qid] = None,
        omit_final_swaps: bool = False,
    ) -> cirq.OP_TREE:
        if not omit_final_swaps:
            # If the number of swap networks was odd, swap the qubits back
            if n_steps & 1 and len(self.eigenvalues) & 1:
                yield swap_network(qubits)
