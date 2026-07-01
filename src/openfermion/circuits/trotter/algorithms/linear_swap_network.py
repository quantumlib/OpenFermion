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
"""A Trotter algorithm using the "fermionic simulation gate"."""

from typing import cast, Iterator, Optional, Sequence, Tuple

import cirq

from openfermion.circuits.gates import Ryxxy, Rxxyy, CRyxxy, CRxxyy, rot111, rot11
from openfermion.ops import DiagonalCoulombHamiltonian
from openfermion.circuits.primitives.swap_network import swap_network
from openfermion.circuits.trotter.trotter_algorithm import (
    Hamiltonian,
    TrotterStep,
    TrotterAlgorithm,
)


class LinearSwapNetworkTrotterAlgorithm(TrotterAlgorithm):
    """A Trotter algorithm using the "fermionic simulation gate".

    This algorithm simulates a DiagonalCoulombHamiltonian. It uses layers of
    fermionic swap networks to simultaneously simulate the one- and two-body
    interactions.

    This algorithm is described in arXiv:1711.04789.
    """

    supported_types = {DiagonalCoulombHamiltonian}

    def symmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, hamiltonian)
        return SymmetricLinearSwapNetworkTrotterStep(dc_hamiltonian)

    def asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, hamiltonian)
        return AsymmetricLinearSwapNetworkTrotterStep(dc_hamiltonian)

    def controlled_symmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, hamiltonian)
        return ControlledSymmetricLinearSwapNetworkTrotterStep(dc_hamiltonian)

    def controlled_asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, hamiltonian)
        return ControlledAsymmetricLinearSwapNetworkTrotterStep(dc_hamiltonian)


LINEAR_SWAP_NETWORK = LinearSwapNetworkTrotterAlgorithm()


class LinearSwapNetworkTrotterStep(TrotterStep):
    def __init__(self, hamiltonian: DiagonalCoulombHamiltonian) -> None:
        super().__init__(hamiltonian)


class SymmetricLinearSwapNetworkTrotterStep(LinearSwapNetworkTrotterStep):
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> Iterator[cirq.OP_TREE]:
        n_qubits = len(qubits)
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, self.hamiltonian)

        # Apply one- and two-body interactions for half of the full time
        def one_and_two_body_interaction(p, q, a, b) -> Iterator[cirq.OP_TREE]:
            yield Rxxyy(0.5 * dc_hamiltonian.one_body[p, q].real * time).on(a, b)
            yield Ryxxy(0.5 * dc_hamiltonian.one_body[p, q].imag * time).on(a, b)
            yield rot11(rads=float(-dc_hamiltonian.two_body[p, q] * time)).on(a, b)

        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (
            cirq.rz(rads=-dc_hamiltonian.one_body[i, i].real * time).on(qubits[i])
            for i in range(n_qubits)
        )

        # Apply one- and two-body interactions for half of the full time
        # This time, reorder the operations so that the entire Trotter step is
        # symmetric
        def one_and_two_body_interaction_reverse_order(p, q, a, b) -> Iterator[cirq.OP_TREE]:
            yield rot11(rads=float(-dc_hamiltonian.two_body[p, q] * time)).on(a, b)
            yield Ryxxy(0.5 * dc_hamiltonian.one_body[p, q].imag * time).on(a, b)
            yield Rxxyy(0.5 * dc_hamiltonian.one_body[p, q].real * time).on(a, b)

        yield swap_network(
            qubits, one_and_two_body_interaction_reverse_order, fermionic=True, offset=True
        )


class ControlledSymmetricLinearSwapNetworkTrotterStep(LinearSwapNetworkTrotterStep):
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> Iterator[cirq.OP_TREE]:
        n_qubits = len(qubits)
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, self.hamiltonian)

        if not isinstance(control_qubit, cirq.Qid):
            raise TypeError('Control qudit must be specified.')

        # Apply one- and two-body interactions for half of the full time
        def one_and_two_body_interaction(p, q, a, b) -> Iterator[cirq.OP_TREE]:
            yield CRxxyy(0.5 * dc_hamiltonian.one_body[p, q].real * time).on(
                cast(cirq.Qid, control_qubit), a, b
            )
            yield CRyxxy(0.5 * dc_hamiltonian.one_body[p, q].imag * time).on(
                cast(cirq.Qid, control_qubit), a, b
            )
            yield rot111(rads=float(-dc_hamiltonian.two_body[p, q] * time)).on(
                cast(cirq.Qid, control_qubit), a, b
            )

        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (
            rot11(rads=-dc_hamiltonian.one_body[i, i].real * time).on(control_qubit, qubits[i])
            for i in range(n_qubits)
        )

        # Apply one- and two-body interactions for half of the full time
        # This time, reorder the operations so that the entire Trotter step is
        # symmetric
        def one_and_two_body_interaction_reverse_order(p, q, a, b) -> Iterator[cirq.OP_TREE]:
            yield rot111(rads=float(-dc_hamiltonian.two_body[p, q] * time)).on(
                cast(cirq.Qid, control_qubit), a, b
            )
            yield CRyxxy(0.5 * dc_hamiltonian.one_body[p, q].imag * time).on(
                cast(cirq.Qid, control_qubit), a, b
            )
            yield CRxxyy(0.5 * dc_hamiltonian.one_body[p, q].real * time).on(
                cast(cirq.Qid, control_qubit), a, b
            )

        yield swap_network(
            qubits, one_and_two_body_interaction_reverse_order, fermionic=True, offset=True
        )

        # Apply phase from constant term
        yield cirq.rz(rads=-dc_hamiltonian.constant * time).on(control_qubit)


class AsymmetricLinearSwapNetworkTrotterStep(LinearSwapNetworkTrotterStep):
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> Iterator[cirq.OP_TREE]:
        n_qubits = len(qubits)
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, self.hamiltonian)

        # Apply one- and two-body interactions for the full time
        def one_and_two_body_interaction(p, q, a, b) -> Iterator[cirq.OP_TREE]:
            yield Rxxyy(dc_hamiltonian.one_body[p, q].real * time).on(a, b)
            yield Ryxxy(dc_hamiltonian.one_body[p, q].imag * time).on(a, b)
            yield rot11(rads=float(-2 * dc_hamiltonian.two_body[p, q] * time)).on(a, b)

        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (
            cirq.rz(rads=-dc_hamiltonian.one_body[i, i].real * time).on(qubits[i])
            for i in range(n_qubits)
        )

    def step_qubit_permutation(
        self, qubits: Sequence[cirq.Qid], control_qubit: Optional[cirq.Qid] = None
    ) -> Tuple[Sequence[cirq.Qid], Optional[cirq.Qid]]:
        # A Trotter step reverses the qubit ordering
        return qubits[::-1], None

    def finish(
        self,
        qubits: Sequence[cirq.Qid],
        n_steps: int,
        control_qubit: Optional[cirq.Qid] = None,
        omit_final_swaps: bool = False,
    ) -> Iterator[cirq.OP_TREE]:
        # If the number of Trotter steps is odd, possibly swap qubits back
        if n_steps & 1 and not omit_final_swaps:
            yield swap_network(qubits, fermionic=True)


class ControlledAsymmetricLinearSwapNetworkTrotterStep(LinearSwapNetworkTrotterStep):
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> Iterator[cirq.OP_TREE]:
        n_qubits = len(qubits)
        dc_hamiltonian = cast(DiagonalCoulombHamiltonian, self.hamiltonian)

        if not isinstance(control_qubit, cirq.Qid):
            raise TypeError('Control qudit must be specified.')

        # Apply one- and two-body interactions for the full time
        def one_and_two_body_interaction(p, q, a, b) -> Iterator[cirq.OP_TREE]:
            yield CRxxyy(dc_hamiltonian.one_body[p, q].real * time).on(
                cast(cirq.Qid, control_qubit), a, b
            )
            yield CRyxxy(dc_hamiltonian.one_body[p, q].imag * time).on(
                cast(cirq.Qid, control_qubit), a, b
            )
            yield rot111(rads=float(-2 * dc_hamiltonian.two_body[p, q] * time)).on(
                cast(cirq.Qid, control_qubit), a, b
            )

        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (
            rot11(rads=-dc_hamiltonian.one_body[i, i].real * time).on(control_qubit, qubits[i])
            for i in range(n_qubits)
        )

        # Apply phase from constant term
        yield cirq.rz(rads=-dc_hamiltonian.constant * time).on(control_qubit)

    def step_qubit_permutation(
        self, qubits: Sequence[cirq.Qid], control_qubit: Optional[cirq.Qid] = None
    ) -> Tuple[Sequence[cirq.Qid], Optional[cirq.Qid]]:
        # A Trotter step reverses the qubit ordering
        return qubits[::-1], control_qubit

    def finish(
        self,
        qubits: Sequence[cirq.Qid],
        n_steps: int,
        control_qubit: Optional[cirq.Qid] = None,
        omit_final_swaps: bool = False,
    ) -> Iterator[cirq.OP_TREE]:
        # If the number of Trotter steps is odd, possibly swap qubits back
        if n_steps & 1 and not omit_final_swaps:
            yield swap_network(qubits, fermionic=True)
