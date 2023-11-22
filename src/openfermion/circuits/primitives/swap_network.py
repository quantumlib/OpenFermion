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
"""The linear swap network."""

from typing import Callable, cast, Iterable, List, Sequence

import cirq

from openfermion.circuits.gates import FSWAP


def swap_network(
    qubits: Sequence[cirq.Qid],
    operation: Callable[
        [int, int, cirq.Qid, cirq.Qid], cirq.OP_TREE
    ] = lambda p, q, p_qubit, q_qubit: (),
    fermionic: bool = False,
    offset: bool = False,
) -> List[cirq.Operation]:
    """Apply operations to pairs of qubits or modes using a swap network.

    This is used for applying operations between arbitrary pairs of qubits or
    fermionic modes using only nearest-neighbor interactions on a linear array
    of qubits. It works by reversing the order of qubits or modes with a
    sequence of swap gates and applying an operation when the relevant qubits
    or modes become adjacent. For fermionic modes, this assumes the
    Jordan-Wigner Transform.

    Examples
    --------

    Input:

    .. testcode::

        import cirq
        from openfermion import swap_network

        qubits = cirq.LineQubit.range(4)
        circuit = cirq.Circuit(swap_network(qubits))
        print(circuit)

    Output:

    .. testoutput::

        0: ───×───────×───────
              │       │
        1: ───×───×───×───×───
                  │       │
        2: ───×───×───×───×───
              │       │
        3: ───×───────×───────

    Input:

    .. testcode::

        circuit = cirq.Circuit(swap_network(qubits, offset=True))
        print(circuit)

    Output:

    .. testoutput::

        0: ───────×───────×───
                  │       │
        1: ───×───×───×───×───
              │       │
        2: ───×───×───×───×───
                  │       │
        3: ───────×───────×───

    Input:

    .. testcode::

        circuit = cirq.Circuit(
            swap_network(
                qubits,
                lambda p, q, a, b: cirq.ISWAP(a, b)**-1 if abs(p - q) == 1
                                   else cirq.CZ(a, b),
                fermionic=True))
        print(circuit)

    Output:

    .. testoutput::

        0: ───iSwap──────×ᶠ────────────@───×ᶠ───────────────────
              │          │             │   │
        1: ───iSwap^-1───×ᶠ───@───×ᶠ───@───×ᶠ───iSwap──────×ᶠ───
                              │   │             │          │
        2: ───iSwap──────×ᶠ───@───×ᶠ───@───×ᶠ───iSwap^-1───×ᶠ───
              │          │             │   │
        3: ───iSwap^-1───×ᶠ────────────@───×ᶠ───────────────────


    Args:
        qubits: The qubits sorted so that the j-th qubit in the Sequence
            represents the j-th qubit or fermionic mode.
        operation: Returns extra interactions to perform between qubits/modes as
            they are swapped past each other. A call to this function takes the
            form ``operation(p, q, p_qubit, q_qubit)`` where p and q are indices
            representing either qubits or fermionic modes, and p_qubit and
            q_qubit are the qubits which are currently storing those modes.
        fermionic: If True, use fermionic swaps under the JWT (that is, swap
            fermionic modes instead of qubits). If False, use normal qubit
            swaps.
        offset: If True, then qubit 0 will participate in odd-numbered layers
            instead of even-numbered layers.
    """
    n_qubits = len(qubits)
    order = list(range(n_qubits))
    swap_gate = FSWAP if fermionic else cirq.SWAP
    result = []  # type: List[cirq.Operation]

    for layer_num in range(n_qubits):
        lowest_active_qubit = (layer_num + offset) % 2
        active_pairs = ((i, i + 1) for i in range(lowest_active_qubit, n_qubits - 1, 2))
        for i, j in active_pairs:
            p, q = order[i], order[j]
            extra_ops = operation(p, q, qubits[i], qubits[j])
            result.extend(cast(Iterable[cirq.Operation], cirq.flatten_op_tree(extra_ops)))
            result.append(swap_gate(qubits[i], qubits[j]))
            order[i], order[j] = q, p

    return result
