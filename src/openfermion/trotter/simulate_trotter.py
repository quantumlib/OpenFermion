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

"""Perform Hamiltonian simulation via a Trotter-Suzuki product formula."""

from typing import Optional, Sequence

import cirq
from openfermion import DiagonalCoulombHamiltonian, InteractionOperator

from openfermioncirq.trotter.trotter_algorithm import (
        Hamiltonian,
        TrotterStep,
        TrotterAlgorithm)
from openfermioncirq.trotter.algorithms import (
        LINEAR_SWAP_NETWORK,
        LOW_RANK)


def simulate_trotter(qubits: Sequence[cirq.Qid],
                     hamiltonian: Hamiltonian,
                     time: float,
                     n_steps: int=1,
                     order: int=0,
                     algorithm: Optional[TrotterAlgorithm]=None,
                     control_qubit: Optional[cirq.Qid]=None,
                     omit_final_swaps: bool=False
                     ) -> cirq.OP_TREE:
    """Simulate Hamiltonian evolution using a Trotter-Suzuki product formula.

    The input is a Hamiltonian represented as an InteractionOperator or
    DiagonalCoulombHamiltonian. Not all types are supported by all algorithm
    options.

    The product formula used is from "General theory of fractal path integrals
    with applications to many-body theories and statistical physics" by
    Masuo Suzuki.

    Args:
        qubits: The qubits on which to apply operations. They should be sorted
            so that the j-th qubit in the Sequence holds the occupation of the
            j-th fermionic mode.
        hamiltonian: The Hamiltonian to simulate.
        time: The evolution time.
        n_steps: The number of Trotter steps to use. Default is 1.
        order: The order of the product formula. The value indexes symmetric
            formulae, e.g., a value of 2 indicates a second-order symmetric,
            sometimes known as a fourth-order, Trotter formula. A value of 0
            indicates an asymmetric Trotter formula. Default is 0.
        algorithm: The algorithm to use to simulate a single Trotter step.
            This is a constant exposed in the openfermioncirq.trotter module.
            If not specified, a default option will be chosen based on the
            type of the given Hamiltonian.
            Available options:
                LINEAR_SWAP_NETWORK: The algorithm from arXiv:1711.04789.
                    Requires the input to be a DiagonalCoulombHamiltonian.
                LOW_RANK: The "low rank" strategy.
                    Requires the input to be an InteractionOperator.
                SPLIT_OPERATOR: The algorithm from arXiv:1706.00023.
                    Requires the input to be a DiagonalCoulombHamiltonian.
        control_qubit: A qubit on which to control the Trotter step.
        omit_final_swaps: If this is set to True, then SWAP or FSWAP gates at
            the end of the circuit may be omitted. This option exists because
            certain Trotter step algorithms, such as those based on swap
            networks, induce a permutation on the qubits or on the ordering in
            which qubits represent fermionic modes. For instance, algorithms
            based on swap networks may reverse the qubits depending on the
            number of Trotter steps used and the order of the Trotter formula
            selected. Setting this option to True will sometimes result in a
            circuit with fewer gates, but with the ordering of qubits or modes
            reversed in the final wavefunction.
    """
    # TODO Document gate complexities of algorithm options
    if order < 0:
        raise ValueError('The order of the Trotter formula must be at least 0.')

    if algorithm is None:
        algorithm = _select_trotter_algorithm(hamiltonian)

    if not isinstance(hamiltonian, tuple(algorithm.supported_types)):
        raise TypeError(
                'The input Hamiltonian was a {} but the chosen Trotter step '
                'algorithm only supports Hamiltonians of type {}'.format(
                    type(hamiltonian).__name__,
                    {cls.__name__ for cls in algorithm.supported_types}))

    # Select the Trotter step to use
    trotter_step = _select_trotter_step(
            hamiltonian, order, algorithm,
            controlled = control_qubit is not None)

    # Get ready to perform Trotter steps
    yield trotter_step.prepare(qubits, control_qubit)

    # Perform Trotter steps
    step_time = time / n_steps
    for _ in range(n_steps):
        yield _perform_trotter_step(
                qubits, step_time, order, trotter_step, control_qubit)
        qubits, control_qubit = trotter_step.step_qubit_permutation(
                qubits, control_qubit)

    # Finish
    yield trotter_step.finish(qubits, n_steps, control_qubit, omit_final_swaps)


def _perform_trotter_step(qubits: Sequence[cirq.Qid],
                          time: float,
                          order: int,
                          trotter_step: TrotterStep,
                          control_qubit: Optional[cirq.Qid]
                          ) -> cirq.OP_TREE:
    """Perform a Trotter step."""
    if order <= 1:
        yield trotter_step.trotter_step(qubits, time, control_qubit)
    else:
        # Recursively split this Trotter step into five smaller steps.
        # The first two and last two steps use this amount of time
        split_time = time / (4 - 4**(1 / (2 * order - 1)))

        for _ in range(2):
            yield _perform_trotter_step(
                    qubits, split_time, order - 1, trotter_step, control_qubit)
            qubits, control_qubit = trotter_step.step_qubit_permutation(
                    qubits, control_qubit)

        yield _perform_trotter_step(
                qubits, time - 4 * split_time, order - 1,
                trotter_step, control_qubit)
        qubits, control_qubit = trotter_step.step_qubit_permutation(
                qubits, control_qubit)

        for _ in range(2):
            yield _perform_trotter_step(
                    qubits, split_time, order - 1, trotter_step, control_qubit)
            qubits, control_qubit = trotter_step.step_qubit_permutation(
                    qubits, control_qubit)


def _select_trotter_algorithm(hamiltonian: Hamiltonian) -> TrotterAlgorithm:
    if isinstance(hamiltonian, DiagonalCoulombHamiltonian):
        return LINEAR_SWAP_NETWORK
    elif isinstance(hamiltonian, InteractionOperator):
        return LOW_RANK
    else:
        raise TypeError('Failed to select a default Trotter algorithm '
                        'for Hamiltonian of type {}.'.format(
                            type(hamiltonian).__name__))


def _select_trotter_step(hamiltonian: Hamiltonian,
                         order: int,
                         algorithm: TrotterAlgorithm,
                         controlled: bool) -> TrotterStep:
    """Select a particular Trotter step from a Trotter step algorithm."""
    if controlled:
        if order == 0:
            trotter_step = algorithm.controlled_asymmetric(hamiltonian)
            if trotter_step is None:
                raise ValueError('The chosen Trotter step algorithm does not '
                                 'support the order 0 (asymmetric) formula '
                                 'with a control qubit.')
        else:
            trotter_step = algorithm.controlled_symmetric(hamiltonian)
            if trotter_step is None:
                raise ValueError('The chosen Trotter step algorithm does not '
                                 'support higher (> 0) order formulas '
                                 'with a control qubit.')
    else:
        if order == 0:
            trotter_step = algorithm.asymmetric(hamiltonian)
            if trotter_step is None:
                raise ValueError('The chosen Trotter step algorithm does not '
                                 'support the order 0 (asymmetric) formula.')
        else:
            trotter_step = algorithm.symmetric(hamiltonian)
            if trotter_step is None:
                raise ValueError('The chosen Trotter step algorithm does not '
                                 'support higher (> 0) order formulas.')
    return trotter_step
