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
"""Classes for representing algorithms for performing Trotter steps."""

from typing import Optional, Sequence, TYPE_CHECKING, Tuple, Union

import abc

import cirq
from openfermion import ops

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Set, Type

Hamiltonian = Union[
    ops.FermionOperator, ops.QubitOperator, ops.InteractionOperator, ops.DiagonalCoulombHamiltonian
]


class TrotterStep(metaclass=abc.ABCMeta):
    """A method for performing a Trotter step.

    This class assumes that Hamiltonian evolution using a Trotter-Suzuki product
    formula is performed in the following steps:

        1. Perform some preparatory operations (for instance, a basis change).
        2. Perform a number of Trotter steps. Each Trotter step may induce a
           permutation on the ordering in which qubits represent fermionic
           modes.
        3. Perform some finishing operations.

    Attributes:
        hamiltonian: The Hamiltonian being simulated.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None:
        self.hamiltonian = hamiltonian

    def prepare(
        self, qubits: Sequence[cirq.Qid], control_qubit: Optional[cirq.Qid] = None
    ) -> cirq.OP_TREE:
        """Operations to perform before doing the Trotter steps.

        Args:
            qubits: The qubits on which to perform operations. They should
                be sorted so that the j-th qubit in the Sequence holds the
                occupation of the j-th fermionic mode.
            hamiltonian: The Hamiltonian to simulate.
            control_qubit: The control qubit, if the algorithm is controlled.
        """
        # Default: do nothing
        return ()

    @abc.abstractmethod
    def trotter_step(
        self, qubits: Sequence[cirq.Qid], time: float, control_qubit: Optional[cirq.Qid] = None
    ) -> cirq.OP_TREE:
        """Yield operations to perform a Trotter step.

        Args:
            qubits: The qubits on which to apply the Trotter step.
            hamiltonian: The Hamiltonian to simulate.
            time: The evolution time.
            control_qubit: The control qubit, if the algorithm is controlled.
        """

    def step_qubit_permutation(
        self, qubits: Sequence[cirq.Qid], control_qubit: Optional[cirq.Qid] = None
    ) -> Tuple[Sequence[cirq.Qid], Optional[cirq.Qid]]:
        """The qubit permutation induced by a single Trotter step.

        Returns:
            A tuple whose first element is the new list of system qubits and
            second element is the new control qubit
        """
        # Default: identity permutation
        return qubits, control_qubit

    def finish(
        self,
        qubits: Sequence[cirq.Qid],
        n_steps: int,
        control_qubit: Optional[cirq.Qid] = None,
        omit_final_swaps: bool = False,
    ) -> cirq.OP_TREE:
        """Operations to perform after all Trotter steps are done.

        Args:
            qubits: The qubits on which to perform operations.
            hamiltonian: The Hamiltonian to simulate.
            n_steps: The total number of Trotter steps that have been performed.
            control_qubit: The control qubit, if the algorithm is controlled.
            omit_final_swaps: Whether or not to omit swap gates at the end of
                the circuit.
        """
        # Default: do nothing
        return ()


class TrotterAlgorithm(metaclass=abc.ABCMeta):
    """An algorithm for performing a Trotter step.

    A Trotter step algorithm contains methods for performing a symmetric or
    asymmetric Trotter step and their controlled versions. It does not need
    to support all the possibilities; for instance, it may support only
    symmetric Trotter steps with no control qubit. Support for a kind of
    Trotter step is implemented by overriding the methods of this class.

    Attributes:
        supported_types: A set containing types of Hamiltonian representations
            that can be simulated using this Trotter step algorithm.
            For example, {DiagonalCoulombHamiltonian, InteractionOperator}.
    """

    supported_types = set()  # type: Set[Type[Hamiltonian]]

    def symmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return None

    def asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return None

    def controlled_symmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return None

    def controlled_asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return None
