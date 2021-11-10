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
"""The Bogoliubov transformation."""

from typing import (Iterable, List, Optional, Sequence, Tuple, Union, cast)

import numpy

import cirq

from openfermion import circuits, linalg


def bogoliubov_transform(
        qubits: Sequence[cirq.Qid],
        transformation_matrix: numpy.ndarray,
        initial_state: Optional[Union[int, Sequence[int]]] = None
) -> cirq.OP_TREE:
    r"""Perform a Bogoliubov transformation.

    This circuit performs the transformation to a basis determined by a new set
    of fermionic ladder operators. It performs the unitary $U$ such that

    $$
        U a^\dagger_p U^{-1} = b^\dagger_p
    $$

    where the $a^\dagger_p$ are the original creation operators and the
    $b^\dagger_p$ are the new creation operators. The new creation
    operators are linear combinations of the original ladder operators with
    coefficients given by the matrix `transformation_matrix`, which will be
    referred to as $W$ in the following.

    If $W$ is an $N \times N$ matrix, then the $b^\dagger_p$ are
    given by

    $$
        b^\dagger_p = \sum_{q=1}^N W_{pq} a^\dagger_q.
    $$

    If $W$ is an $N \times 2N$ matrix, then the $b^\dagger_p$ are
    given by

    $$
        b^\dagger_p = \sum_{q=1}^N W_{pq} a^\dagger_q
                      + \sum_{q=N+1}^{2N} W_{pq} a_q.
    $$

    This algorithm assumes the Jordan-Wigner Transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        transformation_matrix: The matrix $W$ holding the coefficients
            that describe the new creation operators in terms of the original
            ladder operators. Its shape should be either $NxN$ or
            $Nx(2N)$, where $N$ is the number of qubits.
        initial_state: Optionally specifies a computational basis state
            to assume that the qubits start in. This assumption enables
            optimizations that result in a circuit with fewer gates.
            This can be either an integer or a sequence of integers.
            If an integer, it is mapped to a computational basis state via
            "big endian" ordering of the binary representation of the integer.
            For example, the computational basis state on five qubits with
            the first and second qubits set to one is 0b11000, which is 24
            in decimal.
            If a sequence of integers, then it contains the indices of the
            qubits that are set to one (indexing starts from 0). For
            example, the list [2, 3] represents qubits 2 and 3 being set to one.
            Default is 0, the all zeros state.
    """

    n_qubits = len(qubits)
    shape = transformation_matrix.shape
    if shape not in [(n_qubits, n_qubits), (n_qubits, 2 * n_qubits)]:
        raise ValueError('Bad shape for transformation_matrix. '
                         'Expected {} or {} but got {}.'.format(
                             (n_qubits, n_qubits), (n_qubits, 2 * n_qubits),
                             shape))

    if isinstance(initial_state, int):
        initial_state = _occupied_orbitals(initial_state, n_qubits)
    initially_occupied_orbitals = cast(Optional[Sequence[int]], initial_state)

    # If the transformation matrix is block diagonal with two blocks,
    # do each block separately
    if _is_spin_block_diagonal(transformation_matrix):
        up_block = transformation_matrix[:n_qubits // 2, :n_qubits // 2]
        up_qubits = qubits[:n_qubits // 2]

        down_block = transformation_matrix[n_qubits // 2:, n_qubits // 2:]
        down_qubits = qubits[n_qubits // 2:]

        if initially_occupied_orbitals is None:
            up_orbitals = None
            down_orbitals = None
        else:
            up_orbitals = [
                i for i in initially_occupied_orbitals if i < n_qubits // 2
            ]
            down_orbitals = [
                i - n_qubits // 2
                for i in initially_occupied_orbitals
                if i >= n_qubits // 2
            ]

        yield bogoliubov_transform(up_qubits,
                                   up_block,
                                   initial_state=up_orbitals)
        yield bogoliubov_transform(down_qubits,
                                   down_block,
                                   initial_state=down_orbitals)
        return

    if shape == (n_qubits, n_qubits):
        # We're performing a particle-number conserving "Slater" basis change
        yield _slater_basis_change(qubits, transformation_matrix,
                                   initially_occupied_orbitals)
    else:
        # We're performing a more general Gaussian unitary
        yield _gaussian_basis_change(qubits, transformation_matrix,
                                     initially_occupied_orbitals)


def _is_spin_block_diagonal(matrix) -> bool:
    n = matrix.shape[0]
    if n % 2:
        return False
    max_upper_right = numpy.max(numpy.abs(matrix[:n // 2, n // 2:]))
    max_lower_left = numpy.max(numpy.abs(matrix[n // 2:, :n // 2]))
    return (numpy.isclose(max_upper_right, 0.0) and
            numpy.isclose(max_lower_left, 0.0))


def _occupied_orbitals(computational_basis_state: int, n_qubits) -> List[int]:
    """Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4]"""
    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)
    return [j for j in range(len(bitstring)) if bitstring[j] == '1']


def _slater_basis_change(qubits: Sequence[cirq.Qid],
                         transformation_matrix: numpy.ndarray,
                         initially_occupied_orbitals: Optional[Sequence[int]]
                        ) -> cirq.OP_TREE:
    n_qubits = len(qubits)

    if initially_occupied_orbitals is None:
        decomposition, diagonal = linalg.givens_decomposition_square(
            transformation_matrix)
        circuit_description = list(reversed(decomposition))
        # The initial state is not a computational basis state so the
        # phases left on the diagonal in the decomposition matter
        yield (cirq.rz(rads=numpy.angle(diagonal[j])).on(qubits[j])
               for j in range(n_qubits))
    else:
        initially_occupied_orbitals = cast(Sequence[int],
                                           initially_occupied_orbitals)
        transformation_matrix = transformation_matrix[list(
            initially_occupied_orbitals)]
        n_occupied = len(initially_occupied_orbitals)
        # Flip bits so that the first n_occupied are 1 and the rest 0
        initially_occupied_orbitals_set = set(initially_occupied_orbitals)
        yield (cirq.X(qubits[j])
               for j in range(n_qubits)
               if (j < n_occupied) != (j in initially_occupied_orbitals_set))
        circuit_description = circuits.slater_determinant_preparation_circuit(
            transformation_matrix)

    yield _ops_from_givens_rotations_circuit_description(
        qubits, circuit_description)


def _gaussian_basis_change(qubits: Sequence[cirq.Qid],
                           transformation_matrix: numpy.ndarray,
                           initially_occupied_orbitals: Optional[Sequence[int]]
                          ) -> cirq.OP_TREE:
    n_qubits = len(qubits)

    # Rearrange the transformation matrix because the OpenFermion routine
    # expects it to describe annihilation operators rather than creation
    # operators
    left_block = transformation_matrix[:, :n_qubits]
    right_block = transformation_matrix[:, n_qubits:]
    transformation_matrix = numpy.block(
        [numpy.conjugate(right_block),
         numpy.conjugate(left_block)])

    decomposition, left_decomposition, _, left_diagonal = (
        linalg.fermionic_gaussian_decomposition(transformation_matrix))

    if (initially_occupied_orbitals is not None and
            len(initially_occupied_orbitals) == 0):
        # Starting with the vacuum state yields additional symmetry
        circuit_description = list(reversed(decomposition))
    else:
        if initially_occupied_orbitals is None:
            # The initial state is not a computational basis state so the
            # phases left on the diagonal in the Givens decomposition matter
            yield (cirq.rz(rads=numpy.angle(left_diagonal[j])).on(qubits[j])
                   for j in range(n_qubits))
        circuit_description = list(reversed(decomposition + left_decomposition))

    yield _ops_from_givens_rotations_circuit_description(
        qubits, circuit_description)


def _ops_from_givens_rotations_circuit_description(
        qubits: Sequence[cirq.Qid], circuit_description: Iterable[Iterable[
            Union[str, Tuple[int, int, float, float]]]]) -> cirq.OP_TREE:
    """Yield operations from a Givens rotations circuit obtained from
    OpenFermion.
    """
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            if op == 'pht':
                yield cirq.X(qubits[-1])
            else:
                i, j, theta, phi = cast(Tuple[int, int, float, float], op)
                yield circuits.Ryxxy(theta).on(qubits[i], qubits[j])
                yield cirq.Z(qubits[j])**(phi / numpy.pi)
