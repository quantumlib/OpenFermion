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

from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy

import cirq

from openfermion import circuits, linalg


def bogoliubov_transform(
    qubits: Sequence[cirq.Qid],
    transformation_matrix: numpy.ndarray,
    initial_state: Optional[Union[int, Sequence[int]]] = None,
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
        b^\dagger_p = \sum_{q=1}^N W_{pq} a^\dagger_q \\
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
        raise ValueError(
            'Bad shape for transformation_matrix. '
            'Expected {} or {} but got {}.'.format(
                (n_qubits, n_qubits), (n_qubits, 2 * n_qubits), shape
            )
        )

    if isinstance(initial_state, int):
        initial_state = _occupied_orbitals(initial_state, n_qubits)
    initially_occupied_orbitals = cast(Optional[Sequence[int]], initial_state)

    # If the transformation matrix does not mix the two spin sectors, do each
    # sector separately. This is both faster (two half-size decompositions) and
    # more robust: the general fermionic Gaussian decomposition is numerically
    # unreliable on exactly block-structured matrices. See issue #776.
    spin_blocks = _spin_blocks(transformation_matrix)
    if spin_blocks is not None:
        up_block, down_block = spin_blocks
        up_qubits = qubits[: n_qubits // 2]
        down_qubits = qubits[n_qubits // 2 :]

        if initially_occupied_orbitals is None:
            up_orbitals = None
            down_orbitals = None
        else:
            up_orbitals = [i for i in initially_occupied_orbitals if i < n_qubits // 2]
            down_orbitals = [
                i - n_qubits // 2 for i in initially_occupied_orbitals if i >= n_qubits // 2
            ]

        # Spin-down ladder operators carry a Jordan-Wigner Z string over all
        # spin-up qubits, so they anticommute with the spin-up parity operator.
        # If the spin-up circuit flips spin-up parity (an odd number of
        # particle-hole transformations) the factorized circuit would realize
        # the spin-down operators with a flipped sign. A layer of Z gates on the
        # spin-down qubits restores the documented action U a_p^dagger U^-1 =
        # b_p^dagger. When an initial computational basis state is fixed this
        # only contributes an irrelevant global phase, so it is skipped there.
        if initially_occupied_orbitals is None and not _preserves_parity(up_block):
            yield (cirq.Z(q) for q in down_qubits)

        yield bogoliubov_transform(up_qubits, up_block, initial_state=up_orbitals)
        yield bogoliubov_transform(down_qubits, down_block, initial_state=down_orbitals)
        return

    if shape == (n_qubits, n_qubits):
        # We're performing a particle-number conserving "Slater" basis change
        yield _slater_basis_change(qubits, transformation_matrix, initially_occupied_orbitals)
    else:
        # We're performing a more general Gaussian unitary
        yield _gaussian_basis_change(qubits, transformation_matrix, initially_occupied_orbitals)


def _spin_blocks(matrix: numpy.ndarray) -> Optional[Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""Split a transformation matrix into independent spin-sector transforms.

    Returns a pair of transformation matrices (spin-up block, spin-down block)
    if the transformation does not mix the two spin sectors, assuming modes are
    ordered with all spin-up modes before all spin-down modes. Returns None if
    the transformation mixes the sectors.

    For an $N \times N$ matrix the blocks are the diagonal blocks. An
    $N \times 2N$ matrix $W = (W_1 \; W_2)$ acts on the column vector of all
    creation operators followed by all annihilation operators, so it preserves
    the spin sectors if and only if both $W_1$ and $W_2$ are block diagonal;
    the sector transforms are then $(N/2) \times N$ matrices formed from the
    corresponding diagonal blocks of $W_1$ and $W_2$.
    """
    n, m = matrix.shape
    if n % 2:
        return None
    k = n // 2
    if m == n:
        if _is_spin_block_diagonal(matrix):
            return matrix[:k, :k], matrix[k:, k:]
        return None
    left_block = matrix[:, :n]
    right_block = matrix[:, n:]
    if _is_spin_block_diagonal(left_block) and _is_spin_block_diagonal(right_block):
        up_block = numpy.concatenate([left_block[:k, :k], right_block[:k, :k]], axis=1)
        down_block = numpy.concatenate([left_block[k:, k:], right_block[k:, k:]], axis=1)
        return up_block, down_block
    return None


def _is_spin_block_diagonal(matrix) -> bool:
    n = matrix.shape[0]
    if n % 2:
        return False
    max_upper_right = numpy.max(numpy.abs(matrix[: n // 2, n // 2 :]))
    max_lower_left = numpy.max(numpy.abs(matrix[n // 2 :, : n // 2]))
    return numpy.isclose(max_upper_right, 0.0) and numpy.isclose(max_lower_left, 0.0)


def _preserves_parity(transformation_matrix: numpy.ndarray) -> bool:
    r"""Whether the Bogoliubov circuit for this block preserves fermion parity.

    A particle-number-conserving (square) transformation uses only Givens
    rotations and always preserves parity. A general $N \times 2N$ Gaussian
    transformation preserves parity if and only if its fermionic Gaussian
    decomposition contains an even number of particle-hole transformations
    (each of which is realized as an X gate, which flips the parity).
    """
    n, m = transformation_matrix.shape
    if m == n:
        return True
    # Rearrange exactly as _gaussian_basis_change does, so the particle-hole
    # count matches the circuit that will actually be generated.
    left_block = transformation_matrix[:, :n]
    right_block = transformation_matrix[:, n:]
    rearranged = numpy.block([numpy.conjugate(right_block), numpy.conjugate(left_block)])
    decomposition, _, _, _ = linalg.fermionic_gaussian_decomposition(rearranged)
    num_particle_hole = sum(op == 'pht' for parallel_ops in decomposition for op in parallel_ops)
    return num_particle_hole % 2 == 0


def _occupied_orbitals(computational_basis_state: int, n_qubits) -> List[int]:
    """Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4]"""
    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)
    return [j for j in range(len(bitstring)) if bitstring[j] == '1']


def _slater_basis_change(
    qubits: Sequence[cirq.Qid],
    transformation_matrix: numpy.ndarray,
    initially_occupied_orbitals: Optional[Sequence[int]],
) -> cirq.OP_TREE:
    n_qubits = len(qubits)

    if initially_occupied_orbitals is None:
        decomposition, diagonal = linalg.givens_decomposition_square(transformation_matrix)
        circuit_description = list(reversed(decomposition))
        # The initial state is not a computational basis state so the
        # phases left on the diagonal in the decomposition matter
        yield (cirq.rz(rads=numpy.angle(diagonal[j])).on(qubits[j]) for j in range(n_qubits))
    else:
        initially_occupied_orbitals = cast(Sequence[int], initially_occupied_orbitals)
        transformation_matrix = transformation_matrix[list(initially_occupied_orbitals)]
        n_occupied = len(initially_occupied_orbitals)
        # Flip bits so that the first n_occupied are 1 and the rest 0
        initially_occupied_orbitals_set = set(initially_occupied_orbitals)
        yield (
            cirq.X(qubits[j])
            for j in range(n_qubits)
            if (j < n_occupied) != (j in initially_occupied_orbitals_set)
        )
        circuit_description = circuits.slater_determinant_preparation_circuit(transformation_matrix)

    yield _ops_from_givens_rotations_circuit_description(qubits, circuit_description)


def _gaussian_basis_change(
    qubits: Sequence[cirq.Qid],
    transformation_matrix: numpy.ndarray,
    initially_occupied_orbitals: Optional[Sequence[int]],
) -> cirq.OP_TREE:
    n_qubits = len(qubits)

    # Rearrange the transformation matrix because the OpenFermion routine
    # expects it to describe annihilation operators rather than creation
    # operators
    left_block = transformation_matrix[:, :n_qubits]
    right_block = transformation_matrix[:, n_qubits:]
    transformation_matrix = numpy.block([numpy.conjugate(right_block), numpy.conjugate(left_block)])

    decomposition, left_decomposition, _, left_diagonal = linalg.fermionic_gaussian_decomposition(
        transformation_matrix
    )

    if initially_occupied_orbitals is not None and len(initially_occupied_orbitals) == 0:
        # Starting with the vacuum state yields additional symmetry
        circuit_description = list(reversed(decomposition))
    else:
        if initially_occupied_orbitals is None:
            # The initial state is not a computational basis state so the
            # phases left on the diagonal in the Givens decomposition matter
            yield (
                cirq.rz(rads=numpy.angle(left_diagonal[j])).on(qubits[j]) for j in range(n_qubits)
            )
        circuit_description = list(reversed(decomposition + left_decomposition))

    yield _ops_from_givens_rotations_circuit_description(qubits, circuit_description)


def _ops_from_givens_rotations_circuit_description(
    qubits: Sequence[cirq.Qid],
    circuit_description: Iterable[Iterable[Union[str, Tuple[int, int, float, float]]]],
) -> cirq.OP_TREE:
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
                yield cirq.Z(qubits[j]) ** (phi / numpy.pi)
