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

"""Operations for preparing useful quantum states."""

from typing import Iterable, Optional, Sequence, Set, Tuple, Union, cast

import numpy

import cirq
from openfermion import (
        QuadraticHamiltonian,
        gaussian_state_preparation_circuit,
        slater_determinant_preparation_circuit)

from openfermioncirq import Ryxxy


def prepare_gaussian_state(qubits: Sequence[cirq.Qid],
                           quadratic_hamiltonian: QuadraticHamiltonian,
                           occupied_orbitals: Optional[Union[
                               Sequence[int],
                               Tuple[Sequence[int], Sequence[int]]
                               ]]=None,
                           initial_state: Union[int, Sequence[int]]=0
                           ) -> cirq.OP_TREE:
    """Prepare a fermionic Gaussian state from a computational basis state.

    A fermionic Gaussian state is an eigenstate of a quadratic Hamiltonian. If
    the Hamiltonian conserves particle number, then it is a Slater determinant.
    The algorithm used is described in arXiv:1711.05395. It assumes the
    Jordan-Wigner transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        quadratic_hamiltonian: The Hamiltonian whose eigenstate is desired.
        occupied_orbitals: Integers representing the indices of the
            pseudoparticle orbitals to occupy in the Gaussian state.
            If two lists are given, then the first list specifies spin-up
            orbitals and the second list specifies spin-down orbitals,
            and the modes are assumed to be ordered so that
            spin-up modes come before spin-down modes.
            Two lists should be given only if the Hamiltonian contains a
            spin degree of freedom and modes with different spin do not
            interact.
            The orbitals are ordered in ascending order of energy.
            The default behavior is to fill the orbitals with negative energy,
            i.e., prepare the ground state.
        initial_state: The computational basis state that the qubits start in.
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
    if not occupied_orbitals or isinstance(occupied_orbitals[0], int):
        # Generic
        occupied_orbitals = cast(Sequence[int], occupied_orbitals)
        yield _generic_gaussian_circuit(
                qubits, quadratic_hamiltonian, occupied_orbitals, initial_state)
    else:
        # Spin symmetry
        occupied_orbitals = cast(Tuple[Sequence[int], Sequence[int]],
                                 occupied_orbitals)
        yield _spin_symmetric_gaussian_circuit(
                qubits, quadratic_hamiltonian, occupied_orbitals, initial_state)


def _generic_gaussian_circuit(
        qubits: Sequence[cirq.Qid],
        quadratic_hamiltonian: QuadraticHamiltonian,
        occupied_orbitals: Optional[Sequence[int]],
        initial_state: Union[int, Sequence[int]]) -> cirq.OP_TREE:

    n_qubits = len(qubits)
    circuit_description, start_orbitals = gaussian_state_preparation_circuit(
            quadratic_hamiltonian, occupied_orbitals)

    if isinstance(initial_state, int):
        initially_occupied_orbitals = _occupied_orbitals(
                initial_state, n_qubits)
    else:
        initially_occupied_orbitals = initial_state  # type: ignore

    # Flip bits so that the correct starting orbitals are occupied
    yield (cirq.X(qubits[j]) for j in range(n_qubits)
           if (j in initially_occupied_orbitals) != (j in start_orbitals))

    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def _spin_symmetric_gaussian_circuit(
        qubits: Sequence[cirq.Qid],
        quadratic_hamiltonian: QuadraticHamiltonian,
        occupied_orbitals: Tuple[Sequence[int], Sequence[int]],
        initial_state: Union[int, Sequence[int]]) -> cirq.OP_TREE:

    n_qubits = len(qubits)

    if isinstance(initial_state, int):
        initially_occupied_orbitals = _occupied_orbitals(
                initial_state, n_qubits)
    else:
        initially_occupied_orbitals = initial_state  # type: ignore

    for spin_sector in range(2):
        circuit_description, start_orbitals = (
                gaussian_state_preparation_circuit(
                    quadratic_hamiltonian,
                    occupied_orbitals[spin_sector],
                    spin_sector=spin_sector)
        )

        def index_map(i):
            return i + spin_sector*(n_qubits // 2)
        spin_indices = [index_map(i) for i in range(n_qubits // 2)]
        spin_qubits = [qubits[i] for i in spin_indices]

        # Flip bits so that the correct starting orbitals are occupied
        yield (cirq.X(spin_qubits[j]) for j in range(n_qubits // 2)
               if (index_map(j) in initially_occupied_orbitals)
               != (index_map(j) in [index_map(k) for k in start_orbitals]))

        yield _ops_from_givens_rotations_circuit_description(
                spin_qubits, circuit_description)


def prepare_slater_determinant(qubits: Sequence[cirq.Qid],
                               slater_determinant_matrix: numpy.ndarray,
                               initial_state: Union[int, Sequence[int]]=0
                               ) -> cirq.OP_TREE:
    r"""Prepare a Slater determinant from a computational basis state.

    A Slater determinant is described by an :math:`\eta \times N` matrix
    :math:`Q` with orthonormal rows, where :math:`\eta` is the particle number
    and :math:`N` is the total number of modes. The state corresponding to this
    matrix is

    .. math::

        b^\dagger_1 \cdots b^\dagger_{\eta} \lvert \text{vac} \rangle,

    where

    .. math::

        b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k.

    The algorithm used is described in arXiv:1711.05395. It assumes the
    Jordan-Wigner transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        slater_determinant_matrix: The matrix :math:`Q` which describes the
            Slater determinant to be prepared.
        initial_state: The computational basis state that the qubits start in.
            This can be either an integer or a container of integers.
            If an integer, it is mapped to a computational basis state via
            "big endian" ordering of the binary representation of the integer.
            For example, the computational basis state on five qubits with
            the first and second qubits set to one is 0b11000, which is 24
            in decimal.
            If a container of integers, then it contains the indices of the
            qubits that are set to one (indexing starts from 0). For
            example, the list [2, 3] represents qubits 2 and 3 being set to one.
            Default is 0, the all zeros state.
    """
    n_qubits = len(qubits)
    circuit_description = slater_determinant_preparation_circuit(
            slater_determinant_matrix)
    n_occupied = slater_determinant_matrix.shape[0]

    if isinstance(initial_state, int):
        initially_occupied_orbitals = _occupied_orbitals(
                initial_state, n_qubits)
    else:
        initially_occupied_orbitals = initial_state  # type: ignore

    # Flip bits so that the first n_occupied are 1 and the rest 0
    yield (cirq.X(qubits[j]) for j in range(n_qubits)
           if (j < n_occupied) != (j in initially_occupied_orbitals))

    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def _occupied_orbitals(computational_basis_state: int, n_qubits) -> Set[int]:
    """Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4]"""
    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)
    return {j for j in range(len(bitstring)) if bitstring[j] == '1'}


def _ops_from_givens_rotations_circuit_description(
        qubits: Sequence[cirq.Qid],
        circuit_description: Iterable[Iterable[
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
                yield Ryxxy(theta).on(qubits[i], qubits[j])
                yield cirq.Z(qubits[j]) ** (phi / numpy.pi)
