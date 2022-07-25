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
"""The fast fermionic Fourier transform."""

from typing import (Iterable, List, Sequence)

import numpy as np
from sympy.ntheory import factorint

import cirq
import cirq.contrib.acquaintance.permutation
from openfermion.circuits.gates import FSWAP
from .bogoliubov_transform import bogoliubov_transform


class _F0Gate(cirq.MatrixGate):
    r"""Two-qubit gate that performs fermionic Fourier transform of size 2.

    Realizes unitary gate $F_0$ that transforms Fermionic creation
    operators $a_0^\dagger$ and $a_1^\dagger$ according to:

    $$
        F_0^\dagger a_0^\dagger F_0 =
            {1 \over \sqrt{2}} (a_0^\dagger + a_1^\dagger)
    $$

    $$
        F_0^\dagger a_1^\dagger F_0 =
            {1 \over \sqrt{2}} (a_0^\dagger - a_1^\dagger) \, .
    $$

    This gate assumes JWT representation of fermionic modes which are big-endian
    encoded on consecutive qubits:
    $a_0^\dagger \lvert 00 \rangle = \lvert 10 \rangle$ and
    $a_1^\dagger \lvert 00 \rangle = \vert 01 \rangle$.

    Internally, this leads to expansion of $F_0^\dagger$:

    $$
        \langle 00 \rvert F_0^\dagger \lvert 00 \rangle = 1
    $$

    $$
        \langle 01 \rvert F_0^\dagger \lvert 01 \rangle =
            -{1 \over \sqrt{2}}
    $$

    $$
        \langle 10 \rvert F_0^\dagger \lvert 10 \rangle =
        \langle 10 \rvert F_0^\dagger \lvert 01 \rangle =
        \langle 01 \rvert F_0^\dagger \lvert 10 \rangle = {1 \over \sqrt{2}}
    $$

    $$
        \langle 11 \rvert F_0^\dagger \lvert 11 \rangle = -1 \, .
    $$
    """

    def __init__(self):
        """Initializes $F_0$ gate."""
        cirq.MatrixGate.__init__(self,
                                 np.array([[1, 0, 0, 0],
                                           [0, -2**(-0.5), 2**(-0.5), 0],
                                           [0, 2**(-0.5), 2**(-0.5), 0],
                                           [0, 0, 0, -1]]),
                                 qid_shape=(2, 2))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                              ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = 'F₀', 'F₀'
        else:
            symbols = 'F0', 'F0'
        return cirq.CircuitDiagramInfo(wire_symbols=symbols)


class _TwiddleGate(cirq.Gate):
    r"""Gate that introduces arbitrary FFT twiddle factors.

    Realizes unitary gate $\omega^{k\dagger}_n$ that phases creation
    operator $a^\dagger_x$ according to:

    $$
        \omega^{k\dagger}_n a^\dagger_x \omega^k_n =
            e^{-2 \pi i {k \over n}} a^\dagger_x \, .
    $$

    Under JWT representation this is realized by appropriately rotated pauli Z
    gate acting on qubit x.
    """

    def __init__(self, k, n):
        """Initializes Twiddle gate.

        Args:
            k: Numerator appearing in the exponent.
            n: Denominator appearing in the exponent.
        """
        self.k = k
        self.n = n

    def _num_qubits_(self) -> int:
        return 1

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                              ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = 'ω^{}_{}'.format(self.k, self.n),
        else:
            symbols = 'w^{}_{}'.format(self.k, self.n),
        return cirq.CircuitDiagramInfo(wire_symbols=symbols)

    def _decompose_(self, qubits: Iterable[cirq.Qid]):
        q, = qubits
        exponent = -2 * self.k / self.n
        yield cirq.ZPowGate(exponent=exponent, global_shift=0).on(q)


F0 = _F0Gate()


def ffft(qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
    r"""Performs fast fermionic Fourier transform.

    Generates a circuit that performs fast fermionic Fourier transform (FFFT)
    which transforms a set of fermionic creation operators
    $\hat{a}_n^\dagger$, $n \in 1, 2, \dots, N$ according to:

    $$
        \mathit{FFFT}^\dagger \tilde{a}_k^\dagger \mathit{FFFT} =
            {1 \over \sqrt{N}}
            \sum_{n=0}^{N-1} e^{-i {2\pi \over N} n k} \hat{a}^\dagger_n \, ,
    $$

    where $\tilde{a}_k^\dagger$ are transformed operators and $N$ is
    size of the input `qubits` sequence.

    This function assumes JWT representation of fermionic modes which are
    big-endian encoded on consecutive qubits:
    $a_0^\dagger \lvert 0.. \rangle = \lvert 1.. \rangle$,
    $a_1^\dagger \lvert 0.. \rangle = \vert 01.. \rangle$,
    $a_2^\dagger \lvert 0.. \rangle = \vert 001.. \rangle$, $\dots$.

    The gate count of generated circuit is $\theta(N^2)$, generated
    circuit depth is $\theta(N)$ and distinct gates count is
    $\theta(N_1^2 + N_2^2 + \dots + N_n^2)$, where
    $N = N_1 N_2 \dots N_n$ is prime decomposition of $N$. In a case
    where $N$ is some power of 2, it reduces to $\theta(\log(N))$.

    An equivalent circuit can be generated using the
    `openfermion.bogoliubov_transform` function with appropriately prepared
    `transformation_matrix` argument:

    .. testcode::

        import openfermion
        import numpy as np

        def ffft(qubits):
            def fourier_transform_matrix(size):
                root_of_unity = np.exp(-2j * np.pi / size)
                return np.array([[root_of_unity ** (k * n) for n in range(size)]
                                 for k in range(size)]) / np.sqrt(size)

            return openfermion.bogoliubov_transform(
                qubits, fourier_transform_matrix(len(qubits)))

    The advantage of circuit generated by the FFFT algorithm over the one
    created with the `bogoliubov_transform` is that a smaller variety of gates
    is created, which is $O(N^2)$ in case of pure `bogoliubov_transform`.
    This implementation of `FFFT` fall-backs to the `bogoliubov_transform` for
    the inputs where qubits length is prime.

    This implementation of FFFT is based on a generalized, composite size
    Cooley-Tukey algorithm and adapted to nearest neighbourhood connectivity.

    Args:
        qubits: Sequence of qubits that the FFFT circuit will be generated for.
            This sequence represents a sequence of consecutive creation
            operators under big-endian encoded JWT representation. The indices
            assignment is significant since it it used to define the FFFT
            operation itself. The input sequence is assumed to have nearest
            neighbourhood connectivity.

    Returns:
        Circuit that performs FFFT on given qubits.

    Raises:
        ValueError: When length of the input array is not a power of 2.
    """
    n = len(qubits)

    if n == 0:
        raise ValueError('Number of qubits is 0.')

    if n == 1:
        return []

    factors = [f for f, count in factorint(n).items() for _ in range(count)]
    return _ffft(qubits, factors)


def _ffft_prime(qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:

    def fft_matrix(n):
        unit = np.exp(-2j * np.pi / n)
        return np.array([[unit**(j * k) for k in range(n)] for j in range(n)
                        ]) / np.sqrt(n)

    n = len(qubits)

    if n == 2:
        return F0(*qubits)
    else:
        return bogoliubov_transform(qubits, fft_matrix(n))


def _ffft(qubits: Sequence[cirq.Qid], factors: List[int]) -> cirq.OP_TREE:
    if len(factors) == 1:
        return _ffft_prime(qubits)

    n = len(qubits)

    # ny is a first prime factor, nx is a product of all the remaining ones.
    factors_y = factors[:1]
    factors_x = factors[1:]
    ny = np.prod(factors_y)
    nx = n // ny

    permutation = [(i % ny) * nx + (i // ny) for i in range(n)]
    permutation_gate = _permute(qubits, permutation)

    operations = []

    # This follows composite case Cooley-Tukey algorithm. The first step is to
    # recursively conduct ny FFFTs of size nx, where each of the FFFTs is
    # performed on nx consecutive qubits that are ny indices apart in the
    # original sequence.
    # This operation shuffles input qubits so that they are grouped together
    # accordingly. When ny = 2, this operation shuffles qubits to place odd
    # qubits in the first half of a sequence, and even qubits in the second
    # half.
    operations.append(permutation_gate)

    # Performs ny recursive FFFTs, each of size nx.
    for y in range(ny):
        operations.append(_ffft(qubits[nx * y:nx * (y + 1)], factors_x))

    # The second part is to perform ny FFFTs of size nx on qubits which are
    # consecutive in the original sequence. To place qubits in a correct order
    # the original permutation operation is inverted and applied to the previous
    # result.
    operations.append(cirq.inverse(permutation_gate))

    # Performs the nx FFFTs, each of size ny.
    for x in range(nx):
        for y in range(1, ny):
            operations.append(_TwiddleGate(x * y, n).on(qubits[ny * x + y]))
        operations.append(_ffft(qubits[ny * x:ny * (x + 1)], factors_y))

    # The result of performing FFFT on k-th group of nx qubits is a new group
    # of fermionic operators with indices k, k + ny, k + 2ny, ... of the final
    # fermionic operators sequence. The first permutation is applied again,
    # which puts every qubit in its final position.
    operations.append(permutation_gate)

    return operations


def _permute(qubits: Sequence[cirq.Qid],
             permutation: List[int]) -> cirq.OP_TREE:
    """
    Generates a circuit which reorders Fermionic modes.

    JWT representation of Fermionic modes is assumed. This is just a wrapper
    around cirq.contrib.acquaintance.permutation.LinearPermutationGate which
    internally uses bubble sort algorithm to generate permutation gate.

    Args:
        qubits: Sequence of qubits to reorder. Line connectivity is assumed.
        permutation: The permutation function represented as a list that
            reorders the initial qubits. Specifically, if k-th element of
            permutation is j, then k-th qubit should become the j-th qubit after
            applying the circuit to the initial state.

    Return:
        Gate that reorders the qubits accordingly.
    """
    return cirq.contrib.acquaintance.permutation.LinearPermutationGate(
        len(qubits), {i: permutation[i] for i in range(len(permutation))},
        FSWAP).on(*qubits)
