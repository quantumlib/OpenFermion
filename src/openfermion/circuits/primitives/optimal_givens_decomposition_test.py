# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import product
import numpy
import scipy
import cirq

from openfermion.linalg import (givens_matrix_elements, givens_rotate,
                                get_sparse_operator)
from openfermion.ops import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from openfermion.circuits.primitives.optimal_givens_decomposition import \
    optimal_givens_decomposition


def test_givens_inverse():
    r"""
    The Givens rotation in OpenFermion is defined as

    $$
        \begin{pmatrix}
            \cos(\theta) & -e^{i \varphi} \sin(\theta) \\
            \sin(\theta) &     e^{i \varphi} \cos(\theta)
        \end{pmatrix}.
    $$

    confirm numerically its hermitian conjugate is it's inverse
    """
    a = numpy.random.random() + 1j * numpy.random.random()
    b = numpy.random.random() + 1j * numpy.random.random()
    ab_rotation = givens_matrix_elements(a, b, which='right')

    assert numpy.allclose(ab_rotation.dot(numpy.conj(ab_rotation).T),
                          numpy.eye(2))
    assert numpy.allclose(
        numpy.conj(ab_rotation).T.dot(ab_rotation), numpy.eye(2))


def test_row_eliminate():
    """
    Test elemination of element in U[i, j] by rotating in i-1 and i.
    """
    dim = 3
    u_generator = numpy.random.random((dim, dim)) + 1j * numpy.random.random(
        (dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T

    # make sure the generator is actually antihermitian
    assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)

    unitary = scipy.linalg.expm(u_generator)

    # eliminate U[2, 0] by rotating in 1, 2
    gmat = givens_matrix_elements(unitary[1, 0], unitary[2, 0], which='right')
    givens_rotate(unitary, gmat, 1, 2, which='row')
    assert numpy.isclose(unitary[2, 0], 0.0)

    # eliminate U[1, 0] by rotating in 0, 1
    gmat = givens_matrix_elements(unitary[0, 0], unitary[1, 0], which='right')
    givens_rotate(unitary, gmat, 0, 1, which='row')
    assert numpy.isclose(unitary[1, 0], 0.0)

    # eliminate U[2, 1] by rotating in 1, 2
    gmat = givens_matrix_elements(unitary[1, 1], unitary[2, 1], which='right')
    givens_rotate(unitary, gmat, 1, 2, which='row')
    assert numpy.isclose(unitary[2, 1], 0.0)


def create_givens(givens_mat, i, j, dim):
    """
    Create the givens matrix on the larger space

    :param givens_mat: 2x2 matrix with first column is real
    :param i: row index i
    :param j: row index i < j
    :param dim: dimension
    """
    gmat = numpy.eye(dim, dtype=complex)
    gmat[i, i] = givens_mat[0, 0]
    gmat[i, j] = givens_mat[0, 1]
    gmat[j, i] = givens_mat[1, 0]
    gmat[j, j] = givens_mat[1, 1]
    return gmat


def test_col_eliminate():
    """
    Test elimination by rotating in the column space.  Left multiplication of
    inverse givens
    """
    dim = 3
    u_generator = numpy.random.random((dim, dim)) + 1j * numpy.random.random(
        (dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T
    # make sure the generator is actually antihermitian
    assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)
    unitary = scipy.linalg.expm(u_generator)

    # eliminate U[1, 0] by rotation in rows [0, 1] and
    # mixing U[1, 0] and U[0, 0]
    unitary_original = unitary.copy()
    gmat = givens_matrix_elements(unitary[0, 0], unitary[1, 0], which='right')
    vec = numpy.array([[unitary[0, 0]], [unitary[1, 0]]])
    fullgmat = create_givens(gmat, 0, 1, 3)
    zeroed_unitary = fullgmat.dot(unitary)

    givens_rotate(unitary, gmat, 0, 1)
    assert numpy.isclose(unitary[1, 0], 0.0)
    assert numpy.allclose(unitary.real, zeroed_unitary.real)
    assert numpy.allclose(unitary.imag, zeroed_unitary.imag)

    # eliminate U[2, 0] by rotating columns [0, 1] and
    # mixing U[2, 0] and U[2, 1].
    unitary = unitary_original.copy()
    gmat = givens_matrix_elements(unitary[2, 0], unitary[2, 1], which='left')
    vec = numpy.array([[unitary[2, 0]], [unitary[2, 1]]])

    assert numpy.isclose((gmat.dot(vec))[0, 0], 0.0)
    assert numpy.isclose((vec.T.dot(gmat.T))[0, 0], 0.0)
    fullgmat = create_givens(gmat, 0, 1, 3)
    zeroed_unitary = unitary.dot(fullgmat.T)

    # because col takes g[0, 0] * col_i + g[0, 1].conj() * col_j -> col_i
    # this is equivalent ot left multiplication by gmat.T
    givens_rotate(unitary, gmat.conj(), 0, 1, which='col')
    assert numpy.isclose(zeroed_unitary[2, 0], 0.0)
    assert numpy.allclose(unitary, zeroed_unitary)


def test_front_back_iteration():
    """
    Code demonstrating how we iterated over the matrix

    [[ 0.  0.  0.  0.  0.  0.]
     [15.  0.  0.  0.  0.  0.]
     [ 7. 14.  0.  0.  0.  0.]
     [ 6.  8. 13.  0.  0.  0.]
     [ 2.  5.  9. 12.  0.  0.]
     [ 1.  3.  4. 10. 11.  0.]]
    """
    N = 6
    unitary = numpy.zeros((N, N))
    unitary[-1, 0] = 1
    unitary[-2, 0] = 2
    unitary[-1, 1] = 3
    unitary[-1, 2] = 4
    unitary[-2, 1] = 5
    unitary[-3, 0] = 6
    unitary[-4, 0] = 7
    unitary[-3, 1] = 8
    unitary[-2, 2] = 9
    unitary[-1, 3] = 10
    unitary[-1, 4] = 11
    unitary[-2, 3] = 12
    unitary[-3, 2] = 13
    unitary[-4, 1] = 14
    unitary[-5, 0] = 15
    counter = 1
    for i in range(1, N):
        if i % 2 == 1:
            for j in range(0, i):
                print((N - j, i - j), i - j, i - j + 1, "col rotation")
                assert numpy.isclose(unitary[N - j - 1, i - j - 1], counter)
                counter += 1
        else:
            for j in range(1, i + 1):
                print((N + j - i, j), N + j - i - 1, N + j - i, "row rotation")
                assert numpy.isclose(unitary[N + j - i - 1, j - 1], counter)
                counter += 1


def test_circuit_generation_and_accuracy():
    for dim in range(2, 10):
        qubits = cirq.LineQubit.range(dim)
        u_generator = numpy.random.random(
            (dim, dim)) + 1j * numpy.random.random((dim, dim))
        u_generator = u_generator - numpy.conj(u_generator).T
        assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)

        unitary = scipy.linalg.expm(u_generator)
        circuit = cirq.Circuit()
        circuit.append(optimal_givens_decomposition(qubits, unitary))

        fermion_generator = QubitOperator(()) * 0.0
        for i, j in product(range(dim), repeat=2):
            fermion_generator += jordan_wigner(
                FermionOperator(((i, 1), (j, 0)), u_generator[i, j]))

        true_unitary = scipy.linalg.expm(
            get_sparse_operator(fermion_generator).toarray())
        assert numpy.allclose(true_unitary.conj().T.dot(true_unitary),
                              numpy.eye(2**dim, dtype=complex))

        test_unitary = cirq.unitary(circuit)
        assert numpy.isclose(
            abs(numpy.trace(true_unitary.conj().T.dot(test_unitary))), 2**dim)


def test_circuit_generation_state():
    """
    Determine if we rotate the Hartree-Fock state correctly
    """
    simulator = cirq.Simulator()
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit.append([
        cirq.X(qubits[0]),
        cirq.X(qubits[1]),
        cirq.X(qubits[1]),
        cirq.X(qubits[2]),
        cirq.X(qubits[3]),
        cirq.X(qubits[3])
    ])  # alpha-spins are first then beta spins

    wavefunction = numpy.zeros((2**4, 1), dtype=complex)
    wavefunction[10, 0] = 1.0

    dim = 2
    u_generator = numpy.random.random((dim, dim)) + 1j * numpy.random.random(
        (dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T
    unitary = scipy.linalg.expm(u_generator)

    circuit.append(optimal_givens_decomposition(qubits[:2], unitary))

    fermion_generator = QubitOperator(()) * 0.0
    for i, j in product(range(dim), repeat=2):
        fermion_generator += jordan_wigner(
            FermionOperator(((i, 1), (j, 0)), u_generator[i, j]))

    test_unitary = scipy.linalg.expm(
        get_sparse_operator(fermion_generator, 4).toarray())
    test_final_state = test_unitary.dot(wavefunction)
    cirq_wf = simulator.simulate(circuit).final_state_vector

    assert numpy.allclose(cirq_wf, test_final_state.flatten())
