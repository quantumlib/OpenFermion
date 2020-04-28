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

"""bravyi_kitaev_fast transform on fermionic operators."""

import numpy
import networkx

from openfermion.ops import InteractionOperator, QubitOperator
from openfermion.utils import count_qubits


def bravyi_kitaev_fast(operator):
    """
    Find the Pauli-representation of InteractionOperator for Bravyi-Kitaev
    Super fast (BKSF) algorithm. Pauli-representation of general
    FermionOperator is not possible in BKSF. Also, the InteractionOperator
    given as input must be Hermitian. In future we might provide a
    transformation for a restricted set of fermion operator.

    Args:
        operator: Interaction Operator.

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Raises:
        TypeError: If operator is not an InteractionOperator

    """
    if isinstance(operator, InteractionOperator):
        return bravyi_kitaev_fast_interaction_op(operator)
    else:
        raise TypeError("operator must be an InteractionOperator.")


def bravyi_kitaev_fast_interaction_op(iop):
    r"""
    Transform from InteractionOpeator to QubitOperator for Bravyi-Kitaev fast
    algorithm.

    The electronic Hamiltonian is represented in terms of creation and
    annihilation operators. These creation and annihilation operators could be
    used to define Majorana modes as follows:
        c_{2i} = a_i + a^{\dagger}_i,
        c_{2i+1} = (a_i - a^{\dagger}_{i})/(1j)
    These Majorana modes can be used to define edge operators B_i and A_{ij}:
        B_i=c_{2i}c_{2i+1},
        A_{ij}=c_{2i}c_{2j}
    using these edge operators the fermionic algebra can be generated and
    hence all the terms in the electronic Hamiltonian can be expressed in
    terms of edge operators. The terms in electronic Hamiltonian can be
    divided into five types (arXiv 1208.5986). We can find the edge operator
    expression for each of those five types. For example, the excitation
    operator term in Hamiltonian when represented in terms of edge operators
    becomes:
        a_i^{\dagger}a_j+a_j^{\dagger}a_i = (-1j/2)*(A_ij*B_i+B_j*A_ij)
    For the sake of brevity the reader is encouraged to look up the
    expressions of other terms from the code below. The variables for edge
    operators are chosen according to the nomenclature defined above
    (B_i and A_ij). A detailed description of these operators and the terms
    of the electronic Hamiltonian are provided in (arXiv 1712.00446).

    Args:
        iop (Interaction Operator):
        n_qubit (int): Number of qubits

    Returns:
        qubit_operator: An instance of the QubitOperator class.
    """
    n_qubits = count_qubits(iop)

    # Initialize qubit operator as constant.
    qubit_operator = QubitOperator((), iop.constant)
    edge_matrix = bravyi_kitaev_fast_edge_matrix(iop)
    edge_matrix_indices = numpy.array(numpy.nonzero(numpy.triu(edge_matrix) -
                                      numpy.diag(numpy.diag(edge_matrix))))
    # Loop through all indices.
    for p in range(n_qubits):
        for q in range(n_qubits):
            # Handle one-body terms.
            coefficient = complex(iop[(p, 1), (q, 0)])
            if coefficient and p >= q:
                qubit_operator += (coefficient *
                                   one_body(edge_matrix_indices, p, q))

            # Keep looping for the two-body terms.
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coefficient = complex(iop[(p, 1), (q, 1), (r, 0), (s, 0)])

                    # Skip zero terms.
                    if (not coefficient) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        # Handle case of 3 unique indices
                        elif len(set([p, q, r, s])) == 3:
                            transformed_term = two_body(edge_matrix_indices,
                                                        p, q, r, s)
                            transformed_term *= .5 * coefficient
                            qubit_operator += transformed_term
                            continue
                        elif p != r and q < p:
                            continue


                    # Handle the two-body terms.
                    transformed_term = two_body(edge_matrix_indices,
                                                p, q, r, s)
                    transformed_term *= coefficient
                    qubit_operator += transformed_term
    return qubit_operator


def bravyi_kitaev_fast_edge_matrix(iop, n_qubits=None):
    """
    Use InteractionOperator to construct edge matrix required for the algorithm

    Edge matrix contains the information about the edges between vertices.
    Edge matrix is required to build the operators in bravyi_kitaev_fast model.

    Args:
        iop (Interaction Operator):

    Returns:
        edge_matrix (Numpy array): A square numpy array containing information
            about the edges present in the model.
    """
    n_qubits = count_qubits(iop)
    edge_matrix = 1j*numpy.zeros((n_qubits, n_qubits))
    # Loop through all indices.
    for p in range(n_qubits):
        for q in range(n_qubits):

            # Handle one-body terms.
            coefficient = complex(iop[(p, 1), (q, 0)])
            if coefficient and p >= q:
                edge_matrix[p, q] = bool(complex(iop[(p, 1), (q, 0)]))

            # Keep looping for the two-body terms.
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coefficient2 = complex(iop[(p, 1), (q, 1), (r, 0), (s, 0)])

                    # Skip zero terms.
                    if (not coefficient2) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        elif p != r and q < p:
                            continue

                    # Handle case of four unique indices.
                    if len(set([p, q, r, s])) == 4:

                        if coefficient2 and p >= q:
                            edge_matrix[p, q] = bool(complex(
                                iop[(p, 1), (q, 1), (r, 0), (s, 0)]))
                            a, b = sorted([r, s])
                            edge_matrix[b, a] = bool(complex(
                                iop[(p, 1), (q, 1), (r, 0), (s, 0)]))

                    # Handle case of three unique indices.
                    elif len(set([p, q, r, s])) == 3:

                        # Identify equal tensor factors.
                        if p == r:
                            a, b = sorted([q, s])
                            edge_matrix[b, a] = bool(complex(
                                iop[(p, 1), (q, 1), (r, 0), (s, 0)]))
                        elif p == s:
                            a, b = sorted([q, r])
                            edge_matrix[b, a] = bool(complex(
                                iop[(p, 1), (q, 1), (r, 0), (s, 0)]))
                        elif q == r:
                            a, b = sorted([p, s])
                            edge_matrix[b, a] = bool(complex(
                                iop[(p, 1), (q, 1), (r, 0), (s, 0)]))
                        elif q == s:
                            a, b = sorted([p, r])
                            edge_matrix[b, a] = bool(complex(
                                iop[(p, 1), (q, 1), (r, 0), (s, 0)]))

    return edge_matrix.transpose()


def one_body(edge_matrix_indices, p, q):
    r"""
    Map the term a^\dagger_p a_q + a^\dagger_q a_p to QubitOperator.

    The definitions for various operators will be presented in a paper soon.

    Args:
        edge_matrix_indices(numpy array): Specifying the edges
        p and q (int): specifying the one body term.

    Return:
        An instance of QubitOperator()
    """
    # Handle off-diagonal terms.
    qubit_operator = QubitOperator()
    if p != q:
        a, b = sorted([p, q])
        B_a = edge_operator_b(edge_matrix_indices, a)
        B_b = edge_operator_b(edge_matrix_indices, b)
        A_ab = edge_operator_aij(edge_matrix_indices, a, b)
        qubit_operator += -1j / 2. * (A_ab * B_b + B_a * A_ab)

    # Handle diagonal terms.
    else:
        B_p = edge_operator_b(edge_matrix_indices, p)
        qubit_operator += (QubitOperator((), 1) - B_p) / 2.

    return qubit_operator


def two_body(edge_matrix_indices, p, q, r, s):
    r"""
    Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to QubitOperator.

    The definitions for various operators will be covered in a paper soon.

    Args:
        edge_matrix_indices (numpy array): specifying the edges
        p, q, r and s (int): specifying the two body term.

    Return:
        An instance of QubitOperator()
    """
    # Initialize qubit operator.
    qubit_operator = QubitOperator()

    # Handle case of four unique indices.
    if len(set([p, q, r, s])) == 4:
        B_p = edge_operator_b(edge_matrix_indices, p)
        B_q = edge_operator_b(edge_matrix_indices, q)
        B_r = edge_operator_b(edge_matrix_indices, r)
        B_s = edge_operator_b(edge_matrix_indices, s)
        A_pq = edge_operator_aij(edge_matrix_indices, p, q)
        A_rs = edge_operator_aij(edge_matrix_indices, r, s)
        qubit_operator += 1 / 8. * A_pq * A_rs * (
            -QubitOperator(()) - B_p*B_q + B_p*B_r + B_p*B_s + B_q*B_r +
            B_q*B_s - B_r*B_s + B_p*B_q*B_r*B_s)
        return qubit_operator

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:
        # Identify equal tensor factors.
        if p == r:
            B_p = edge_operator_b(edge_matrix_indices, p)
            B_q = edge_operator_b(edge_matrix_indices, q)
            B_s = edge_operator_b(edge_matrix_indices, s)
            A_qs = edge_operator_aij(edge_matrix_indices, q, s)
            qubit_operator += 1j * (A_qs*B_s + B_q*A_qs) * (
                QubitOperator(()) - B_p) / 4.

        elif p == s:
            B_p = edge_operator_b(edge_matrix_indices, p)
            B_q = edge_operator_b(edge_matrix_indices, q)
            B_r = edge_operator_b(edge_matrix_indices, r)
            A_qr = edge_operator_aij(edge_matrix_indices, q, r)
            qubit_operator += -1j * (A_qr*B_r + B_q*A_qr) * (
                QubitOperator(()) - B_p) / 4.

        elif q == r:
            B_p = edge_operator_b(edge_matrix_indices, p)
            B_q = edge_operator_b(edge_matrix_indices, q)
            B_s = edge_operator_b(edge_matrix_indices, s)
            A_ps = edge_operator_aij(edge_matrix_indices, p, s)
            qubit_operator += -1j * (A_ps*B_s + B_p*A_ps) * (
                QubitOperator(()) - B_q) / 4.

        elif q == s:
            B_p = edge_operator_b(edge_matrix_indices, p)
            B_q = edge_operator_b(edge_matrix_indices, q)
            B_r = edge_operator_b(edge_matrix_indices, r)
            A_pr = edge_operator_aij(edge_matrix_indices, p, r)
            qubit_operator += 1j * (A_pr*B_r + B_p*A_pr)*(
                QubitOperator(()) - B_q) / 4.

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:
        # Get coefficient.
        if p == s:
            B_p = edge_operator_b(edge_matrix_indices, p)
            B_q = edge_operator_b(edge_matrix_indices, q)
            qubit_operator += (QubitOperator(()) - B_p) * (
                QubitOperator(()) - B_q) / 4.

        else:
            B_p = edge_operator_b(edge_matrix_indices, p)
            B_q = edge_operator_b(edge_matrix_indices, q)
            qubit_operator += -(QubitOperator(()) - B_p) * (
                QubitOperator(()) - B_q) / 4.

    return qubit_operator


def edge_operator_b(edge_matrix_indices, i):
    """Calculate the edge operator B_i. The definitions used here are
    consistent with arXiv:quant-ph/0003137

    Args:
        edge_matrix_indices(numpy array(square and symmetric):
        i (int): index for specifying the edge operator B.

    Returns:
        An instance of QubitOperator
    """
    B_i = QubitOperator()
    qubit_position_matrix = numpy.array(numpy.where(edge_matrix_indices == i))
    qubit_position = qubit_position_matrix[1][:]
    qubit_position = numpy.sort(qubit_position)
    operator = tuple()
    for d1 in qubit_position:
        operator += ((int(d1), 'Z'),)
    B_i += QubitOperator(operator)
    return B_i


def edge_operator_aij(edge_matrix_indices, i, j):
    """Calculate the edge operator A_ij. The definitions used here are
    consistent with arXiv:quant-ph/0003137

    Args:
        edge_matrix_indices(numpy array): specifying the edges
        i,j (int): specifying the edge operator A

    Returns:
        An instance of QubitOperator
    """
    a_ij = QubitOperator()
    operator = tuple()
    position_ij = -1
    qubit_position_i = numpy.array(numpy.where(edge_matrix_indices == i))
    for edge_index in range(numpy.size(edge_matrix_indices[0, :])):
        if set((i, j)) == set(edge_matrix_indices[:, edge_index]):
            position_ij = edge_index
    operator += ((int(position_ij), 'X'),)

    for edge_index in range(numpy.size(qubit_position_i[0, :])):
        if edge_matrix_indices[int(not(qubit_position_i[0, edge_index]))][
                qubit_position_i[1, edge_index]] < j:
            operator += ((int(qubit_position_i[1, edge_index]), 'Z'),)
    qubit_position_j = numpy.array(numpy.where(edge_matrix_indices == j))
    for edge_index in range(numpy.size(qubit_position_j[0, :])):
        if edge_matrix_indices[int(not(qubit_position_j[0, edge_index]))][
                qubit_position_j[1, edge_index]] < i:
            operator += ((int(qubit_position_j[1, edge_index]), 'Z'),)
    a_ij += QubitOperator(operator, 1)
    if j < i:
        a_ij = -1*a_ij
    return a_ij


def vacuum_operator(edge_matrix_indices):
    """Use the stabilizers to find the vacuum state in bravyi_kitaev_fast.

    Args:
        edge_matrix_indices(numpy array): specifying the edges

    Return:
        An instance of QubitOperator

    """
    # Initialize qubit operator.
    g = networkx.Graph()
    g.add_edges_from(tuple(edge_matrix_indices.transpose()))
    stabs = numpy.array(networkx.cycle_basis(g))
    vac_operator = QubitOperator(())
    for stab in stabs:

        A = QubitOperator(())
        stab = numpy.array(stab)
        for i in range(numpy.size(stab)):
            if i == (numpy.size(stab) - 1):
                A = (1j)*A * edge_operator_aij(edge_matrix_indices,
                                               stab[i], stab[0])
            else:
                A = (1j)*A * edge_operator_aij(edge_matrix_indices,
                                               stab[i], stab[i+1])
        vac_operator = vac_operator*(QubitOperator(()) + A)/numpy.sqrt(2)

    return vac_operator


def number_operator(iop, mode_number=None):
    """Find the qubit operator for the number operator in bravyi_kitaev_fast
    representation

    Args:
        iop (InteractionOperator):
        mode_number: index mode_number corresponding to the mode
            for which number operator is required.

    Return:
        A QubitOperator

   """
    n_qubit = iop.n_qubits
    num_operator = QubitOperator()
    edge_matrix = bravyi_kitaev_fast_edge_matrix(iop)
    edge_matrix_indices = numpy.array(numpy.nonzero(numpy.triu(edge_matrix) -
                                      numpy.diag(numpy.diag(edge_matrix))))
    if mode_number is None:
        for i in range(n_qubit):
            num_operator += (QubitOperator(()) -
                             edge_operator_b(edge_matrix_indices, i))/2.

    else:
        num_operator += (QubitOperator(()) -
                         edge_operator_b(edge_matrix_indices, mode_number))/2.

    return num_operator


def generate_fermions(edge_matrix_indices, i, j):
    """The QubitOperator for generating fermions in bravyi_kitaev_fast
    representation

    Args:
        edge_matrix_indices(numpy array): specifying the edges

    Return:
        A QubitOperator
    """
    gen_fer_operator = (-1j/2.)*(edge_operator_aij(edge_matrix_indices, i, j) *
                                 edge_operator_b(edge_matrix_indices, j) -
                                 edge_operator_b(edge_matrix_indices, i) *
                                 edge_operator_aij(edge_matrix_indices, i, j))
    return gen_fer_operator
