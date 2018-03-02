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

"""Verstraete-Cirac transform on fermionic operators."""
from __future__ import absolute_import

import itertools
import networkx
import numpy

from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from openfermion.utils import majorana_operator 


def verstraete_cirac_2d_square(operator, x_dimension, y_dimension,
                               add_auxiliary_hamiltonian=True,
                               snake=False):
    """Apply the Verstraete-Cirac transform on a 2-d square lattice.

    Note that this transformation adds one auxiliary fermionic mode
    for each mode already present, and hence it doubles the number of qubits
    needed to represent the system.

    Currently only supports even values of x_dimension and only works
    for spinless models.

    Args:
        operator (FermionOperator): The operator to transform.
        x_dimension (int): The number of columns of the grid.
        y_dimension (int): The number of rows of the grid.
        snake (bool, optional): Indicates whether the fermions are already
            ordered according to the 2-d "snake" ordering. If False,
            we assume they are in "lexicographic" order by row and column
            index. Default is False.

    Returns:
        transformed_operator: A QubitOperator.
    """
    if x_dimension % 2 != 0:
        raise NotImplementedError('Currently only even x_dimension '
                                  'is supported.')

    # Obtain the vertical edges of the snake ordering
    vert_edges = vertical_edges_snake(x_dimension, y_dimension)

    # Initialize a coefficient to scale the auxiliary Hamiltonian by.
    # The gap of the auxiliary Hamiltonian needs to be large enough to
    # to ensure the ground state of the original operator is preserved.
    aux_ham_coefficient = 1.

    transformed_operator = QubitOperator()
    for term in operator.terms:

        indices = [ladder_operator[0] for ladder_operator in term]
        raise_or_lower = [ladder_operator[1] for ladder_operator in term]
        coefficient = operator.terms[term]

        # If the indices aren't in snake order, we need to convert them
        if not snake:
            indices = [lexicographic_index_to_snake_index(
                index, x_dimension, y_dimension) for index in indices]

        # Convert the indices to indices of system qubits in the combined
        # system, which includes the auxiliary qubits interleaved
        transformed_indices = [expand_sys_index(index) for index in indices]

        # Initialize the transformed term as a FermionOperator
        transformed_term = FermionOperator(
            tuple(zip(transformed_indices, raise_or_lower)), coefficient)

        # If necessary, multiply the transformed term by a stabilizer to
        # cancel out Jordan-Wigner strings
        if len(indices) == 2:
            # Term is quadratic
            i, j = indices[0], indices[1]
            if (i, j) in vert_edges or (j, i) in vert_edges:
                # Term corresponds to a vertical edge, so we need to multiply
                # by a stabilizer
                top = min(i, j)
                bot = max(i, j)

                # Get the indices of the corresponding auxiliary qubits
                top_aux = expand_aux_index(top)
                bot_aux = expand_aux_index(bot)

                # Get the column that this edge is on
                col, row = snake_index_to_coordinates(
                    top, x_dimension, y_dimension)

                # Multiply by a stabilizer. If the column is even, the
                # stabilizer corresponds to an edge that points down;
                # otherwise, the edge points up
                if col % 2 == 0:
                    transformed_term *= stabilizer(top_aux, bot_aux)
                else:
                    transformed_term *= stabilizer(bot_aux, top_aux)

                # Update the auxiliary Hamiltonian coefficient
                aux_ham_coefficient += abs(coefficient)

        # Transform the term to a QubitOperator and add it to the operator
        transformed_operator += jordan_wigner(transformed_term)

    # Add the auxiliary Hamiltonian if requested and compute the
    # resulting energy shift
    if add_auxiliary_hamiltonian:
        # Construct the auxiliary Hamiltonian graph
        aux_ham_graph = auxiliary_graph_2d_square(x_dimension, y_dimension)

        # Construct the auxiliary Hamiltonian
        aux_ham = FermionOperator()
        for i, j in aux_ham_graph.edges():
            aux_ham -= stabilizer_local_2d_square(
                i, j, x_dimension, y_dimension)

        # Add an identity term to ensure that the auxiliary Hamiltonian
        # has ground energy equal to zero
        aux_ham += FermionOperator((), aux_ham_graph.size())

        # Scale the auxiliary Hamiltonian
        aux_ham *= aux_ham_coefficient

        # Add it to the operator
        transformed_operator += jordan_wigner(aux_ham)

    return transformed_operator


def stabilizer(i, j):
    """Stabilizer operators which act on the auxiliary space.
    In the original paper, these are referred to as P_{ij}."""
    c_i = majorana_operator((i, 1), numpy.sqrt(2.))
    d_j = majorana_operator((j, 0), numpy.sqrt(2.))
    return 1.j * c_i * d_j


def stabilizer_local_2d_square(i, j, x_dimension, y_dimension):
    """The local version of the stabilizers for a 2-d grid.

    i and j are indices on the auxiliary graph.
    Currently this only works for even x_dimension.
    """
    i_col, i_row = snake_index_to_coordinates(i, x_dimension, y_dimension)
    j_col, j_row = snake_index_to_coordinates(j, x_dimension, y_dimension)
    if not (abs(i_row - j_row) == 1 and i_col == j_col or
            abs(i_col - j_col) == 1 and i_row == j_row):
        raise ValueError("Vertices i and j are not adjacent")

    # Get the JWT indices in the combined system
    i_expanded = expand_aux_index(i)
    j_expanded = expand_aux_index(j)

    stab = stabilizer(i_expanded, j_expanded)
    if abs(i_row - j_row) == 1:
        # Term is vertical, so we may need to multiply by extra stabilizers
        top_row = min(i_row, j_row)
        if top_row % 2 == 0:
            # Term is right-closed
            if i_col < x_dimension - 1:
                extra_term_top = expand_aux_index(coordinates_to_snake_index(
                    i_col + 1, top_row, x_dimension, y_dimension))
                extra_term_bot = expand_aux_index(coordinates_to_snake_index(
                    i_col + 1, top_row + 1, x_dimension, y_dimension))
                if (i_col + 1) % 2 == 0:
                    stab *= stabilizer(extra_term_top, extra_term_bot)
                else:
                    stab *= stabilizer(extra_term_bot, extra_term_top)
        else:
            # Term is left-closed
            if i_col > 0:
                extra_term_top = expand_aux_index(coordinates_to_snake_index(
                    i_col - 1, top_row, x_dimension, y_dimension))
                extra_term_bot = expand_aux_index(coordinates_to_snake_index(
                    i_col - 1, top_row + 1, x_dimension, y_dimension))
                if (i_col - 1) % 2 == 0:
                    stab *= stabilizer(extra_term_top, extra_term_bot)
                else:
                    stab *= stabilizer(extra_term_bot, extra_term_top)

    return stab


def auxiliary_graph_2d_square(x_dimension, y_dimension):
    """Obtain the auxiliary graph for a 2-d grid.

    Currently this only works for even x_dimension.
    """
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(x_dimension * y_dimension))

    for k in range(0, x_dimension, 2):
        # Create the loop spanning columns k and k + 1
        # Add top edge
        graph.add_edge(k + 1, k)
        # Add bottom edge
        graph.add_edge(coordinates_to_snake_index(k, y_dimension - 1,
                                                  x_dimension, y_dimension),
                       coordinates_to_snake_index(k + 1, y_dimension - 1,
                                                  x_dimension, y_dimension))
        for l in range(y_dimension - 1):
            # Add edges between rows l and l + 1
            # Add left edge
            graph.add_edge(
                coordinates_to_snake_index(k, l, x_dimension, y_dimension),
                coordinates_to_snake_index(k, l + 1,
                                           x_dimension, y_dimension))
            # Add right edge
            graph.add_edge(
                coordinates_to_snake_index(k + 1, l + 1,
                                           x_dimension, y_dimension),
                coordinates_to_snake_index(k + 1, l,
                                           x_dimension, y_dimension))

    return graph


def coordinates_to_snake_index(column, row, x_dimension, y_dimension):
    """Obtain the index in the snake ordering of a coordinate on a 2-d grid."""
    if column > x_dimension - 1:
        raise ValueError('Column index exceeds x_dimension - 1.')
    if row > y_dimension - 1:
        raise ValueError('Row index exceeds y_dimension - 1.')

    if row % 2 == 0:
        index = row * x_dimension + column
    else:
        index = (row + 1) * x_dimension - 1 - column

    return index


def snake_index_to_coordinates(index, x_dimension, y_dimension):
    """Obtain the column and row coordinates corresponding to a snake ordering
    index on a 2-d grid.
    """
    if index > x_dimension * y_dimension - 1:
        raise ValueError('Index exceeds x_dimension * y_dimension - 1.')

    row = index // x_dimension
    if row % 2 == 0:
        column = index % x_dimension
    else:
        column = x_dimension - 1 - index % x_dimension

    return column, row


def lexicographic_index_to_snake_index(index, x_dimension, y_dimension):
    """Convert an index from lexicographic (row, col) order to snake order."""
    row = index // x_dimension
    col = index % x_dimension
    snake_index = coordinates_to_snake_index(col, row,
                                             x_dimension, y_dimension)
    return snake_index


def expand_sys_index(index):
    """Convert the index of a system fermion to the combined system."""
    return 2 * index


def expand_aux_index(index):
    """Convert the index of a system fermion to the combined system."""
    return 2 * index + 1


def row_indices_snake(row, x_dimension):
    """Obtain the indices in a row from left to right in the
    2-d snake ordering."""
    indices = range(row * x_dimension, (row + 1) * x_dimension)
    if row % 2 != 0:
        indices = reversed(indices)
    return list(indices)


def vertical_edges_snake(x_dimension, y_dimension):
    """Obtain the vertical edges in the 2-d snake ordering."""
    edges = []
    for row in range(y_dimension - 1):
        upper_row = row_indices_snake(row, x_dimension)
        lower_row = row_indices_snake(row + 1, x_dimension)
        edges += zip(upper_row, lower_row)
    return edges
