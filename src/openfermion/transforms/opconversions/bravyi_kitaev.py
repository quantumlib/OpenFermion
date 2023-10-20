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
"""Bravyi-Kitaev transform on fermionic operators."""

from openfermion.ops.operators import FermionOperator, MajoranaOperator, QubitOperator
from openfermion.ops.representations import InteractionOperator
from openfermion.utils.operator_utils import count_qubits


def bravyi_kitaev(operator, n_qubits=None):
    """Apply the Bravyi-Kitaev transform.

    Implementation from arXiv:quant-ph/0003137 and
    "A New Data Structure for Cumulative Frequency Tables" by Peter M. Fenwick.

    Note that this implementation is equivalent to the one described in
    arXiv:1208.5986, and is different from the one described in
    arXiv:1701.07072. The one described in arXiv:1701.07072 is implemented
    in OpenFermion as `bravyi_kitaev_tree`.

    Args:
        operator (openfermion.ops.FermionOperator):
            A FermionOperator to transform.
        n_qubits (int|None):
            Can force the number of qubits in the resulting operator above the
            number that appear in the input operator.

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Raises:
        ValueError: Invalid number of qubits specified.
    """
    if isinstance(operator, FermionOperator):
        return _bravyi_kitaev_fermion_operator(operator, n_qubits)
    if isinstance(operator, MajoranaOperator):
        return _bravyi_kitaev_majorana_operator(operator, n_qubits)
    if isinstance(operator, InteractionOperator):
        return _bravyi_kitaev_interaction_operator(operator, n_qubits)
    raise TypeError(
        "Couldn't apply the Bravyi-Kitaev Transform to object " "of type {}.".format(type(operator))
    )


def _update_set(index, n_qubits):
    """The bits that need to be updated upon flipping the occupancy
    of a mode."""
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1
    # Ensure index is not a member of the set
    index += index & -index

    while index <= n_qubits:
        indices.add(index - 1)
        # Add least significant one to index
        # E.g. 00010100 -> 00011000
        index += index & -index
    return indices


def _occupation_set(index):
    """The bits whose parity stores the occupation of mode `index`."""
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    indices.add(index - 1)
    parent = index & (index - 1)
    index -= 1
    while index != parent:
        indices.add(index - 1)
        # Remove least significant one from index
        # E.g. 00010100 -> 00010000
        index &= index - 1
    return indices


def _parity_set(index):
    """The bits whose parity stores the parity of the bits 0 .. `index`."""
    indices = set()

    while index > 0:
        indices.add(index - 1)
        # Remove least significant one from index
        # E.g. 00010100 -> 00010000
        index &= index - 1
    return indices


def _bravyi_kitaev_majorana_operator(operator, n_qubits):
    # Compute the number of qubits.
    N = count_qubits(operator)
    if n_qubits is None:
        n_qubits = N
    if n_qubits < N:
        raise ValueError('Invalid number of qubits specified.')

    # Compute transformed operator.
    transformed_terms = (
        _transform_majorana_term(term=term, coefficient=coeff, n_qubits=n_qubits)
        for term, coeff in operator.terms.items()
    )
    return inline_sum(summands=transformed_terms, seed=QubitOperator())


def _transform_majorana_term(term, coefficient, n_qubits):
    # Build the Bravyi-Kitaev transformed operators.
    transformed_ops = (
        _transform_majorana_operator(majorana_index, n_qubits) for majorana_index in term
    )
    return inline_product(factors=transformed_ops, seed=QubitOperator((), coefficient))


def _transform_majorana_operator(majorana_index, n_qubits):
    q, b = divmod(majorana_index, 2)

    update_set = _update_set(q, n_qubits)
    update_set.add(q)
    occupation_set = _occupation_set(q)
    parity_set = _parity_set(q)

    if b:
        return QubitOperator(
            [(q, 'Y')]
            + [(i, 'X') for i in update_set - {q}]
            + [(i, 'Z') for i in (parity_set ^ occupation_set) - {q}]
        )
    else:
        return QubitOperator([(i, 'X') for i in update_set] + [(i, 'Z') for i in parity_set])


def _transform_operator_term(term, coefficient, n_qubits):
    """
    Args:
        term (list[tuple[int, int]]):
            A list of (mode, raising-vs-lowering) ladder operator terms.
        coefficient (float):
        n_qubits (int):
    Returns:
        QubitOperator:
    """

    # Build the Bravyi-Kitaev transformed operators.
    transformed_ladder_ops = (
        _transform_ladder_operator(ladder_operator, n_qubits) for ladder_operator in term
    )
    return inline_product(factors=transformed_ladder_ops, seed=QubitOperator((), coefficient))


def _bravyi_kitaev_fermion_operator(operator, n_qubits):
    # Compute the number of qubits.
    N = count_qubits(operator)
    if n_qubits is None:
        n_qubits = N
    if n_qubits < N:
        raise ValueError('Invalid number of qubits specified.')

    # Compute transformed operator.
    transformed_terms = (
        _transform_operator_term(term=term, coefficient=operator.terms[term], n_qubits=n_qubits)
        for term in operator.terms
    )
    return inline_sum(summands=transformed_terms, seed=QubitOperator())


def _transform_ladder_operator(ladder_operator, n_qubits):
    """
    Args:
        ladder_operator (tuple[int, int]): the ladder operator
        n_qubits (int): the number of qubits
    Returns:
        QubitOperator
    """
    index, action = ladder_operator

    update_set = _update_set(index, n_qubits)
    update_set.add(index)
    occupation_set = _occupation_set(index)
    parity_set = _parity_set(index)

    # Initialize the transformed majorana operator (a_p^\dagger + a_p) / 2
    transformed_operator = QubitOperator(
        [(i, 'X') for i in update_set] + [(i, 'Z') for i in parity_set], 0.5
    )
    # Get the transformed (a_p^\dagger - a_p) / 2
    # Below is equivalent to X(update_set) * Z(parity_set ^ occupation_set)
    transformed_majorana_difference = QubitOperator(
        [(index, 'Y')]
        + [(i, 'X') for i in update_set - {index}]
        + [(i, 'Z') for i in (parity_set ^ occupation_set) - {index}],
        -0.5j,
    )

    # Raising
    if action == 1:
        transformed_operator += transformed_majorana_difference
    # Lowering
    else:
        transformed_operator -= transformed_majorana_difference

    return transformed_operator


def inline_sum(summands, seed):
    """Computes a sum, using the __iadd__ operator.
    Args:
        seed (T): The starting total. The zero value.
        summands (iterable[T]): Values to add (with +=) into the total.
    Returns:
        T: The result of adding all the factors into the zero value.
    """
    for r in summands:
        seed += r
    return seed


def inline_product(factors, seed):
    """Computes a product, using the __imul__ operator.
    Args:
        seed (T): The starting total. The unit value.
        factors (iterable[T]): Values to multiply (with *=) into the total.
    Returns:
        T: The result of multiplying all the factors into the unit value.
    """
    for r in factors:
        seed *= r
    return seed


def _bravyi_kitaev_interaction_operator(interaction_operator, n_qubits):
    """Implementation of the Bravyi-Kitaev transformation for OpenFermion
    Interaction Operators. This implementation is equivalent to that described
    in arXiv:1208.5986, and has been written to optimize compute time by using
    algebraic expressions for general products a_i^dagger a_j^dagger as outlined
    in Table II of Seeley, Richard, Love.
    """

    one_body = interaction_operator.one_body_tensor
    two_body = interaction_operator.two_body_tensor
    constant_term = interaction_operator.constant

    # Compute the number of qubits.
    N = len(one_body)
    if n_qubits is None:
        n_qubits = N
    if n_qubits < N:
        raise ValueError('Invalid number of qubits specified.')

    qubit_hamiltonian = QubitOperator()
    qubit_hamiltonian_op = []
    qubit_hamiltonian_coef = []

    #  For cases A - F see Table II of Seeley, Richard, Love
    for i in range(n_qubits):
        # A. Number operators: n_i
        if abs(one_body[i, i]) > 0:
            qubit_hamiltonian += _qubit_operator_creation(
                *_seeley_richard_love(i, i, one_body[i, i], n_qubits)
            )

        for j in range(i):
            # Case B: Coulomb and exchange operators
            if abs(one_body[i, j]) > 0:
                operators, coef_list = _seeley_richard_love(i, j, one_body[i, j], n_qubits)
                qubit_hamiltonian_op.extend(operators)
                qubit_hamiltonian_coef.extend(coef_list)

                operators, coef_list = _seeley_richard_love(j, i, one_body[i, j].conj(), n_qubits)
                qubit_hamiltonian_op.extend(operators)
                qubit_hamiltonian_coef.extend(coef_list)

            coef = _two_body_coef(two_body, i, j, j, i) / 4
            if abs(coef) > 0:
                qubit_hamiltonian_op.append(tuple((index, "Z") for index in _occupation_set(i)))
                qubit_hamiltonian_op.append(tuple((index, "Z") for index in _occupation_set(j)))
                qubit_hamiltonian_op.append(tuple((index, "Z") for index in _F_ij_set(i, j)))

                qubit_hamiltonian_coef.append(-coef)
                qubit_hamiltonian_coef.append(-coef)
                qubit_hamiltonian_coef.append(coef)
                constant_term += coef

    # C. Number-excitation operators: n_i a_j^d a_k
    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(j):
                if i not in (j, k):
                    coef = _two_body_coef(two_body, i, j, k, i)

                    if abs(coef) > 0:
                        number = _qubit_operator_creation(*_seeley_richard_love(i, i, 1, n_qubits))

                        excitation_op, excitation_coef = _seeley_richard_love(j, k, coef, n_qubits)
                        operators_hc, coef_list_hc = _seeley_richard_love(
                            k, j, coef.conj(), n_qubits
                        )
                        excitation_op.extend(operators_hc)
                        excitation_coef.extend(coef_list_hc)
                        excitation = _qubit_operator_creation(excitation_op, excitation_coef)

                        number *= excitation
                        qubit_hamiltonian += number

    # D. Double-excitation operators: c_i^d c_j^d c_k c_l
    for i in range(n_qubits):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    coef = -_two_body_coef(two_body, i, j, k, l)
                    if abs(coef) > 0:
                        qubit_hamiltonian += _hermitian_one_body_product(i, j, k, l, coef, n_qubits)

                    coef = -_two_body_coef(two_body, i, k, j, l)
                    if abs(coef) > 0:
                        qubit_hamiltonian += _hermitian_one_body_product(i, k, j, l, coef, n_qubits)

                    coef = -_two_body_coef(two_body, i, l, j, k)
                    if abs(coef) > 0:
                        qubit_hamiltonian += _hermitian_one_body_product(i, l, j, k, coef, n_qubits)

    qubit_hamiltonian_op.append(())
    qubit_hamiltonian_coef.append(constant_term)
    qubit_hamiltonian += _qubit_operator_creation(qubit_hamiltonian_op, qubit_hamiltonian_coef)

    return qubit_hamiltonian


def _two_body_coef(two_body, a, b, c, d):
    return two_body[a, b, c, d] - two_body[a, b, d, c] + two_body[b, a, d, c] - two_body[b, a, c, d]


def _hermitian_one_body_product(a, b, c, d, coef, n_qubits):
    """Takes the 4 indices for a two-body operator and constructs the
    Bravyi-Kitaev form by splitting the two-body into 2 one-body operators,
    multiplying them together and then re-adding the Hermitian conjugate to
    give a Hermitian operator."""

    c_dag_c_ac = _qubit_operator_creation(*_seeley_richard_love(a, c, coef, n_qubits))
    c_dag_c_bd = _qubit_operator_creation(*_seeley_richard_love(b, d, 1, n_qubits))
    c_dag_c_ac *= c_dag_c_bd
    hermitian_sum = c_dag_c_ac

    c_dag_c_ca = _qubit_operator_creation(*_seeley_richard_love(c, a, coef.conj(), n_qubits))
    c_dag_c_db = _qubit_operator_creation(*_seeley_richard_love(d, b, 1, n_qubits))
    c_dag_c_ca *= c_dag_c_db

    hermitian_sum += c_dag_c_ca
    return hermitian_sum


def _qubit_operator_creation(operators, coefficents):
    """Takes a list of tuples for operators/indices, and another for
    coefficents"""

    qubit_operator = QubitOperator()

    for index in zip(operators, coefficents):
        qubit_operator += QubitOperator(index[0], index[1])

    return qubit_operator


def _seeley_richard_love(i, j, coef, n_qubits):
    """Algebraic expressions for general products of the form a_i^d a_j term in
    the Bravyi-Kitaev basis. These expressions vary in form depending on the
    parity of the indices i and j, as well as onthe overlaps between the parity
    and update sets of the indices"""

    seeley_richard_love_op = []
    seeley_richard_love_coef = []
    coef *= 0.25
    # Case 0
    if i == j:  # Simplifies to the number operator
        seeley_richard_love_op.append(tuple((index, "Z") for index in _occupation_set(i)))
        seeley_richard_love_coef.append(-coef * 2)

        seeley_richard_love_op.append(())
        seeley_richard_love_coef.append(coef * 2)

    # Case 1
    elif i % 2 == 0 and j % 2 == 0:
        x_pad = tuple((index, "X") for index in _U_diff_a_set(i, j, n_qubits))
        y_pad = tuple((index, "Y") for index in _alpha_set(i, j, n_qubits))
        z_pad = tuple((index, "Z") for index in _P0_ij_diff_a_set(i, j, n_qubits))

        left_pad = x_pad + y_pad + z_pad

        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")))
        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")))
        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")))
        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")))

        if i < j:
            seeley_richard_love_coef.append(coef)
            seeley_richard_love_coef.append(-coef)
            seeley_richard_love_coef.append(complex(0, -coef))
            seeley_richard_love_coef.append(complex(0, -coef))

        else:  # whenever i < j introduce phase of -j
            seeley_richard_love_coef.append(complex(0, -coef))
            seeley_richard_love_coef.append(complex(0, coef))
            seeley_richard_love_coef.append(-coef)
            seeley_richard_love_coef.append(-coef)

    # Case 2
    elif i % 2 == 1 and j % 2 == 0 and i not in _parity_set(j):
        x_pad = tuple((index, "X") for index in _U_diff_a_set(i, j, n_qubits))
        y_pad = tuple((index, "Y") for index in _alpha_set(i, j, n_qubits))

        left_pad = x_pad + y_pad

        right_pad_1 = tuple((index, "Z") for index in _P0_ij_set(i, j) - _alpha_set(i, j, n_qubits))
        right_pad_2 = tuple((index, "Z") for index in _P2_ij_set(i, j) - _alpha_set(i, j, n_qubits))

        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")) + right_pad_1)
        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")) + right_pad_1)
        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")) + right_pad_2)
        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")) + right_pad_2)

        if i < j:
            seeley_richard_love_coef.append(coef)
            seeley_richard_love_coef.append(complex(0, -coef))
            seeley_richard_love_coef.append(-coef)
            seeley_richard_love_coef.append(complex(0, -coef))

        else:
            seeley_richard_love_coef.append(complex(0, -coef))
            seeley_richard_love_coef.append(-coef)
            seeley_richard_love_coef.append(complex(0, coef))
            seeley_richard_love_coef.append(-coef)

    # Case 3
    elif i % 2 == 1 and j % 2 == 0 and i in _parity_set(j):
        left_pad = tuple((index, "X") for index in _U_ij_set(i, j, n_qubits))
        right_pad_1 = tuple((index, "Z") for index in _P0_ij_set(i, j) - {i})
        right_pad_2 = tuple((index, "Z") for index in _P2_ij_set(i, j) - {i})

        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")) + right_pad_1)
        seeley_richard_love_coef.append(coef)

        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")) + right_pad_1)
        seeley_richard_love_coef.append(complex(0, -coef))

        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")) + right_pad_2)
        seeley_richard_love_coef.append(coef)

        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")) + right_pad_2)
        seeley_richard_love_coef.append(complex(0, coef))

    # Case 4
    elif (
        i % 2 == 0 and j % 2 == 1 and i not in _parity_set(j) and j not in _update_set(i, n_qubits)
    ):
        x_pad = tuple((index, "X") for index in _U_diff_a_set(i, j, n_qubits))
        y_pad = tuple((index, "Y") for index in _alpha_set(i, j, n_qubits))
        left_pad = x_pad + y_pad

        right_pad_1 = tuple((index, "Z") for index in _P0_ij_set(i, j) - _alpha_set(i, j, n_qubits))
        right_pad_2 = tuple((index, "Z") for index in _P1_ij_set(i, j) - _alpha_set(i, j, n_qubits))

        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")) + right_pad_1)
        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")) + right_pad_1)
        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")) + right_pad_2)
        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")) + right_pad_2)

        if i < j:
            seeley_richard_love_coef.append(-coef)
            seeley_richard_love_coef.append(complex(0, -coef))
            seeley_richard_love_coef.append(coef)
            seeley_richard_love_coef.append(complex(0, -coef))
        else:
            seeley_richard_love_coef.append(complex(0, coef))
            seeley_richard_love_coef.append(-coef)
            seeley_richard_love_coef.append(complex(0, -coef))
            seeley_richard_love_coef.append(-coef)

    # Case 5
    elif i % 2 == 0 and j % 2 == 1 and i not in _parity_set(j) and j in _update_set(i, n_qubits):
        x_range_1 = _U_ij_set(i, j, n_qubits) - {j}
        left_pad_1 = tuple((index, "X") for index in x_range_1)

        x_range_2 = x_range_1 - _alpha_set(i, j, n_qubits)
        left_pad_2 = tuple((index, "X") for index in x_range_2)

        y_pad = tuple((index, "Y") for index in _alpha_set(i, j, n_qubits))
        z_range_1 = _P0_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        z_pad = tuple((index, "Z") for index in z_range_1)
        right_pad_1 = y_pad + z_pad

        z_range_2 = _P1_ij_set(i, j).union({j})
        right_pad_2 = tuple((index, "Z") for index in z_range_2)

        seeley_richard_love_op.append(left_pad_2 + ((i, "Y"),) + right_pad_1)
        seeley_richard_love_coef.append(-coef)

        seeley_richard_love_op.append(left_pad_2 + ((i, "X"),) + right_pad_1)
        seeley_richard_love_coef.append(
            complex(0, -coef)
        )  # Phase flip of -1 relative to original paper

        seeley_richard_love_op.append(left_pad_1 + ((i, "Y"),) + right_pad_2)
        seeley_richard_love_coef.append(complex(0, coef))

        seeley_richard_love_op.append(left_pad_1 + ((i, "X"),) + right_pad_2)
        seeley_richard_love_coef.append(-coef)

    # Case 6
    elif i % 2 == 0 and j % 2 == 1 and i in _parity_set(j) and j in _update_set(i, n_qubits):
        left_pad = tuple((index, "X") for index in _U_ij_set(i, j, n_qubits) - {j})
        right_pad = tuple((index, "Z") for index in _P1_ij_set(i, j).union({j}))

        seeley_richard_love_op.append(left_pad + ((i, "X"),))
        seeley_richard_love_coef.append(coef)

        seeley_richard_love_op.append(left_pad + ((i, "Y"),))
        seeley_richard_love_coef.append(complex(0, -coef))

        seeley_richard_love_op.append(left_pad + ((i, "Y"),) + right_pad)
        seeley_richard_love_coef.append(complex(0, coef))

        seeley_richard_love_op.append(left_pad + ((i, "X"),) + right_pad)
        seeley_richard_love_coef.append(-coef)

    # Case 7
    elif (
        i % 2 == 1 and j % 2 == 1 and i not in _parity_set(j) and j not in _update_set(i, n_qubits)
    ):
        x_pad = tuple((index, "X") for index in _U_diff_a_set(i, j, n_qubits))
        y_pad = tuple((index, "Y") for index in _alpha_set(i, j, n_qubits))
        left_pad = x_pad + y_pad

        z_range_1 = _P0_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        right_pad_1 = tuple((index, "Z") for index in z_range_1)

        z_range_2 = _P1_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        right_pad_2 = tuple((index, "Z") for index in z_range_2)

        z_range_3 = _P2_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        right_pad_3 = tuple((index, "Z") for index in z_range_3)

        z_range_4 = _P3_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        right_pad_4 = tuple((index, "Z") for index in z_range_4)

        if i < j:
            seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")) + right_pad_1)
            seeley_richard_love_coef.append(complex(0, -coef))

            seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")) + right_pad_2)
            seeley_richard_love_coef.append(coef)

            seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")) + right_pad_3)
            seeley_richard_love_coef.append(-coef)

            seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")) + right_pad_4)
            seeley_richard_love_coef.append(complex(0, -coef))

        else:
            seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")) + right_pad_1)
            seeley_richard_love_coef.append(-coef)

            seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")) + right_pad_2)
            seeley_richard_love_coef.append(complex(0, -coef))

            seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")) + right_pad_3)
            seeley_richard_love_coef.append(complex(0, coef))

            seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")) + right_pad_4)
            seeley_richard_love_coef.append(-coef)

    # Case 8
    elif i % 2 == 1 and j % 2 == 1 and i in _parity_set(j) and j not in _update_set(i, n_qubits):
        left_pad = tuple((index, "X") for index in _U_ij_set(i, j, n_qubits))

        z_range_1 = _P0_ij_set(i, j) - {i}
        right_pad_1 = tuple((index, "Z") for index in z_range_1)

        z_range_2 = _P1_ij_set(i, j) - {i}
        right_pad_2 = tuple((index, "Z") for index in z_range_2)

        z_range_3 = _P2_ij_set(i, j) - {i}
        right_pad_3 = tuple((index, "Z") for index in z_range_3)

        z_range_4 = _P3_ij_set(i, j) - {i}
        right_pad_4 = tuple((index, "Z") for index in z_range_4)

        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "Y")) + right_pad_1)
        seeley_richard_love_coef.append(complex(0, -coef))

        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "Y")) + right_pad_2)
        seeley_richard_love_coef.append(coef)

        seeley_richard_love_op.append(left_pad + ((j, "X"), (i, "X")) + right_pad_3)
        seeley_richard_love_coef.append(coef)

        seeley_richard_love_op.append(left_pad + ((j, "Y"), (i, "X")) + right_pad_4)
        seeley_richard_love_coef.append(complex(0, coef))

    # Case 9
    elif i % 2 == 1 and j % 2 == 1 and i not in _parity_set(j) and j in _update_set(i, n_qubits):
        x_range_1 = _U_ij_set(i, j, n_qubits) - {j}
        left_pad_3 = tuple((index, "X") for index in x_range_1)

        x_range_2 = x_range_1 - _alpha_set(i, j, n_qubits)
        left_pad_1 = tuple((index, "X") for index in x_range_2)

        x_range_3 = x_range_1.union({i}) - _alpha_set(i, j, n_qubits)
        left_pad_2 = tuple((index, "X") for index in x_range_3)

        z_range_1 = _P2_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        z_pad_1 = tuple((index, "Z") for index in z_range_1)
        y_range = _alpha_set(i, j, n_qubits)
        y_pad = tuple((index, "Y") for index in y_range)
        right_pad_1 = z_pad_1 + y_pad

        z_range_2 = _P0_ij_set(i, j) - _alpha_set(i, j, n_qubits)
        z_pad_2 = tuple((index, "Z") for index in z_range_2)
        right_pad_2 = z_pad_2 + y_pad

        z_range_3 = _P1_ij_set(i, j).union({j})
        right_pad_3 = tuple((index, "Z") for index in z_range_3)

        z_range_4 = _P3_ij_set(i, j).union({j})
        right_pad_4 = tuple((index, "Z") for index in z_range_4)

        seeley_richard_love_op.append(left_pad_1 + ((i, "Y"),) + right_pad_1)
        seeley_richard_love_coef.append(-coef)  # phase of -j relative to original paper

        seeley_richard_love_op.append(left_pad_2 + right_pad_2)
        seeley_richard_love_coef.append(complex(0, -coef))  # phase of -j relative to original paper

        seeley_richard_love_op.append(left_pad_3 + ((i, "X"),) + right_pad_3)
        seeley_richard_love_coef.append(-coef)

        seeley_richard_love_op.append(left_pad_3 + ((i, "Y"),) + right_pad_4)
        seeley_richard_love_coef.append(complex(0, coef))

    # Case 10
    elif i % 2 == 1 and j % 2 == 1 and i in _parity_set(j) and j in _update_set(i, n_qubits):
        left_pad = tuple((index, "X") for index in _U_ij_set(i, j, n_qubits) - {j})
        right_pad_1 = tuple((index, "Z") for index in _P0_ij_set(i, j) - {i})
        right_pad_2 = tuple((index, "Z") for index in _P2_ij_set(i, j) - {i})
        right_pad_3 = tuple((index, "Z") for index in _P1_ij_set(i, j))
        right_pad_4 = tuple((index, "Z") for index in _P3_ij_set(i, j))

        seeley_richard_love_op.append(left_pad + ((i, "Y"),) + right_pad_1)
        seeley_richard_love_coef.append(complex(0, -coef))

        seeley_richard_love_op.append(left_pad + ((i, "X"),) + right_pad_2)
        seeley_richard_love_coef.append(coef)

        seeley_richard_love_op.append(left_pad + ((j, "Z"), (i, "X")) + right_pad_3)
        seeley_richard_love_coef.append(-coef)

        seeley_richard_love_op.append(left_pad + ((j, "Z"), (i, "Y")) + right_pad_4)
        seeley_richard_love_coef.append(complex(0, coef))

    return seeley_richard_love_op, seeley_richard_love_coef


def _remainder_set(index):
    return _parity_set(index) - _occupation_set(index)


def _F_ij_set(i, j):
    return _occupation_set(i).symmetric_difference(_occupation_set(j))


def _P0_ij_set(i, j):
    """The symmetric difference of sets P(i) and P(j)."""
    return _parity_set(i).symmetric_difference(_parity_set(j))


def _P1_ij_set(i, j):
    """The symmetric difference of sets P(i) and R(j)."""
    return _parity_set(i).symmetric_difference(_remainder_set(j))


def _P2_ij_set(i, j):
    """The symmetric difference of sets R(i) and P(j)."""
    return _remainder_set(i).symmetric_difference(_parity_set(j))


def _P3_ij_set(i, j):
    """The symmetric difference of sets R(i) and R(j)."""
    return _remainder_set(i).symmetric_difference(_remainder_set(j))


def _U_ij_set(i, j, n_qubits):
    """The symmetric difference of sets U(i) and U(j)"""
    return _update_set(i, n_qubits).symmetric_difference(_update_set(j, n_qubits))


def _U_diff_a_set(i, j, n_qubits):
    """Calculates the set {member U_ij diff alpha_ij}."""
    return _U_ij_set(i, j, n_qubits) - _alpha_set(i, j, n_qubits)


def _P0_ij_diff_a_set(i, j, n_qubits):
    """Calculates the set {member P_ij diff alpha_ij}."""
    return _P0_ij_set(i, j) - _alpha_set(i, j, n_qubits)


def _alpha_set(i, j, n_qubits):
    return _update_set(i, n_qubits).intersection(_parity_set(j))
