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

from openfermion.ops import FermionOperator, MajoranaOperator, QubitOperator
from openfermion.utils import count_qubits, inline_sum


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
    raise TypeError("Couldn't apply the Bravyi-Kitaev Transform to object "
                    "of type {}.".format(type(operator)))


def _update_set(index, n_qubits):
    """The bits that need to be updated upon flipping the occupancy
    of a mode."""
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

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

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

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
        _transform_majorana_term(term=term,
                                 coefficient=coeff,
                                 n_qubits=n_qubits)
        for term, coeff in operator.terms.items()
    )
    return inline_sum(summands=transformed_terms, seed=QubitOperator())


def _transform_majorana_term(term, coefficient, n_qubits):
    # Build the Bravyi-Kitaev transformed operators.
    transformed_ops = (
        _transform_majorana_operator(majorana_index, n_qubits)
        for majorana_index in term
    )
    return inline_product(factors=transformed_ops,
                          seed=QubitOperator((), coefficient))


def _transform_majorana_operator(majorana_index, n_qubits):
    q, b = divmod(majorana_index, 2)

    update_set = _update_set(q, n_qubits)
    occupation_set = _occupation_set(q)
    parity_set = _parity_set(q - 1)

    if b:
        return QubitOperator(
                [(q, 'Y')] +
                [(i, 'X') for i in update_set - {q}] +
                [(i, 'Z') for i in (parity_set ^ occupation_set) - {q}])
    else:
        return QubitOperator(
                [(i, 'X') for i in update_set] +
                [(i, 'Z') for i in parity_set])


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
        _transform_ladder_operator(ladder_operator, n_qubits)
        for ladder_operator in term
    )
    return inline_product(factors=transformed_ladder_ops,
                          seed=QubitOperator((), coefficient))


def _bravyi_kitaev_fermion_operator(operator, n_qubits):
    # Compute the number of qubits.
    N = count_qubits(operator)
    if n_qubits is None:
        n_qubits = N
    if n_qubits < N:
        raise ValueError('Invalid number of qubits specified.')

    # Compute transformed operator.
    transformed_terms = (
        _transform_operator_term(term=term,
                                 coefficient=operator.terms[term],
                                 n_qubits=n_qubits)
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
    occupation_set = _occupation_set(index)
    parity_set = _parity_set(index - 1)

    # Initialize the transformed majorana operator (a_p^\dagger + a_p) / 2
    transformed_operator = QubitOperator(
            [(i, 'X') for i in update_set] +
            [(i, 'Z') for i in parity_set],
            .5)
    # Get the transformed (a_p^\dagger - a_p) / 2
    # Below is equivalent to X(update_set) * Z(parity_set ^ occupation_set)
    transformed_majorana_difference = QubitOperator(
            [(index, 'Y')] +
            [(i, 'X') for i in update_set - {index}] +
            [(i, 'Z') for i in (parity_set ^ occupation_set) - {index}],
            -.5j)

    # Raising
    if action == 1:
        transformed_operator += transformed_majorana_difference
    # Lowering
    else:
        transformed_operator -= transformed_majorana_difference

    return transformed_operator


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
