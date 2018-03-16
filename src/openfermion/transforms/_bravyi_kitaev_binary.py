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

from openfermion.ops import QubitOperator


def bravyi_kitaev(operator, n_qubits=None):
    """Apply the Bravyi-Kitaev transform and return qubit operator.

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
    # Compute the number of qubits.
    from openfermion.utils import count_qubits
    if n_qubits is None:
        n_qubits = count_qubits(operator)
    if n_qubits < count_qubits(operator):
        raise ValueError('Invalid number of qubits specified.')


    # Compute transformed operator.
    transformed_terms = (
        _transform_operator_term(term=term,
                                 coefficient=operator.terms[term],
                                 n_qubits=n_qubits)
        for term in operator.terms
    )
    return inline_sum(seed=QubitOperator(), summands=transformed_terms)


def update_set(index, n_qubits):
    indices = set()
    index += 1

    while index <= n_qubits:
        indices.add(index - 1)
        # Add least significant one to index
        index += index & -index
    return indices


def occupation_set(index):
    indices = set()
    index += 1

    indices.add(index - 1)
    parent = index & (index - 1)
    index -= 1
    while index != parent:
        indices.add(index - 1)
        index &= index - 1
    return indices


def parity_set(index):
    indices = set()
    index += 1

    while index > 0:
        indices.add(index - 1)
        # Remove least significant one from index
        index &= index - 1
    return indices


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
    return inline_product(seed=QubitOperator((), coefficient),
                          factors=transformed_ladder_ops)


def _transform_ladder_operator(ladder_operator, n_qubits):
    """
    Args:
        ladder_operator (tuple[int, int]): the ladder operator
        n_qubits (int): the number of qubits
    Returns:
        QubitOperator
    """
    index, action = ladder_operator

    update_set_ = update_set(index, n_qubits)
    occupation_set_ = occupation_set(index)
    parity_set_ = parity_set(index - 1)

    # The transformed (a_p^\dagger + a_p) / 2
    transformed_majorana_sum = (
            QubitOperator([(index, 'X') for index in update_set_], .5) *
            QubitOperator([(index, 'Z') for index in parity_set_]))
    # The transformed (a_p^\dagger - a_p) / 2
    transformed_majorana_difference = (
            QubitOperator([(index, 'X') for index in update_set_], .5) *
            QubitOperator([(index, 'Z') for index in
                           parity_set_ ^ occupation_set_]))

    # raising
    if action == 1:
        return transformed_majorana_sum + transformed_majorana_difference
    # lowering
    else:
        return transformed_majorana_sum - transformed_majorana_difference


def inline_sum(seed, summands):
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


def inline_product(seed, factors):
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
