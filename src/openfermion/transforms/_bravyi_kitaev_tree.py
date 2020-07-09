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
from openfermion.utils import inline_sum
from openfermion.transforms._bravyi_kitaev import inline_product
from openfermion.transforms._fenwick_tree import FenwickTree


def bravyi_kitaev_tree(operator, n_qubits=None):
    """Apply the "tree" Bravyi-Kitaev transform.

    Implementation from arxiv:1701.07072

    Note that this implementation is different from the one described in
    arXiv:quant-ph/0003137. In particular, it gives different results
    when the total number of modes is not a power of 2. The one described
    in arXiv:quant-ph/0003137 is the same as the one described in
    arXiv:1208.5986, and it is implemented in OpenFermion under the name
    `bravyi_kitaev`.

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

    # Build the Fenwick tree.
    fenwick_tree = FenwickTree(n_qubits)

    # Compute transformed operator.
    transformed_terms = (
        _transform_operator_term(term=term,
                                 coefficient=operator.terms[term],
                                 fenwick_tree=fenwick_tree)
        for term in operator.terms
    )
    return inline_sum(summands=transformed_terms, seed=QubitOperator())


def _transform_operator_term(term, coefficient, fenwick_tree):
    """
    Args:
        term (list[tuple[int, int]]):
            A list of (mode, raising-vs-lowering) ladder operator terms.
        coefficient (float):
        fenwick_tree (FenwickTree):
    Returns:
        QubitOperator:
    """

    # Build the Bravyi-Kitaev transformed operators.
    transformed_ladder_ops = (
        _transform_ladder_operator(ladder_operator, fenwick_tree)
        for ladder_operator in term
    )
    return inline_product(factors=transformed_ladder_ops,
                          seed=QubitOperator((), coefficient))


def _transform_ladder_operator(ladder_operator, fenwick_tree):
    """
    Args:
        ladder_operator (tuple[int, int]):
        fenwick_tree (FenwickTree):
    Returns:
        QubitOperator:
    """
    index = ladder_operator[0]

    # Parity set. Set of nodes to apply Z to.
    parity_set = [node.index for node in
                  fenwick_tree.get_parity_set(index)]

    # Update set. Set of ancestors to apply X to.
    ancestors = [node.index for node in
                 fenwick_tree.get_update_set(index)]

    # The C(j) set.
    ancestor_children = [node.index for node in
                         fenwick_tree.get_remainder_set(index)]

    # Switch between lowering/raising operators.
    d_coefficient = -.5j if ladder_operator[1] else .5j

    # The fermion lowering operator is given by
    # a = (c+id)/2 where c, d are the majoranas.
    d_majorana_component = QubitOperator(
        (((ladder_operator[0], 'Y'),) +
         tuple((index, 'Z') for index in ancestor_children) +
         tuple((index, 'X') for index in ancestors)),
        d_coefficient)

    c_majorana_component = QubitOperator(
        (((ladder_operator[0], 'X'),) +
         tuple((index, 'Z') for index in parity_set) +
         tuple((index, 'X') for index in ancestors)),
        0.5)

    return c_majorana_component + d_majorana_component
