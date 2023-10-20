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
""" Code to generate Pauli strings for measurement of local operators"""
from itertools import chain, zip_longest
import numpy

from openfermion.ops.operators import QubitOperator


def binary_partition_iterator(qubit_list, num_iterations=None):
    """Generator for a list of 2-partitions of N qubits
    such that all pairs of qubits are split in at least one partition,
    This follows a variation on ArXiv:1908.0562 - instead of
    explicitly partitioning the list based on the binary indices of
    the qubits, we repeatedly divide the list in two and then
    zip it back together.

    Args:
        qubit_list(list): list of qubits to be partitioned
        num_iterations(int or None): number of iterations to perform.
            If None, will be set to ceil(log2(len(qubit_list)))

    Returns:
        partition(iterator of tuples of lists): the required partitioning
    """

    # Some edge cases
    if num_iterations is not None and num_iterations == 0:
        return
    num_qubits = len(qubit_list)
    if num_qubits < 2:
        raise ValueError('Need at least 2 qubits to partition')
    if num_qubits == 2:
        yield ([qubit_list[0]], [qubit_list[1]])
        return

    if num_iterations is None:
        num_iterations = int(numpy.ceil(numpy.log2(num_qubits)))

    # Calculate the point where we need to split the list each time.
    half_point = int(numpy.ceil(num_qubits / 2))

    # Repeat the division and zip steps as many times
    # as required.
    for _ in range(num_iterations):
        # Divide the qubit list in two and return it
        partition = (qubit_list[:half_point], qubit_list[half_point:])
        yield partition
        # Zip the partition together to remake the qubit list.
        qubit_list = list(chain(*zip_longest(partition[0], partition[1])))
        # If len(qubit_list) is odd, the end of the list will be 'None'
        # which we delete.
        if qubit_list[-1] is None:
            del qubit_list[-1]


def partition_iterator(qubit_list, partition_size, num_iterations=None):
    """Generator for a list of k-partitions of N qubits such that
    all sets of k qubits are perfectly split in at least one
    partition, following ArXiv:1908.05628

    Args:
        qubit_list(list): list of qubits to be partitioned
        partition_size(int): the number of sets in the partition.
        num_iterations(int or None): the number of iterations in the
            outer iterator. If None, set to ceil(log2(len(qubit_list)))

    Returns:
        partition(iterator of tuples of lists): the required partitioning
    """

    # Some edge cases
    if num_iterations == 0:
        return
    if partition_size == 1:
        yield (qubit_list,)
        return
    elif partition_size == 2:
        for p in binary_partition_iterator(qubit_list, num_iterations):
            yield p
        return
    num_qubits = len(qubit_list)
    if partition_size == num_qubits:
        yield tuple([q] for q in qubit_list)
        return
    elif partition_size > num_qubits:
        raise ValueError('I cant k-partition less than k qubits')

    if num_iterations is None:
        num_iterations = int(numpy.ceil(numpy.log2(num_qubits)))

    # First iterate over the outer binary partition
    outer_iterator = binary_partition_iterator(qubit_list, num_iterations=num_iterations)
    for set1, set2 in outer_iterator:
        # Each new partition needs to be subdivided fewer times
        # to prevent an additional k! factor in the scaling.
        num_iterations -= 1

        # Iterate over all possibilities of partitioning the first
        # set into l parts and the second set into k - l parts.
        for inner_partition_size in range(1, partition_size):
            if inner_partition_size > len(set1) or partition_size - inner_partition_size > len(
                set2
            ):
                continue

            # subdivide the first partition
            inner_iterator1 = partition_iterator(set1, inner_partition_size, num_iterations)
            for inner_partition1 in inner_iterator1:
                # subdivide the second partition
                inner_iterator2 = partition_iterator(
                    set2, partition_size - inner_partition_size, num_iterations
                )
                for inner_partition2 in inner_iterator2:
                    yield inner_partition1 + inner_partition2


def pauli_string_iterator(num_qubits, max_word_size=2):
    """Generates a set of Pauli strings such that each word
    of k Pauli operators lies in at least one string.

    Args:
        num_qubits(int): number of qubits in string
        max_word_size(int): maximum required word

    Returns:
        pauli_string(iterator of strings): iterator
            over Pauli strings
    """
    if max_word_size > num_qubits:
        raise ValueError('Number of qubits is too few')
    if max_word_size <= 0:
        raise ValueError('Word size too small')

    qubit_list = list(range(num_qubits))
    partitions = partition_iterator(qubit_list, max_word_size)
    pauli_string = ['I' for temp in range(num_qubits)]
    pauli_letters = ['X', 'Y', 'Z']
    for partition in partitions:
        for lettering in range(3**max_word_size):
            for p in partition:
                letter = pauli_letters[lettering % 3]
                for qubit in p:
                    pauli_string[qubit] = letter
                lettering = lettering // 3
            yield tuple(pauli_string)


def _find_compatible_basis(term, bases):
    for basis in bases:
        basis_qubits = {op[0] for op in basis}
        conflicts = ((i, P) for (i, P) in term if i in basis_qubits and (i, P) not in basis)
        if any(conflicts):
            continue
        return basis
    return None


def group_into_tensor_product_basis_sets(operator, seed=None):
    """
    Split an operator (instance of QubitOperator) into `sub-operator`
    QubitOperators, where each sub-operator has terms that are diagonal
    in the same tensor product basis.

    Each `sub-operator` can be measured using the same qubit post-rotations
    in expectation estimation. Grouping into these tensor product basis
    sets has been found to improve the efficiency of expectation estimation
    significantly for some Hamiltonians in the context of
    VQE (see section V(A) in the supplementary material of
    https://arxiv.org/pdf/1704.05018v2.pdf). The more general problem
    of grouping operators into commutitative groups is discussed in
    section IV (B2) of https://arxiv.org/pdf/1509.04279v1.pdf. The
    original input operator is the union of all output sub-operators,
    and all sub-operators are disjoint (do not share any terms).

    Args:
        operator (QubitOperator): the operator that will be split into
            sub-operators (tensor product basis sets).
        seed (int): default None. Random seed used to initialize the
            numpy.RandomState pseudo-random number generator.

    Returns:
        sub_operators (dict): a dictionary where each key defines a
            tensor product basis, and each corresponding value is a
            QubitOperator with terms that are all diagonal in
            that basis.
            **key** (tuple of tuples): Each key is a term, which defines
                a tensor product basis. A term is a product of individual
                factors; each factor is represented by a tuple of the form
                (`index`, `action`), and these tuples are collected into a
                larger tuple which represents the term as the product of
                its factors. `action` is from the set {'X', 'Y', 'Z'} and
                `index` is a non-negative integer corresponding to the
                index of a qubit.
            **value** (QubitOperator): A QubitOperator with terms that are
                diagonal in the basis defined by the key it is stored in.

    Raises:
       TypeError: Operator of invalid type.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError(
            'Can only split QubitOperator into tensor product'
            ' basis sets. {} is not supported.'.format(type(operator).__name__)
        )

    sub_operators = {}
    r = numpy.random.RandomState(seed)
    for term, coefficient in operator.terms.items():
        bases = list(sub_operators.keys())
        r.shuffle(bases)
        basis = _find_compatible_basis(term, bases)
        if basis is None:
            sub_operators[term] = QubitOperator(term, coefficient)
        else:
            sub_operator = sub_operators.pop(basis)
            sub_operator += QubitOperator(term, coefficient)
            additions = tuple(op for op in term if op not in basis)
            basis = tuple(sorted(basis + additions, key=lambda factor: factor[0]))
            sub_operators[basis] = sub_operator

    return sub_operators
