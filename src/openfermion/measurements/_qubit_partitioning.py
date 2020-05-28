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
from itertools import chain
import numpy
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest


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
    half_point = int(numpy.ceil(num_qubits/2))

    # Repeat the division and zip steps as many times
    # as required.
    for _ in range(num_iterations):
        # Divide the qubit list in two and return it
        partition = (qubit_list[:half_point],
                     qubit_list[half_point:])
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
        yield (qubit_list, )
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
    outer_iterator = binary_partition_iterator(
        qubit_list, num_iterations=num_iterations)
    for set1, set2 in outer_iterator:

        # Each new partition needs to be subdivided fewer times
        # to prevent an additional k! factor in the scaling.
        num_iterations -= 1

        # Iterate over all possibilities of partitioning the first
        # set into l parts and the second set into k - l parts.
        for inner_partition_size in range(1, partition_size):
            if inner_partition_size > len(set1) or\
                    partition_size - inner_partition_size > len(set2):
                continue

            # subdivide the first partition
            inner_iterator1 = partition_iterator(
                set1, inner_partition_size, num_iterations)
            for inner_partition1 in inner_iterator1:

                # subdivide the second partition
                inner_iterator2 = partition_iterator(
                    set2, partition_size-inner_partition_size,
                    num_iterations)
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
