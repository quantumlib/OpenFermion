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
import numpy
from itertools import chain
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest


def binary_partition_iterator(qubit_list, num_partitions=None):
    '''Generator for a list of 2-partitions of N qubits
    such that all pairs of qubits are split in at least one partition,
    following ArXiv:1908.05628

    Args:
        qubit_list(list): list of qubits to be partitioned

    Returns:
        partition(iterator of lists of lists): the required partitioning
    '''
    if num_partitions is not None and num_partitions == 0:
        return
    num_qubits = len(qubit_list)
    if num_qubits < 2:
        raise ValueError('Need at least 2 qubits to partition')
    if num_qubits == 2:
        yield [[qubit_list[0]], [qubit_list[1]]]
        return

    num_partitions = num_partitions or\
        int(numpy.ceil(numpy.log2(num_qubits)))

    partition = [qubit_list[:num_qubits//2], qubit_list[num_qubits//2:]]

    for j in range(num_partitions):
        yield partition
        merged_partition = list(
            chain(*zip_longest(partition[1], partition[0])))
        if merged_partition[-1] is None:
            del merged_partition[-1]
        partition = [merged_partition[:num_qubits//2],
                     merged_partition[num_qubits//2:]]


def partition_iterator(qubit_list, partition_size, num_partitions=None):
    '''Generator for a list of k-partitions of N qubits such that
    all sets of k qubits are perfectly split in at least one
    partition, following ArXiv:1908.05628

    Args:
        qubit_list(list): list of qubits to be partitioned
        partition_size(int): the number of partitions.

    Returns:
        partition(iterator of lists of lists): the required partitioning
    '''
    if num_partitions is not None and num_partitions == 0:
        return
    if partition_size == 1:
        yield [qubit_list]
        return
    elif partition_size == 2:
        for p in binary_partition_iterator(qubit_list, num_partitions):
            yield p
        return

    num_qubits = len(qubit_list)
    if partition_size == num_qubits:
        yield [[q] for q in qubit_list]
        return
    elif partition_size > num_qubits:
        raise ValueError('I cant k-partition less than k qubits')

    num_partitions = num_partitions or\
        int(numpy.ceil(numpy.log2(num_qubits)))

    outer_iterator = binary_partition_iterator(
        qubit_list, num_partitions=num_partitions)

    for p1, p2 in outer_iterator:
        num_partitions -= 1

        for inner_partition_size in range(1, partition_size):
            if inner_partition_size > len(p1) or\
                    partition_size - inner_partition_size > len(p2):
                continue

            inner_iterator1 = partition_iterator(
                p1, inner_partition_size, num_partitions)

            for ips1 in inner_iterator1:
                inner_iterator2 = partition_iterator(
                    p2, partition_size-inner_partition_size,
                    num_partitions)

                for ips2 in inner_iterator2:
                    yield ips1 + ips2


def pauli_string_iterator(num_qubits, max_word_size=2):
    '''Generates a set of Pauli strings such that each word
    of k Pauli operators lies in at least one string.

    Args:
        num_qubits(int): number of qubits in string
        max_word_size(int): maximum required word

    Returns:
        pauli_string(iterator of strings): iterator
            over Pauli strings
    '''
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
            yield pauli_string