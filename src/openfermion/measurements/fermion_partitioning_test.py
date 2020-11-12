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
"""Tests for _qubit_partitioning.py"""
import unittest
import numpy
from .fermion_partitioning import (pair_within, pair_within_simultaneously,
                                   pair_within_simultaneously_binned,
                                   pair_within_simultaneously_symmetric,
                                   _gen_partitions, _get_padding,
                                   _parallel_iter, _asynchronous_iter)


class TestPairWithin(unittest.TestCase):

    def test_zero(self):
        count = 0
        for _ in pair_within([]):
            count += 1  # pragma: no cover
        self.assertEqual(count, 0)

    def test_pair(self):
        labels = [0, 1]
        pairings_list = [pairing for pairing in pair_within(labels)]
        self.assertEqual(len(pairings_list), 1)
        self.assertEqual(len(pairings_list[0]), 1)
        self.assertEqual(pairings_list[0][0], (0, 1))

    def test_threeway(self):
        labels = [0, 1, 2]
        pairings_list = [pairing for pairing in pair_within(labels)]
        self.assertEqual(len(pairings_list), 3)
        all_pairs = [(0, 1), (0, 2), (1, 2)]
        checksum = [0, 0, 0]
        for pairing in pairings_list:
            self.assertEqual(len(pairing), 2)
            for j in range(3):
                if all_pairs[j] in pairing:
                    checksum[j] += 1
        print(checksum)
        for j in range(3):
            self.assertEqual(checksum[j], 1)

    def test_many(self):
        for num_indices in range(2, 16):
            print()
            print(num_indices)
            labels = [j for j in range(num_indices)]
            pairings_list = [pairing for pairing in pair_within(labels)]
            if num_indices % 2 == 0:
                self.assertEqual(len(pairings_list), num_indices - 1)
            else:
                self.assertEqual(len(pairings_list), num_indices)
            for pairing in pairings_list:
                print(pairing)
            all_pairs = [(i, j)
                         for i in range(num_indices)
                         for j in range(i + 1, num_indices)]
            checksum = [0 for _ in all_pairs]
            for pairing in pairings_list:
                for j, pair in enumerate(all_pairs):
                    if pair in pairing:
                        checksum[j] += 1
            print(checksum)
            for cs in checksum:
                self.assertGreaterEqual(cs, 1)


class TestPairWithinSimultaneously(unittest.TestCase):

    def test_small(self):
        for num_indices in [4, 5, 7, 10, 15]:
            print()
            print('Trying with num_indices = {}'.format(num_indices))
            labels = [j for j in range(num_indices)]
            all_quads = [((i, j, k, l)) for i in range(num_indices)
                         for j in range(i + 1, num_indices)
                         for k in range(j + 1, num_indices)
                         for l in range(k + 1, num_indices)]
            checksum = [0 for pp2 in all_quads]
            for pairing in pair_within_simultaneously(labels):
                print(pairing)
                for j, quad in enumerate(all_quads):
                    for n1 in range(3):
                        n2 = (n1 + 1) % 3
                        n3 = (n1 + 2) % 3
                        pair1 = (quad[0], quad[n1 + 1])
                        pair2 = (quad[min(n2, n3) + 1], quad[max(n2, n3) + 1])
                        if pair1 in pairing and pair2 in pairing:
                            checksum[j] += 1

            for j in range(len(checksum)):
                print(all_quads[j], checksum[j])
                self.assertGreaterEqual(checksum[j], 1)


class TestPadding(unittest.TestCase):

    def test_primes(self):
        bin_size = 11
        for num_bins in range(10):
            self.assertEqual(_get_padding(num_bins, bin_size), bin_size)

    def test_composite(self):
        bin_size = 12
        for num_bins in range(4, 13):
            self.assertEqual(_get_padding(num_bins, bin_size), 13)


class TestIterators(unittest.TestCase):

    def test_asynchronous_three(self):
        lists = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        test_matrices = [
            [numpy.zeros((3, 3)) for j in range(4)] for k in range(4)
        ]
        iterator = _asynchronous_iter(lists)
        count = 0
        for next_tuple in iterator:
            print(next_tuple)
            count += 1
            for j in range(4):
                for k in range(4):
                    test_matrices[j][k][next_tuple[j], next_tuple[k]] += 1

        self.assertEqual(count, 9)
        for j in range(4):
            for k in range(j + 1, 4):
                for i1 in range(3):
                    for i2 in range(3):
                        self.assertEqual(test_matrices[j][k][i1, i2], 1)

    def test_asynchronous_four(self):
        lists = [[x for x in range(4)] for y in range(4)]
        iterator = _asynchronous_iter(lists)
        count = 0
        for _ in iterator:
            count += 1
        self.assertEqual(count, 25)

    def test_parallel(self):

        def iter1():
            for j in range(4):
                yield j

        def iter2():
            for j in range(5):
                yield j

        iterators = [iter1(), iter2()]
        count = 0
        for _ in _parallel_iter(iterators):
            count += 1
        self.assertEqual(count, 5)

    def test_parallel_flatten(self):

        def iter1():
            for j in range(4):
                yield [j]

        def iter2():
            for j in range(5):
                yield [j]

        iterators = [iter1(), iter2()]
        for res in _parallel_iter(iterators, flatten=False):
            self.assertTrue(isinstance(res[0], list))
        iterators = [iter1(), iter2()]
        for res in _parallel_iter(iterators, flatten=True):
            self.assertTrue(isinstance(res[0], int))


class TestPairingWithSymmetries(unittest.TestCase):

    def test_two_fermions(self):
        bins = [[1], [2], [3], [4]]
        count = 0
        for pairing in pair_within_simultaneously_binned(bins):
            count += 1
            self.assertEqual(len(pairing), 2)
            print(pairing)
        self.assertEqual(count, 1)
        count = 0
        for pairing in pair_within_simultaneously_symmetric(2, 2):
            count += 1
            self.assertEqual(len(pairing), 2)
        self.assertEqual(count, 1)
        count = 0
        for pairing in pair_within_simultaneously_symmetric(2, 1):
            count += 1
            self.assertEqual(len(pairing), 2)
        self.assertEqual(count, 1)
        count = 0
        for pairing in pair_within_simultaneously_symmetric(2, 0):
            count += 1
            self.assertEqual(len(pairing), 2)
        self.assertEqual(count, 1)

    def test_four_fermions(self):
        print('Trying with 0 symmetries')
        count = 0
        for pairing in pair_within_simultaneously_symmetric(4, 0):
            print(pairing)
            count += 1
            self.assertEqual(len(pairing), 4)
        self.assertEqual(count, 18)
        print('Trying with 1 symmetry')
        count = 0
        for pairing in pair_within_simultaneously_symmetric(4, 1):
            print(pairing)
            count += 1
            self.assertEqual(len(pairing), 4)
        self.assertEqual(count, 10)
        print('Trying with 2 symmetries')
        count = 0
        for pairing in pair_within_simultaneously_symmetric(4, 2):
            print(pairing)
            count += 1
            self.assertEqual(len(pairing), 4)
        self.assertEqual(count, 5)
        print('Trying with 3 symmetries')
        count = 0
        for pairing in pair_within_simultaneously_symmetric(4, 3):
            print(pairing)
            count += 1
            self.assertEqual(len(pairing), 4)
        self.assertEqual(count, 3)

    def test_four_symmetries(self):
        for num_fermions in [5, 8, 9]:
            for _ in pair_within_simultaneously_symmetric(num_fermions, 3):
                pass


def test_gen_partitions_1input():
    labels = [
        0,
    ]
    count = 0
    for _ in _gen_partitions(labels):
        count += 1
    assert count == 1
