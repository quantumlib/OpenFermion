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
from ._fermion_partitioning import pair_within, pair_within_simultaneously


class TestPairWithin(unittest.TestCase):

    def test_pair(self):
        labels = [0, 1]
        pairings_list = [
            pairing for pairing in pair_within(labels)]
        self.assertEqual(len(pairings_list), 1)
        self.assertEqual(len(pairings_list[0]), 1)
        self.assertEqual(pairings_list[0][0], (0, 1))

    def test_threeway(self):
        labels = [0, 1, 2]
        pairings_list = [
            pairing for pairing in pair_within(labels)]
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
            pairings_list = [
                pairing for pairing in pair_within(labels)]
            if num_indices % 2 == 0:
                self.assertEqual(len(pairings_list), num_indices-1)
            else:
                self.assertEqual(len(pairings_list), num_indices)
            for pairing in pairings_list:
                print(pairing)
            all_pairs = [(i, j) for i in range(num_indices)
                         for j in range(i+1, num_indices)]
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
        for num_indices in [4, 5, 7, 10, 15, 16, 25]:
            print()
            print('Trying with num_indices = {}'.format(num_indices))
            labels = [j for j in range(num_indices)]
            all_quads = [((i, j, k, l)) for i in range(num_indices)
                         for j in range(i+1, num_indices)
                         for k in range(j+1, num_indices)
                         for l in range(k+1, num_indices)]
            checksum = [0 for pp2 in all_quads]
            for pairing in pair_within_simultaneously(labels):
                print(pairing)
                for j, quad in enumerate(all_quads):
                    for n1 in range(3):
                        n2 = (n1 + 1) % 3
                        n3 = (n1 + 2) % 3
                        pair1 = (quad[0], quad[n1+1])
                        pair2 = (quad[min(n2, n3)+1], quad[max(n2, n3) + 1])
                        if pair1 in pairing and pair2 in pairing:
                            checksum[j] += 1

            for j in range(len(checksum)):
                print(all_quads[j], checksum[j])
                self.assertGreaterEqual(checksum[j], 1)
