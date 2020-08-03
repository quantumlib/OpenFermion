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
"""Tests for qubit_partitioning.py"""
import unittest

from openfermion.ops.operators import (QubitOperator, FermionOperator,
                                       BosonOperator)

from .qubit_partitioning import (binary_partition_iterator, partition_iterator,
                                 pauli_string_iterator,
                                 group_into_tensor_product_basis_sets)


class BinaryPartitionIteratorTest(unittest.TestCase):

    def test_num_partitions(self):
        qubit_list = range(6)
        bpi = binary_partition_iterator(qubit_list)
        count = 0
        for _, _ in bpi:
            count += 1
        self.assertEqual(count, 3)

    def test_partition_size(self):
        qubit_list = range(6)
        bpi = binary_partition_iterator(qubit_list)
        for p1, p2 in bpi:
            self.assertEqual(len(p1), 3)
            self.assertEqual(len(p2), 3)

    def test_partition_size_odd(self):
        qubit_list = range(7)
        bpi = binary_partition_iterator(qubit_list)
        for p1, p2 in bpi:
            self.assertEqual(len(p1), 4)
            self.assertEqual(len(p2), 3)

    def test_partitioning(self):
        qubit_list = list(range(6))
        for i in range(6):
            for j in range(i + 1, 6):
                print(i, j)
                flag = False
                bpi = binary_partition_iterator(qubit_list)
                for partition in bpi:
                    print(type(partition))
                    self.assertTrue(type(partition) is tuple)
                    p1, p2 = partition
                    print(p1, p2)
                    if (i in p1 and j in p2) or (j in p1 and i in p2):
                        flag = True
                self.assertTrue(flag)

    def test_partitioning_odd(self):
        qubit_list = list(range(7))
        for i in range(7):
            for j in range(i + 1, 7):
                print(i, j)
                flag = False
                bpi = binary_partition_iterator(qubit_list)
                for p1, p2 in bpi:
                    print(p1, p2)
                    if (i in p1 and j in p2) or (j in p1 and i in p2):
                        flag = True
                self.assertTrue(flag)

    def test_exception(self):
        with self.assertRaises(ValueError):
            bpi = binary_partition_iterator([])
            next(bpi)

    def test_partition_of_two(self):
        bpi = binary_partition_iterator([0, 1])
        count = 0
        for p1, p2 in bpi:
            count += 1
            self.assertEqual(p1[0], 0)
            self.assertEqual(p2[0], 1)
        self.assertEqual(count, 1)

    def test_zero_counting(self):
        bpi = binary_partition_iterator([0, 1], 0)
        with self.assertRaises(StopIteration):
            next(bpi)


class PartitionIteratorTest(unittest.TestCase):

    def test_unary_case(self):
        qubit_list = list(range(6))
        bpi = partition_iterator(qubit_list, 1)
        for p1, in bpi:
            self.assertEqual(p1, qubit_list)

    def test_binary_case(self):
        qubit_list = list(range(6))
        for i in range(6):
            for j in range(i + 1, 6):
                print(i, j)
                flag = False
                bpi = partition_iterator(qubit_list, 2)
                for p1, p2 in bpi:
                    print(p1, p2)
                    if (i in p1 and j in p2) or (j in p1 and i in p2):
                        flag = True
                self.assertTrue(flag)

    def test_exception(self):
        with self.assertRaises(ValueError):
            pi = partition_iterator([1, 2], 3)
            next(pi)

    def test_threepartition_three(self):
        bpi = partition_iterator([1, 2, 3], 3)
        count = 0
        for partition in bpi:
            print(type(partition))
            self.assertTrue(type(partition) is tuple)
            p1, p2, p3 = partition
            print(p1, p2, p3)
            self.assertEqual(len(p1), 1)
            self.assertEqual(p1[0], 1)
            self.assertEqual(len(p2), 1)
            self.assertEqual(p2[0], 2)
            self.assertEqual(len(p3), 1)
            self.assertEqual(p3[0], 3)
            count += 1
        self.assertEqual(count, 1)

    def test_partition_three(self):
        for num_qubits in range(1, 16):
            qubit_list = list(range(num_qubits))
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    for k in range(j + 1, num_qubits):
                        print('Testing {}, {}, {}'.format(i, j, k))
                        pi = partition_iterator(qubit_list, 3)
                        count = 0
                        for p1, p2, p3 in pi:
                            self.assertEqual(
                                len(p1) + len(p2) + len(p3), len(qubit_list))
                            self.assertEqual(set(p1 + p2 + p3), set(qubit_list))
                            print('Partition obtained: ', p1, p2, p3)
                            if max(
                                    sum(1
                                        for x in p
                                        if x in [i, j, k])
                                    for p in [p1, p2, p3]) == 1:
                                count += 1
                        print('count = {}'.format(count))
                        self.assertTrue(count > 0)
                        print()


class PauliStringIteratorTest(unittest.TestCase):

    def test_eightpartition_three(self):
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                for i3 in range(i2 + 1, 8):
                    for l1 in ['X', 'Y', 'Z']:
                        for l2 in ['X', 'Y', 'Z']:
                            for l3 in ['X', 'Y', 'Z']:
                                psg = pauli_string_iterator(8, 3)
                                count = 0
                                for pauli_string in psg:
                                    if (pauli_string[i1] == l1 and
                                            pauli_string[i2] == l2 and
                                            pauli_string[i3] == l3):
                                        count += 1
                                self.assertTrue(count > 0)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            psi = pauli_string_iterator(1, 2)
            next(psi)
        with self.assertRaises(ValueError):
            psi = pauli_string_iterator(3, -1)
            next(psi)

    def test_small_run_cases(self):
        for num_qubits in range(4, 20):
            for word_length in range(2, min(num_qubits, 5)):
                psi = pauli_string_iterator(num_qubits, word_length)
                for _ in psi:
                    pass


class GroupTensorProductBasisTest(unittest.TestCase):

    def test_demo_qubit_operator(self):
        for seed in [None, 0, 10000]:
            op = QubitOperator('X0 Y1', 2.) + QubitOperator('X1 Y2', 3.j)
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected = {
                ((0, 'X'), (1, 'Y')): QubitOperator('X0 Y1', 2.),
                ((1, 'X'), (2, 'Y')): QubitOperator('X1 Y2', 3.j)
            }
            self.assertEqual(sub_operators, expected)

            op = QubitOperator('X0 Y1', 2.) + QubitOperator('Y1 Y2', 3.j)
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected = {((0, 'X'), (1, 'Y'), (2, 'Y')): op}
            self.assertEqual(sub_operators, expected)

            op = QubitOperator('', 4.) + QubitOperator('X1', 2.j)
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected = {((1, 'X'),): op}
            self.assertEqual(sub_operators, expected)

            op = (QubitOperator('X0 X1', 0.1) + QubitOperator('X1 X2', 2.j) +
                  QubitOperator('Y2 Z3', 3.) + QubitOperator('X3 Z4', 5.))
            sub_operators = group_into_tensor_product_basis_sets(op, seed=seed)
            expected1 = {
                ((0, 'X'), (1, 'X'), (2, 'X'), (3, 'X'), (4, 'Z')):
                (QubitOperator('X0 X1', 0.1) + QubitOperator('X1 X2', 2.j) +
                 QubitOperator('X3 Z4', 5.)),
                ((2, 'Y'), (3, 'Z')):
                QubitOperator('Y2 Z3', 3.)
            }
            expected2 = {
                ((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Z')):
                (QubitOperator('X0 X1', 0.1) + QubitOperator('Y2 Z3', 3.)),
                ((1, 'X'), (2, 'X'), (3, 'X'), (4, 'Z')):
                (QubitOperator('X1 X2', 2.j) + QubitOperator('X3 Z4', 5.))
            }
            self.assertTrue(sub_operators == expected1 or
                            sub_operators == expected2)

    def test_empty_qubit_operator(self):
        sub_operators = group_into_tensor_product_basis_sets(QubitOperator())
        self.assertTrue(sub_operators == {})

    def test_fermion_operator_bad_type(self):
        with self.assertRaises(TypeError):
            _ = group_into_tensor_product_basis_sets(FermionOperator())

    def test_boson_operator_bad_type(self):
        with self.assertRaises(TypeError):
            _ = group_into_tensor_product_basis_sets(BosonOperator())

    def test_none_bad_type(self):
        with self.assertRaises(TypeError):
            _ = group_into_tensor_product_basis_sets(None)
