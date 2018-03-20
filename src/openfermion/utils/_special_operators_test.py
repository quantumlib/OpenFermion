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

"""testing angular momentum generators. _fermion_spin_operators.py"""
import numpy
import unittest
from openfermion.ops import FermionOperator
from openfermion.utils import (majorana_operator, s_minus_operator,
                               s_plus_operator, s_squared_operator,
                               sx_operator, sy_operator, sz_operator,
                               up_index, down_index)


class FermionSpinOperatorsTest(unittest.TestCase):

    def test_up_index(self):
        self.assertTrue(numpy.isclose(up_index(2), 4))
        self.assertTrue(numpy.isclose(up_index(5), 10))

    def test_up_down(self):
        self.assertTrue(numpy.isclose(down_index(2), 5))
        self.assertTrue(numpy.isclose(down_index(5), 11))

    def test_s_plus_operator(self):
        op = s_plus_operator(2)
        expected = (FermionOperator(((0, 1), (1, 0))) +
                    FermionOperator(((2, 1), (3, 0))))
        self.assertTrue(op.isclose(expected))

    def test_s_minus_operator(self):
        op = s_minus_operator(3)
        expected = (FermionOperator(((1, 1), (0, 0))) +
                    FermionOperator(((3, 1), (2, 0))) +
                    FermionOperator(((5, 1), (4, 0))))
        self.assertTrue(op.isclose(expected))

    def test_sx_operator(self):
        op = sx_operator(2)
        expected = (FermionOperator(((0, 1), (1, 0)), 0.5) +
                    FermionOperator(((1, 1), (0, 0)), 0.5) +
                    FermionOperator(((2, 1), (3, 0)), 0.5) +
                    FermionOperator(((3, 1), (2, 0)), 0.5))
        self.assertTrue(op.isclose(expected))

    def test_sy_operator(self):
        op = sy_operator(2)
        expected = (FermionOperator(((0, 1), (1, 0)), -0.5j) -
                    FermionOperator(((1, 1), (0, 0)), -0.5j) +
                    FermionOperator(((2, 1), (3, 0)), -0.5j) -
                    FermionOperator(((3, 1), (2, 0)), -0.5j))
        self.assertTrue(op.isclose(expected))

    def test_sz_operator(self):
        op = sz_operator(2)
        expected = (FermionOperator(((0, 1), (0, 0)), 0.5) -
                    FermionOperator(((1, 1), (1, 0)), 0.5) +
                    FermionOperator(((2, 1), (2, 0)), 0.5) -
                    FermionOperator(((3, 1), (3, 0)), 0.5))
        self.assertTrue(op.isclose(expected))

    def test_s_squared_operator(self):
        op = s_squared_operator(2)
        s_minus = (FermionOperator(((1, 1), (0, 0))) +
                   FermionOperator(((3, 1), (2, 0))))
        s_plus = (FermionOperator(((0, 1), (1, 0))) +
                  FermionOperator(((2, 1), (3, 0))))
        s_z = (FermionOperator(((0, 1), (0, 0)), 0.5) -
               FermionOperator(((1, 1), (1, 0)), 0.5) +
               FermionOperator(((2, 1), (2, 0)), 0.5) -
               FermionOperator(((3, 1), (3, 0)), 0.5))
        expected = s_minus * s_plus + s_z * s_z + s_z
        self.assertTrue(op.isclose(expected))

    def test_relations(self):
        n_spatial_orbitals = 2

        s_plus = s_plus_operator(n_spatial_orbitals)
        s_minus = s_minus_operator(n_spatial_orbitals)
        s_x = s_x_operator(n_spatial_orbitals)
        s_y = s_y_operator(n_spatial_orbitals)
        s_z = s_z_operator(n_spatial_orbitals)
        s_squared = s_squared_operator(n_spatial_orbitals)

        #TODO

    def test_invalid_input(self):

        with self.assertRaises(TypeError):
            s_minus_operator('a')

        with self.assertRaises(TypeError):
            s_plus_operator('a')

        with self.assertRaises(TypeError):
            sx_operator('a')

        with self.assertRaises(TypeError):
            sy_operator('a')

        with self.assertRaises(TypeError):
            sz_operator('a')

        with self.assertRaises(TypeError):
            s_squared_operator('a')


class MajoranaOperatorTest(unittest.TestCase):

    def test_init(self):
        # Test 'c' operator
        op1 = majorana_operator((2, 0))
        op2 = majorana_operator('c2')
        correct = FermionOperator('2^') + FermionOperator('2')
        self.assertTrue(op1.isclose(op2))
        self.assertTrue(op1.isclose(correct))

        # Test 'd' operator
        op1 = majorana_operator((3, 1))
        op2 = majorana_operator('d3')
        correct = FermionOperator('3^', 1.j) - FermionOperator('3', 1.j)
        self.assertTrue(op1.isclose(op2))
        self.assertTrue(op1.isclose(correct))

    def test_none_term(self):
        majorana_op = majorana_operator()
        self.assertTrue(majorana_operator().isclose(FermionOperator()))

    def test_bad_coefficient(self):
        with self.assertRaises(ValueError):
            majorana_op = majorana_operator((1, 1), 'a')

    def test_bad_term(self):
        with self.assertRaises(ValueError):
            majorana_op = majorana_operator((2, 2))
        with self.assertRaises(ValueError):
            majorana_op = majorana_operator('a')
