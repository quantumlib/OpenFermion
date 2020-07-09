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
'''Tests for the qubit_operator_transforms module'''

import unittest
import numpy

from openfermion.ops import QubitOperator
from openfermion.ops import FermionOperator
from openfermion.transforms import (
    project_onto_sector, projection_error, rotate_qubit_by_pauli)
from openfermion.utils import count_qubits


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_function_errors(self):
        """Test main function errors."""
        operator = (QubitOperator('Z0 X1', 1.0) +
                    QubitOperator('X1', 2.0))
        sector1 = [0]
        sector2 = [1]
        qbt_list = [0]
        with self.assertRaises(TypeError):
            project_onto_sector(operator=1.0, qubits=qbt_list, sectors=sector1)
        with self.assertRaises(TypeError):
            projection_error(operator=1.0, qubits=qbt_list, sectors=sector1)
        with self.assertRaises(TypeError):
            project_onto_sector(operator=operator, qubits=0.0, sectors=sector2)
        with self.assertRaises(TypeError):
            projection_error(operator=operator, qubits=0.0, sectors=sector2)
        with self.assertRaises(TypeError):
            project_onto_sector(operator=operator,
                                qubits=qbt_list, sectors=operator)
        with self.assertRaises(TypeError):
            projection_error(operator=operator,
                             qubits=qbt_list, sectors=operator)
        with self.assertRaises(ValueError):
            project_onto_sector(operator=operator, qubits=[
                                0, 1], sectors=sector1)
        with self.assertRaises(ValueError):
            projection_error(operator=operator, qubits=[0, 1], sectors=sector1)
        with self.assertRaises(ValueError):
            project_onto_sector(operator=operator,
                                qubits=qbt_list, sectors=[0, 0])
        with self.assertRaises(ValueError):
            projection_error(operator=operator,
                             qubits=qbt_list, sectors=[0, 0])
        with self.assertRaises(ValueError):
            project_onto_sector(operator=operator,
                                qubits=qbt_list, sectors=[-1])
        with self.assertRaises(ValueError):
            projection_error(operator=operator, qubits=qbt_list, sectors=[-1])

    def test_projection(self):
        coefficient = 0.5
        opstring = ((0, 'X'), (1, 'X'), (2, 'Z'))
        opstring2 = ((0, 'X'), (2, 'Z'), (3, 'Z'))
        operator = QubitOperator(opstring, coefficient)
        operator += QubitOperator(opstring2, coefficient)
        new_operator = project_onto_sector(
            operator, qubits=[2, 3], sectors=[0, 1])
        error = projection_error(operator, qubits=[2, 3], sectors=[0, 1])
        self.assertEqual(count_qubits(new_operator), 2)
        self.assertEqual(error, 0)
        self.assertTrue(((0, 'X'), (1, 'X')) in new_operator.terms)
        self.assertEqual(new_operator.terms[((0, 'X'), (1, 'X'))], 0.5)
        self.assertTrue(((0, 'X'),) in new_operator.terms)
        self.assertEqual(new_operator.terms[((0, 'X'),)], -0.5)

    def test_projection_error(self):
        coefficient = 0.5
        opstring = ((0, 'X'), (1, 'X'), (2, 'Z'))
        opstring2 = ((0, 'X'), (2, 'Z'), (3, 'Z'))
        operator = QubitOperator(opstring, coefficient)
        operator += QubitOperator(opstring2, coefficient)
        new_operator = project_onto_sector(operator, qubits=[1], sectors=[0])
        error = projection_error(operator, qubits=[1], sectors=[0])
        self.assertEqual(count_qubits(new_operator), 3)
        self.assertTrue(((0, 'X'), (1, 'Z'), (2, 'Z')) in new_operator.terms)
        self.assertEqual(new_operator.terms[((0, 'X'), (1, 'Z'), (2, 'Z'))],
                         0.5)
        self.assertEqual(error, 0.5)


class UnitaryRotationsTest(unittest.TestCase):

    def setup(self):
        pass

    def test_rotation(self):
        qop = QubitOperator('X0 X1', 1)
        qop += QubitOperator('Z0 Z1', 1)
        rot_op = QubitOperator('Z1', 1)

        rotated_qop = rotate_qubit_by_pauli(qop, rot_op, numpy.pi / 4)
        comp_op = QubitOperator('Z0 Z1', 1)
        comp_op += QubitOperator('X0 Y1', 1)
        self.assertEqual(comp_op, rotated_qop)

    def test_exception_Pauli(self):
        qop = QubitOperator('X0 X1', 1)
        qop += QubitOperator('Z0 Z1', 1)
        rot_op = QubitOperator('Z1', 1)
        rot_op += QubitOperator('Z0', 1)
        rot_op2 = QubitOperator('Z1', 1)
        ferm_op = FermionOperator('1^ 2', 1)
        with self.assertRaises(TypeError):
            rotate_qubit_by_pauli(qop, rot_op, numpy.pi / 4)
        with self.assertRaises(TypeError):
            rotate_qubit_by_pauli(ferm_op, rot_op2, numpy.pi / 4)
