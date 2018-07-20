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

"""Tests  _unitary_rotations.py."""
from __future__ import absolute_import
from openfermion.transforms import rotate_qubit_by_pauli
from openfermion.ops import QubitOperator
import numpy

import unittest


class UnitaryRotationsTest(unittest.TestCase):

    def setup(self):
        pass

    def test_rotation(self):
        qop = QubitOperator('X0 X1', 1)
        qop += QubitOperator('Z0 Z1', 1)
        rot_op = QubitOperator('Z1', 1)

        rotated_qop = rotate_qubit_by_pauli(qop, rot_op, numpy.pi/4)
        comp_op = QubitOperator('Z0 Z1', 1)
        comp_op += QubitOperator('X0 Y1', -1)
        self.assertEqual(comp_op, rotated_qop)
