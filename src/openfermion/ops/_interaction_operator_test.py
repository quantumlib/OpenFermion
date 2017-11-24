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

"""Tests for interaction_operator.py."""
from __future__ import absolute_import

import numpy
import unittest

from openfermion.ops import InteractionOperator


class InteractionOperatorTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.constant = 0.
        self.one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        self.two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                     self.n_qubits, self.n_qubits), float)
        self.interaction_operator = InteractionOperator(self.constant,
                                                        self.one_body,
                                                        self.two_body)

    def test_four_point_iter(self):
        constant = 100.0
        one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits), float)
        one_body[1, 1] = 10.0
        one_body[2, 3] = 11.0
        one_body[3, 2] = 11.0
        two_body[1, 2, 3, 4] = 12.0
        two_body[2, 1, 4, 3] = 12.0
        two_body[3, 4, 1, 2] = 12.0
        two_body[4, 3, 2, 1] = 12.0
        interaction_operator = InteractionOperator(
            constant, one_body, two_body)

        want_str = '100.0\n10.0\n11.0\n12.0\n'
        got_str = ''
        for key in interaction_operator.unique_iter(complex_valued=True):
            got_str += '{}\n'.format(interaction_operator[key])
        self.assertEqual(want_str, got_str)

    def test_eight_point_iter(self):
        constant = 100.0
        one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits), float)
        one_body[1, 1] = 10.0
        one_body[2, 3] = 11.0
        one_body[3, 2] = 11.0
        two_body[1, 2, 3, 4] = 12.0
        two_body[2, 1, 4, 3] = 12.0
        two_body[3, 4, 1, 2] = 12.0
        two_body[4, 3, 2, 1] = 12.0
        two_body[1, 4, 3, 2] = 12.0
        two_body[2, 3, 4, 1] = 12.0
        two_body[3, 2, 1, 4] = 12.0
        two_body[4, 1, 2, 3] = 12.0
        interaction_operator = InteractionOperator(
            constant, one_body, two_body)

        want_str = '100.0\n10.0\n11.0\n12.0\n'
        got_str = ''
        for key in interaction_operator.unique_iter():
            got_str += '{}\n'.format(interaction_operator[key])
        self.assertEqual(want_str, got_str)
