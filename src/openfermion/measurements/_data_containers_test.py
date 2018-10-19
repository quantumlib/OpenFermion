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
"""Tests for _data_containers.py"""

import unittest
from ._data_containers import(
    QPERoundData,
    QPEExperimentData)

class QPERoundDataTest(unittest.TestCase):

    def test_init(self):
        rd = QPERoundData(
            num_rotations=1,
            final_rotation=0,
            measurement=0)
        self.assertEqual(rd.num_rotations, 1)
        self.assertEqual(rd.final_rotation, 0)
        self.assertEqual(rd.measurement, 0)
        self.assertEqual(rd.true_measurement, None)

class QPEExperimentDataTest(unittest.TestCase):

    def test_blank_init(self):
        ed = QPEExperimentData()
        self.assertEqual(len(ed.rounds), 0)

    def test_init_rounddata(self):
        rd = QPERoundData(
            num_rotations=1,
            final_rotation=0,
            measurement=0)
        ed = QPEExperimentData(rounds=[rd])
        self.assertEqual(len(ed.rounds), 1)
        self.assertEqual(ed.rounds[0].num_rotations, 1)
        self.assertEqual(ed.rounds[0].final_rotation, 0)
        self.assertEqual(ed.rounds[0].measurement, 0)
