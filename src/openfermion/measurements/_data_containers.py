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
"""
Classes to store experimental data for postprocessing.
"""


class QPERoundData(object):
    """ Results from a single round in a QPE experiment.
    Attributes:
        num_rotations (int): the number of rotations performed in this round
        final_rotation (float): the final rotation performed on the ancilla qubit
            in this round.
        measurement (bool): the measurement observed in this round
        true_measurement (bool): if the ancilla is not reset between rounds, the
            'true' measurement (in this case msmt is calculated as the
            difference between the tmsmt of different rounds).
    """

    def __init__(self,
                 num_rotations,
                 final_rotation,
                 measurement,
                 true_measurement=None):
        """
        Args:
            num_rotations (int): the number of rotations performed in this round
            final_rotation (float): the final rotation performed on the ancilla qubit
                in this round.
            measurement (bool): the measurement observed in this round
            true_measurement (bool, optional): the 'true measurement' in each round:
                In some physical setups where ancilla reset is costly, one can perform
                QPE without resetting the ancilla qubit between rounds. In this case,
                if a string of measurements t_r is obtained for round r=1,2,... the
                'measurement' required by QPE for round r is m_r=t_r-t_{r-1}
                (with m_0=0). In the absence of errors the actual value of the t_r
                strings is not of interest, but T1 noise on the ancilla qubit affects
                the t_r values instead of the m_r values. This can be corrected for
                in some QPE estimators, which in turn require the value of t_r.
        """
        self.num_rotations = num_rotations
        self.final_rotation = final_rotation
        self.measurement = measurement
        self.true_measurement = true_measurement


class QPEExperimentData(object):
    """ Results from a single QPE experiment. A QPE experiment consists of
    multiple rounds, each round involving k controlled-U rotations on the
    system register, a final rotation by an angle beta on the ancilla qubit,
    and reports a measurement of m.

    Attributes:
        rounds: list of RoundData objects
    """
    def __init__(self, rounds=None):
        """
        Args (accepted in order):
            rounds: list of RoundData objects.
        """

        if rounds:
            self.rounds = rounds
        else:  # Empty experiment
            self.rounds = []
