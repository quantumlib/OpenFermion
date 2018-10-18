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
        nrot (int): the number of rotations performed in this round
        finrot (float): the final rotation performed on the ancilla qubit
            in this round.
        msmt (bool): the measurement observed in this round
        tmsmt (bool): if the ancilla is not reset between rounds, the
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
            nrot (int): the number of rotations performed in this round
            finrot (float): the final rotation performed on the ancilla qubit
                in this round.
            msmt (bool): the measurement observed in this round
            tmsmt (bool): if the ancilla is not reset between rounds, the
                'true' measurement (in this case msmt is calculated as the
                difference between the tmsmt of different rounds).
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
    def __init__(self,
            rounds=None,
            list_num_rotations=None,
            list_final_rotation=None,
            list_measurement=None,
            list_true_measurement=None):
        """
        Initialization can be performed in one of two ways:
        as stated in the arguments

        Args (accepted in order):
            rounds: list of RoundData objects or lists of dicts
                to be converted to RoundData objects.

            list_num_rotations (list of integers),
            list_final_rotation (list of floats),
            list_measurement (list of booleans),
            tmsmt_list(optional) (list of booleans):
                experimental parameters (see RoundData class for description).
        """

        if rounds:
            if type(rounds[0]) is dict:
                rounds = [QPERoundData(**r) for r in rounds]
            self.rounds = rounds
        elif list_num_rotations:
            if list_true_measurement:
                self.rounds = [QPERoundData(nrot,finrot,msmt,tmsmt)
                               for nrot, finrot, msmt, tmsmt in zip(
                                list_num_rotations,
                                list_final_rotation,
                                list_measurement,
                                list_true_measurement)]
            else:
                self.rounds = [QPERoundData(nrot,finrot,msmt)
                               for nrot, finrot, msmt in zip(
                                list_num_rotations,
                                list_final_rotation,
                                list_measurement)]
        else:  # Empty experiment
            self.rounds = []