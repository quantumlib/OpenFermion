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

"""This module provides various order functions that can be used with
reorder function from _operator_utils. Every order function must take
in a mode_idx and num_modes."""


def up_then_down(mode_idx, num_modes):
    """ up then down reordering, given the operator has the default even-odd
     ordering. Otherwise this function will reorder indices where all even
     indices now come before odd indices.

     Example:
         0,1,2,3,4,5 -> 0,2,4,1,3,5

    The function takes in the index of the mode that will be relabeled and
    the total number modes.

    Args:
        mode_idx (int): the mode index that is being reordered
        num_modes (int): the total number of modes of the operator.

    Returns (int): reordered index of the mode.
    """
    halfway = num_modes / 2
    if mode_idx % 2 == 0:
        return mode_idx // 2
    else:
        return mode_idx // 2 + halfway