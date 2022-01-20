#coverage:ignore
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Union
import numpy as np

Step = int
Schedule = Callable[[Step], float]


def adagrad(step_size, momentum=0.9):
    """Construct optimizer triple for Adagrad.
  Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
  http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    momentum: optional, a positive scalar value for momentum
  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
    step_size = make_schedule(step_size)

    def init(x0):
        g_sq = np.zeros_like(x0)
        m = np.zeros_like(x0)
        return x0, g_sq, m

    def update(i, g, state):
        x, g_sq, m = state
        g_sq += np.square(g)
        g_sq_inv_sqrt = np.where(g_sq > 0, 1. / np.sqrt(g_sq), 0.0)
        m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
        x = x - step_size(i) * m
        return x, g_sq, m

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


### learning rate schedules


def constant(step_size) -> Schedule:

    def schedule(i):
        return step_size

    return schedule


def exponential_decay(step_size, decay_steps, decay_rate):

    def schedule(i):
        return step_size * decay_rate**(i / decay_steps)

    return schedule


def inverse_time_decay(step_size, decay_steps, decay_rate, staircase=False):
    if staircase:

        def schedule(i):
            return step_size / (1 + decay_rate * np.floor(i / decay_steps))
    else:

        def schedule(i):
            return step_size / (1 + decay_rate * i / decay_steps)

    return schedule


def polynomial_decay(step_size, decay_steps, final_step_size, power=1.0):

    def schedule(step_num):
        step_num = np.minimum(step_num, decay_steps)
        step_mult = (1 - step_num / decay_steps)**power
        return step_mult * (step_size - final_step_size) + final_step_size

    return schedule


def piecewise_constant(boundaries: Any, values: Any):
    boundaries = np.array(boundaries)
    values = np.array(values)
    if not boundaries.ndim == values.ndim == 1:
        raise ValueError("boundaries and values must be sequences")
    if not boundaries.shape[0] == values.shape[0] - 1:
        raise ValueError(
            "boundaries length must be one shorter than values length")

    def schedule(i):
        return values[np.sum(i > boundaries)]

    return schedule


def make_schedule(scalar_or_schedule: Union[float, Schedule]) -> Schedule:
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif np.ndim(scalar_or_schedule) == 0:
        return constant(scalar_or_schedule)
    else:
        raise TypeError(type(scalar_or_schedule))
