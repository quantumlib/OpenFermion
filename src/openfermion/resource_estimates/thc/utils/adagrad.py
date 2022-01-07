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
"""
JAX free implementation of Adagrad

Timing indicates lots of XLA stuff happening that I don't want in the
jax implementation.

Pure python implementaiton here.

Optimizers for use with JAX (but removing JAX).
This module contains some convenient optimizer definitions, specifically
initialization and update functions, which can be used with ndarrays or
arbitrarily-nested tuple/list/dicts of ndarrays.
An optimizer is modeled as an ``(init_fun, update_fun, get_params)`` triple of
functions, where the component functions have these signatures:
::
  init_fun(params)
  Args:
    params: pytree representing the initial parameters.
  Returns:
    A pytree representing the initial optimizer state, which includes the
    initial parameters and may also include auxiliary values like initial
    momentum. The optimizer state pytree structure generally differs from that
    of `params`.
::
  update_fun(step, grads, opt_state)
  Args:
    step: integer representing the step index.
    grads: a pytree with the same structure as `get_params(opt_state)`
      representing the gradients to be used in updating the optimizer state.
    opt_state: a pytree representing the optimizer state to be updated.
  Returns:
    A pytree with the same structure as the `opt_state` argument representing
    the updated optimizer state.
::
  get_params(opt_state)
  Args:
    opt_state: pytree representing an optimizer state.
  Returns:
    A pytree representing the parameters extracted from `opt_state`, such that
    the invariant `params == get_params(init_fun(params))` holds true.
Notice that an optimizer implementation has a lot of flexibility in the form of
opt_state: it just has to be a pytree of JaxTypes (so that it can be passed to
the JAX transforms defined in api.py) and it has to be consumable by update_fun
and get_params.
Example Usage:
.. code-block:: python
  opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
  opt_state = opt_init(params)
  def step(step, opt_state):
    value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state
  for i in range(num_steps):
    value, opt_state = step(i, opt_state)
"""
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
