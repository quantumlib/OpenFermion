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

import random

import numpy as np
import pytest

import openfermion
from openfermion.testing import random_interaction_operator_term


@pytest.mark.parametrize('order,real,seed', [(k, r, random.randrange(2 << 30))
                                             for k in [1, 2, 3, 4, 5]
                                             for r in [0, 1] for _ in range(5)])
def test_random_interaction_operator_term(order, real, seed):
    op = random_interaction_operator_term(order, real, seed)

    assert openfermion.is_hermitian(op)

    assert op.constant == 0
    assert op.one_body_tensor.shape == (order,) * 2
    assert op.two_body_tensor.shape == (order,) * 4

    for tensor in (op.one_body_tensor, op.two_body_tensor):
        for indices in np.argwhere(tensor):
            assert len(set(indices)) == order

    op_2 = random_interaction_operator_term(order, real, seed)
    assert op == op_2

    if order == 1:
        assert op.one_body_tensor != 0
        assert op.two_body_tensor != 0
    elif order == 2:
        assert np.all((op.one_body_tensor == 0) == np.eye(2))
    elif order == 3:
        assert np.all(op.one_body_tensor == 0)
    elif order == 4:
        assert np.all(op.one_body_tensor == 0)
    else:
        assert np.all(op.one_body_tensor == 0)
        assert np.all(op.two_body_tensor == 0)
