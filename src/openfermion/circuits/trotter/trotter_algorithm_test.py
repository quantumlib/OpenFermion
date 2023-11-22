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

import pytest

from openfermion import FermionOperator
from openfermion.circuits.trotter import TrotterStep


def test_trotter_algorithm_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = TrotterStep(FermionOperator())


def test_trotter_algorithm_is_abstract_must_implement():
    class Missing(TrotterStep):
        pass

    with pytest.raises(TypeError):
        _ = Missing(FermionOperator())


def test_trotter_algorithm_is_abstract_can_implement():
    class Included(TrotterStep):
        def trotter_step(self, qubits, time, control_qubit):
            pass

    assert isinstance(Included(FermionOperator()), TrotterStep)
