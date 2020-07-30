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
import numpy as np
import pytest
import cirq

import openfermion


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_crxxyy_unitary(rads):
    np.testing.assert_allclose(
        cirq.unitary(openfermion.CRxxyy(rads)),
        cirq.unitary(cirq.ControlledGate(openfermion.Rxxyy(rads))),
        atol=1e-8)


@pytest.mark.parametrize('rads', [
    2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.1 * np.pi, 0.0, -0.5 * np.pi
])
def test_cryxxy_unitary(rads):
    np.testing.assert_allclose(
        cirq.unitary(openfermion.CRyxxy(rads)),
        cirq.unitary(cirq.ControlledGate(openfermion.Ryxxy(rads))),
        atol=1e-8)
