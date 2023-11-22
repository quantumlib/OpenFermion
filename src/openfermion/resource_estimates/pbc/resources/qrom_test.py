# coverage: ignore
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

from openfermion.resource_estimates.pbc.resources.qrom import QR2, QI2


def test_qr2():
    L = 728
    npp = 182
    bpp = 21
    test_val = QR2(L + 1, npp, bpp)
    assert np.isclose(test_val, 3416)

    L = 56
    npp = 28
    bpp = 91
    test_val = QR2(L + 1, npp, bpp)
    assert np.isclose(test_val, 679)


def test_qi2():
    L1 = 728
    npp = 182
    test_val = QI2(L1 + 1, npp)
    assert np.isclose(test_val, 785)

    L1 = 56
    npp = 28
    test_val = QI2(L1 + 1, npp)
    assert np.isclose(test_val, 88)
