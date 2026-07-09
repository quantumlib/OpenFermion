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

import os
import random
import sys
from typing import Any

import numpy as np
import pytest

# Ensure src/ is in sys.path so that the OpenFermion utils module can be
# imported at Pytest startup time.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def pytest_configure(config: Any) -> None:
    # Set seeds for collection-time parameterization.
    random.seed(0)
    np.random.seed(0)

    os.environ['CIRQ_TESTING'] = "true"


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set a fixed random seed when testing."""
    random.seed(0)
    np.random.seed(0)
