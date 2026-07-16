# pylint: disable=wrong-import-position,wrong-import-order
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
import sys

# Ensure src/ is in sys.path so that the OpenFermion utils module can be
# imported at Pytest startup time.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from openfermion.config import set_threading_limits

set_threading_limits()

import random
from typing import Any

import numpy as np
import pytest


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


@pytest.fixture(autouse=True, scope="session")
def set_threadpool_limits():
    """Limit number of threads to prevent oversubscription with pytest-xdist.

    This only has an effect if the Python threadpoolctl package is installed,
    and it only influences parallelism in some numerical libraries used in
    packages such as NumPy.
    """
    try:
        import threadpoolctl  # type: ignore[import-untyped, import-not-found]
    except ImportError:
        yield
        return

    if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
        from openfermion.config import get_available_cpu_count

        # Limit native library thread pools for this worker.
        with threadpoolctl.threadpool_limits(limits=get_available_cpu_count()):
            yield
    else:
        yield


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--skipslow", action="store_true", help="skips slow tests")


def pytest_runtest_setup(item: Any) -> None:
    if "slow" in item.keywords and item.config.getvalue("skipslow"):
        pytest.skip("skipped because of --skipslow option")
