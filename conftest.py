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


@pytest.fixture(autouse=True, scope="session")
def set_threadpool_limits():
    """Limit number of threads to prevent oversubscription with pytest-xdist.

    This only has an effect if the Python threadpoolctl package is installed,
    and it only influences parallellism in some numerical libraries used in
    packages such as NumPy.
    """
    try:
        import threadpoolctl  # type: ignore[import-untyped]
    except ImportError:
        yield
        return

    if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
        from openfermion.utils import get_available_cpu_count

        try:
            n_workers = max(1, int(os.environ["PYTEST_XDIST_WORKER_COUNT"]))
        except ValueError:
            n_workers = 1
        max_threads_per_worker = max(1, get_available_cpu_count() // n_workers)
        # Limit native library thread pools for this worker.
        with threadpoolctl.threadpool_limits(limits=max_threads_per_worker):
            yield
    else:
        yield
