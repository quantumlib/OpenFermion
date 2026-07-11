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

import numbers
import os

# Tolerance to consider number zero.
EQ_TOLERANCE = 1e-8

# Numeric types accepted as operator and tensor coefficients. numbers.Number
# covers Python and NumPy scalar types alike (NumPy scalars are not subclasses
# of the built-in int/float/complex types).
COEFFICIENT_TYPES = (int, float, complex, numbers.Number)

# Molecular data directory.
THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.realpath(os.path.join(THIS_DIRECTORY, 'testing/data'))


def get_available_cpu_count() -> int:
    """Returns the number of CPU cores available to the current process.

    This function respects active CPU limits such as process affinity,
    Docker container limits, and pytest-xdist worker distribution to
    prevent oversubscription.
    """
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        cpus = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):  # Unix/Linux
        try:
            cpus = len(os.sched_getaffinity(0))
        except Exception:  # pylint: disable=broad-exception-caught
            cpus = os.cpu_count() or 1
    else:  # Fallback for older Python on Windows/macOS
        cpus = os.cpu_count() or 1

    # Divide by the number of active pytest-xdist workers.
    if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
        try:
            worker_count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
            if worker_count > 0:
                cpus = max(1, cpus // worker_count)
        except ValueError:
            pass

    return cpus


def set_threading_limits() -> None:
    """Sets thread limits for underlying numerical libraries (MKL, OpenBLAS, OMP).

    This prevents libraries like NumPy, SciPy, and JAX from oversubscribing
    available CPU resources on high-core-count or multi-worker environments.
    """
    if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
        limit = str(max(get_available_cpu_count(), 1))
    else:
        limit = str(max(get_available_cpu_count() - 1, 1))
    for var in ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = limit
