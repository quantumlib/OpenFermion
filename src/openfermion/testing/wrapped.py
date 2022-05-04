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

from typing import Any, Dict, Optional, Sequence, Type, Union

import sympy

import cirq

_setup_code = ("import cirq\n"
               "import numpy as np\n"
               "import sympy\n"
               "import openfermion\n")


def assert_equivalent_repr(value: Any, *,
                           setup_code: str = _setup_code) -> None:
    """Checks that eval(repr(v)) == v.

    Args:
        value: A value whose repr should be evaluatable python
            code that produces an equivalent value.
        setup_code: Code that must be executed before the repr can be evaluated.
            Ideally this should just be a series of 'import' lines.
    """
    cirq.testing.assert_equivalent_repr(value, setup_code=setup_code)


def assert_implements_consistent_protocols(
        val: Any,
        *,
        exponents: Sequence[Any] = (0, 1, -1, 0.5, 0.25, -0.5, 0.1,
                                    sympy.Symbol("s")),
        qubit_count: Optional[int] = None,
        ignoring_global_phase: bool = False,
        setup_code: str = _setup_code,
        global_vals: Optional[Dict[str, Any]] = None,
        local_vals: Optional[Dict[str, Any]] = None) -> None:
    """Checks that a value is internally consistent and has a good __repr__."""
    try:
        # Cirq 0.14
        cirq.testing.assert_implements_consistent_protocols(
            val,
            exponents=exponents,
            qubit_count=qubit_count,
            ignoring_global_phase=ignoring_global_phase,
            ignore_decompose_to_default_gateset=True,
            setup_code=setup_code,
            global_vals=global_vals,
            local_vals=local_vals,
        )
    except TypeError:  # coverage: ignore
        # Cirq 0.12
        cirq.testing.assert_implements_consistent_protocols(  # coverage: ignore
            val,  # coverage: ignore
            exponents=exponents,  # coverage: ignore
            qubit_count=qubit_count,  # coverage: ignore
            ignoring_global_phase=ignoring_global_phase,  # coverage: ignore
            setup_code=setup_code,  # coverage: ignore
            global_vals=global_vals,  # coverage: ignore
            local_vals=local_vals,  # coverage: ignore
        )  # coverage: ignore
