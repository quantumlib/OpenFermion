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

from typing import Any, Callable
import functools
import warnings

import pytest
import deprecation

from cirq._compat import deprecated

import openfermion
from openfermion._compat import wrap_module


def deprecated_test(test: Callable) -> Callable:
    """Marks a test as using deprecated functionality.

    Ensures the test is executed within the `pytest.deprecated_call()` context.

    Args:
        test: The test.

    Returns:
        The decorated test.
    """

    @functools.wraps(test)
    def decorated_test(*args: Any, **kwargs: Any) -> Any:
        with pytest.deprecated_call():
            test(*args, **kwargs)

    return decorated_test


@deprecation.deprecated()
def f() -> None:
    pass


def test_deprecated_test() -> None:
    @deprecated_test
    def test() -> None:
        f()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warnings.simplefilter('default', DeprecationWarning)
        test()
        assert len(w) == 0


def test_wrap_module() -> None:
    openfermion.deprecated_attribute = None  # type: ignore
    wrapped_openfermion = wrap_module(openfermion, {'deprecated_attribute': ('', '')})
    with pytest.deprecated_call():
        _ = wrapped_openfermion.deprecated_attribute  # type: ignore


def test_cirq_deprecations() -> None:
    @deprecated(deadline="v0.12", fix="use new_func")
    def old_func() -> None:
        pass

    with pytest.raises(ValueError, match="deprecated .* not allowed"):
        old_func()
