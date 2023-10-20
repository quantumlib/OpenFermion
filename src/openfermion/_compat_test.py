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
from cirq.testing import assert_deprecated

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
    def decorated_test(*args, **kwargs) -> Any:
        with pytest.deprecated_call():
            test(*args, **kwargs)

    return decorated_test


@deprecation.deprecated()
def f():
    pass


def test_deprecated_test():
    @deprecated_test
    def test():
        f()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warnings.simplefilter('default', DeprecationWarning)
        test()
        assert len(w) == 0


def test_wrap_module():
    openfermion.deprecated_attribute = None
    wrapped_openfermion = wrap_module(openfermion, {'deprecated_attribute': ('', '')})
    with pytest.deprecated_call():
        _ = wrapped_openfermion.deprecated_attribute


def test_cirq_deprecations():
    @deprecated(deadline="v0.12", fix="use new_func")
    def old_func():
        pass

    with pytest.raises(ValueError, match="deprecated .* not allowed"):
        old_func()
