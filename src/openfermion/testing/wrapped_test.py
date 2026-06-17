# Copyright 2026 Google LLC
#
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

import pytest
from unittest import mock
import openfermion.testing.wrapped as wrapped


class MockCirqTesting:
    @property
    def retry_once_with_later_random_values(self):
        raise AttributeError("No such attribute")


class MockCirq:
    testing = MockCirqTesting()


def test_retry_once_fallback_success():
    with mock.patch('openfermion.testing.wrapped.cirq', MockCirq()):

        call_count = 0

        @wrapped.retry_once_with_later_random_values
        def successful_test():
            nonlocal call_count
            call_count += 1
            return "Success"

        assert successful_test() == "Success"
        assert call_count == 1


def test_retry_once_fallback_flaky():
    with mock.patch('openfermion.testing.wrapped.cirq', MockCirq()):

        call_count = 0

        @wrapped.retry_once_with_later_random_values
        def flaky_test():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AssertionError("Failed first time")
            return "Success"

        with pytest.warns(
            UserWarning, match="Retrying in case we got a failing seed from pytest-randomly."
        ):
            assert flaky_test() == "Success"

        assert call_count == 2


def test_retry_once_fallback_failure():
    with mock.patch('openfermion.testing.wrapped.cirq', MockCirq()):

        call_count = 0

        @wrapped.retry_once_with_later_random_values
        def failing_test():
            nonlocal call_count
            call_count += 1
            raise AssertionError("Failed both times")

        with pytest.warns(
            UserWarning, match="Retrying in case we got a failing seed from pytest-randomly."
        ):
            with pytest.raises(AssertionError, match="Failed both times"):
                failing_test()

        assert call_count == 2
