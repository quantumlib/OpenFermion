# Copyright 2025 Google LLC
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

import importlib.util
import pathlib
from typing import Any

_CONFTEST_PATH = pathlib.Path(__file__).parents[3] / "conftest.py"
_CONFTEST_SPEC = importlib.util.spec_from_file_location("openfermion_pytest_config", _CONFTEST_PATH)
assert _CONFTEST_SPEC is not None
assert _CONFTEST_SPEC.loader is not None
conftest = importlib.util.module_from_spec(_CONFTEST_SPEC)
_CONFTEST_SPEC.loader.exec_module(conftest)


class _FakeHook:
    def __init__(self) -> None:
        self.deselected: list[Any] = []

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected = list(items)


class _FakeConfig:
    def __init__(self, *, run_slow: bool = False, markexpr: str = "") -> None:
        self._run_slow = run_slow
        self._markexpr = markexpr
        self.hook = _FakeHook()

    def getoption(self, name: str, default: Any = None) -> Any:
        if name == "--run-slow":
            return self._run_slow
        if name == "-m":
            return self._markexpr
        return default


class _FakeItem:
    def __init__(self, *, slow: bool = False) -> None:
        self.keywords = {"slow": True} if slow else {}


def test_fake_config_returns_default_for_unknown_options() -> None:
    config = _FakeConfig()

    assert config.getoption("--unknown", default="fallback") == "fallback"


def test_default_collection_skips_slow_tests() -> None:
    config = _FakeConfig()
    slow_item = _FakeItem(slow=True)
    fast_item = _FakeItem()
    items = [slow_item, fast_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [fast_item]
    assert config.hook.deselected == [slow_item]


def test_run_slow_keeps_slow_tests() -> None:
    config = _FakeConfig(run_slow=True)
    slow_item = _FakeItem(slow=True)
    fast_item = _FakeItem()
    items = [slow_item, fast_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [slow_item, fast_item]
    assert config.hook.deselected == []


def test_explicit_slow_mark_expression_keeps_slow_tests() -> None:
    config = _FakeConfig(markexpr="slow")
    slow_item = _FakeItem(slow=True)
    fast_item = _FakeItem()
    items = [slow_item, fast_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [slow_item, fast_item]
    assert config.hook.deselected == []


def test_unrelated_mark_expression_keeps_default_slow_filter() -> None:
    config = _FakeConfig(markexpr="not slowpoke")
    slow_item = _FakeItem(slow=True)
    fast_item = _FakeItem()
    items = [slow_item, fast_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [fast_item]
    assert config.hook.deselected == [slow_item]
