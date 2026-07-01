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

import re
from typing import Any

_SLOW_MARKER_RE = re.compile(r"(?<![\w:])slow(?![\w:])")


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow")


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    markexpr = config.getoption("-m", default="")
    if config.getoption("--run-slow") or _SLOW_MARKER_RE.search(markexpr):
        return

    selected = []
    deselected = []
    for item in items:
        if "slow" in item.keywords:
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
