#!/bin/bash
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

set -o errexit
set -o nounset

declare -r usage="Usage: ${0} [-h] [UV_OPTIONS]
Generate environment files for OpenFermion development using uv's
'universal' option, making the result compatible with multiple
Python versions. The output is written to subdirectories under
dev_tools/requirements/.

Options:
  -h   Show this help message and exit

All other options on the command line will be passed directly to
'uv pip compile'. Run 'uv pip compile --help' to learn about the
options available."

while [[ $# -gt 0 ]]; do
    case "${1}" in
        -h|--help) echo "${usage}"; exit 0 ;;
        *) break ;;
    esac
done

# Go to the top of the local TFQ git tree. Do it early in case this fails.
script_dir=$(CDPATH="" cd -- "$(dirname -- "${0}")" && pwd -P)
repo_dir=$(git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null)
cd "${repo_dir}"

mkdir -p dev_tools/requirements/envs dev_tools/requirements/max_compat

# ~~~~ Generate basic requirements files ~~~~

uv pip compile "$@" \
    -o dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/format.txt \
    dev_tools/requirements/deps/mypy.txt \
    dev_tools/requirements/deps/packaging.txt \
    dev_tools/requirements/deps/pylint.txt \
    dev_tools/requirements/deps/pytest.txt \
    dev_tools/requirements/deps/resource_estimates_runtime.txt \
    dev_tools/requirements/deps/runtime.txt \
    dev_tools/requirements/deps/shellcheck.txt

uv pip compile "$@" \
    -o dev_tools/requirements/envs/format.env.txt \
    -c dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/format.txt \
    dev_tools/requirements/deps/runtime.txt

uv pip compile "$@" \
    -o dev_tools/requirements/envs/pylint.env.txt \
    -c dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/pylint.txt \
    dev_tools/requirements/deps/runtime.txt

uv pip compile "$@" \
    -o dev_tools/requirements/envs/pytest.env.txt \
    -c dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/pytest.txt \
    dev_tools/requirements/deps/runtime.txt

uv pip compile "$@" \
    -o dev_tools/requirements/envs/pytest-extra.env.txt \
    -c dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/pytest.txt \
    dev_tools/requirements/deps/resource_estimates_runtime.txt \
    dev_tools/requirements/deps/runtime.txt

uv pip compile "$@" \
    -o dev_tools/requirements/envs/mypy.env.txt \
    -c dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/mypy.txt \
    dev_tools/requirements/deps/runtime.txt

uv pip compile "$@" \
    -o dev_tools/requirements/envs/shellcheck.env.txt \
    -c dev_tools/requirements/envs/dev.env.txt \
    dev_tools/requirements/deps/shellcheck.txt

# ~~~~ Generate max_compat files ~~~~

uv pip compile "$@" \
    -o dev_tools/requirements/max_compat/pytest-max-compat.env.txt \
    -c dev_tools/requirements/deps/oldest-versions.txt \
    dev_tools/requirements/deps/pytest.txt \
    dev_tools/requirements/deps/runtime.txt
