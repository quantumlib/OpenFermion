#!/usr/bin/env bash

# Copyright 2018 The OpenFermion Developers
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

################################################################################
# Produces wheels that can be uploaded to the pypi package repository.
#
# First argument must be the output directory. Second argument is an optional
# version specifier. If not set, the version from `_version.py` is used. If set,
# it overwrites `_version.py`.
#
# Usage:
#     dev_tools/packaging/produce-package.sh output_dir [version]
#
# Note: since around 2021, the use of setup.py to build distributions has been
# deprecated and replaced with using the Python 'build' package, and along with
# this, stricter requirements for versioning were introduced. This means that
# the "version" argument to this script must conform to patterns described at
# https://packaging.python.org/en/latest/discussions/versioning/.
################################################################################

PROJECT_NAME=openfermion
RED=$'\033[31m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'

set -e

if [ -z "$1" ]; then
  echo -e "${RED}No output directory given.${RESET}"
  exit 1
fi

mkdir -p "$1"
out_dir=$(realpath "$1")

SPECIFIED_VERSION="$2"
if [ -n "$SPECIFIED_VERSION" ]; then
    echo "Will set version to $SPECIFIED_VERSION"
fi

# Get the working directory to the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")"
repo_dir=$(git rev-parse --show-toplevel)
cd "$repo_dir"

function confirm() {
    local question="$1"
    read -r -p "$question (y/n) " answer
    [[ "$answer" =~ ^[Yy]$ ]]
}

# Make a clean copy of HEAD, without files ignored by git (but potentially kept
# by setup.py).
if [ -n "$(git status --short)" ]; then
    echo -e "${RED}WARNING: There are uncommitted git changes."
    echo -e "They won't be included in the package.${RESET}"
    # shellcheck disable=SC2310
    if ! confirm "${YELLOW}Proceed anyway?${RESET}"; then
        echo "Stopping."
        exit 1
    fi
fi

tmp_git_dir=$(mktemp -d "/tmp/produce-package-git.XXXXXXXXXXXXXXXX")
trap '{ rm -rf "$tmp_git_dir"; }' EXIT

cd "$tmp_git_dir"
git init --quiet
git fetch "$repo_dir" HEAD --quiet --depth=1
git checkout FETCH_HEAD -b work --quiet

if [ -n "$SPECIFIED_VERSION" ]; then
    echo '__version__ = "'"$SPECIFIED_VERSION"'"' \
         > "$tmp_git_dir/src/$PROJECT_NAME/_version.py"
fi

# Python wheel.
echo -e "${GREEN}Producing Python package files...${RESET}"
python -m build --outdir "$out_dir"

echo -e "${GREEN}Finished – output is in $out_dir${RESET}"
