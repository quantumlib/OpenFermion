#!/bin/bash

#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

set -ex

# https://stackoverflow.com/q/59895
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
readonly SCRIPT_DIR

# We use docker to run the actual `pip-compile` command because its
# behavior depends on the platform from which it is run. Please see
# ./Dockerfile for the list of `pip-compile` commands that are run.
docker build -t openfermion-pip-compile "$SCRIPT_DIR"

# Create a container from the image so we can copy out the outputs
container_id=$(docker create openfermion-pip-compile)

# Copy out the files and organize them
docker cp "$container_id:/pip-compile/envs" "$SCRIPT_DIR/"

# Clean up
docker rm -v "$container_id"


# Do it again for `max_compat`.
# Set the docker build args; use a unique image name; use the correct output directory.
docker build -t openfermion-pip-compile-max-compat --build-arg='PYTHON_VERSION=3.11' --build-arg='PLATFORM=max_compat' "$SCRIPT_DIR"
container_id=$(docker create openfermion-pip-compile-max-compat)
docker cp "$container_id:/pip-compile/max_compat" "$SCRIPT_DIR/"
docker rm -v "$container_id"
