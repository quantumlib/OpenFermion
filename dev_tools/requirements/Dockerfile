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

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}

WORKDIR /pip-compile

# Step 0: install pip-tools
COPY envs/pip-tools.env.txt ./
RUN pip install -r pip-tools.env.txt

# Step 1: compile a complete & consistent environment with all dependencies
COPY deps/ ./deps/
COPY run-pip-compiles.py ./

ARG PLATFORM="default"
RUN python run-pip-compiles.py --platform $PLATFORM