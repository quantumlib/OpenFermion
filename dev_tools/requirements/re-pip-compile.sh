#!/bin/bash

# This script calls pip-compile for each environment.
# It forwards any command-line arguments to each call to `pip-compile`.
# Pass `--upgrade` to upgrade all dependencies; see `pip-compile --help` for more info.

pip-compile $@ --output-file=runtime.env.txt --resolver=backtracking deps/runtime.txt

pip-compile $@ --output-file=format.env.txt --resolver=backtracking deps/format.txt    runtime.env.txt
pip-compile $@ --output-file=pylint.env.txt --resolver=backtracking deps/pylint.txt    runtime.env.txt
pip-compile $@ --output-file=pytest.env.txt --resolver=backtracking deps/pytest.txt    runtime.env.txt
pip-compile $@ --output-file=mypy.env.txt   --resolver=backtracking deps/mypy.txt      runtime.env.txt
pip-compile $@ --output-file=dev.env.txt    --resolver=backtracking deps/dev-tools.txt runtime.env.txt
pip-compile $@ --output-file=dev.env.txt    --resolver=backtracking deps/resource_estimates.txt pytest.env.txt
