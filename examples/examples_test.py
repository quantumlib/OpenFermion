# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import nbformat
import pytest


def find_examples_jupyter_notebook_paths():
    examples_folder = os.path.dirname(__file__)
    for filename in os.listdir(examples_folder):
        if not filename.endswith('.ipynb'):
            continue
        yield os.path.join(examples_folder, filename)


@pytest.mark.parametrize('path', find_examples_jupyter_notebook_paths())
def test_can_run_examples_jupyter_notebook(path):
    assert_jupyter_notebook_has_working_code_cells(path)


def assert_jupyter_notebook_has_working_code_cells(path):
    """Checks that code cells in a Jupyter notebook actually run in sequence.

    State is kept between code cells. Imports and variables defined in one
    cell will be visible in later cells.
    """

    notebook = nbformat.read(path, nbformat.NO_CONVERT)
    state = {}

    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert_code_cell_runs_and_prints_expected(cell, state)


def assert_code_cell_runs_and_prints_expected(cell, state):
    """Executes a code cell and compares captured output to saved output."""

    if cell.outputs and hasattr(cell.outputs[0], 'text'):
        expected_outputs = cell.outputs[0].text.strip().split('\n')
    else:
        expected_outputs = ['']
    expected_lines = [canonicalize_printed_line(line)
                      for line in expected_outputs]

    output_lines = []

    def print_capture(*values, sep=' '):
        output_lines.extend(
                sep.join(str(e) for e in values).split('\n'))

    state['print'] = print_capture
    exec(strip_magics_and_shows(cell.source), state)
    output_lines = '\n'.join(output_lines).strip().split('\n')

    actual_lines = [canonicalize_printed_line(line) for line in output_lines]

    assert len(actual_lines) == len(expected_lines)

    for i, line in enumerate(actual_lines):
        assert line == expected_lines[i]


def strip_magics_and_shows(text):
    """Remove Jupyter magics and pyplot show commands."""
    lines = [line for line in text.split('\n')
             if not contains_magic_or_show(line)]
    return '\n'.join(lines)


def contains_magic_or_show(line) -> str:
    return (line.strip().startswith('%') or
            'pyplot.show(' in line or
            'plt.show(' in line)


def canonicalize_printed_line(line):
    """Replace highly precise numbers with less precise numbers.

    This allows the test to pass on machines that perform arithmetic slightly
    differently.

    Args:
        line: The line to canonicalize.

    Returns:
        The canonicalized line.
    """
    prev_end = 0
    result = []
    for match in re.finditer(r"[0-9]\.[0-9]+", line):
        start = match.start()
        end = match.end()
        result.append(line[prev_end:start])
        result.append(format(round(float(line[start:end]), 4), '.4f'))
        prev_end = end
    result.append(line[prev_end:])
    return ''.join(result).rstrip()


def test_canonicalize_printed_line():
    x = 'first 20.37378859061888 then 20.37378627319067'
    assert canonicalize_printed_line(x) == 'first 20.3738 then 20.3738'
