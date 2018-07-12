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

"""Tests the code in the examples directory of the git repo."""

import os
import re
import sys
import unittest

import nbformat
import numpy


class ExamplesTest(unittest.TestCase):

    def setUp(self):

        self.examples_folder = os.path.join(
                os.path.dirname(__file__),  # Start at this file's directory.
                '..', 'tests',  # Hacky check that we're under tests/.
                '..', '..', '..', 'examples')

    def test_performance_benchmarks(self):
        """Unit test for examples/performance_benchmark.py."""

        # Import performance benchmarks and seed random number generator.
        sys.path.append(self.examples_folder)
        from performance_benchmarks import (
            run_fermion_math_and_normal_order,
            run_jordan_wigner_sparse,
            run_molecular_operator_jordan_wigner,
            run_linear_qubit_operator,
        )
        numpy.random.seed(1)

        runtime_upper_bound = 600

        # Run InteractionOperator.jordan_wigner_transform() benchmark.
        runtime = run_molecular_operator_jordan_wigner(n_qubits=10)
        self.assertLess(runtime, runtime_upper_bound)

        # Run benchmark on FermionOperator math and normal-ordering.
        runtime_math, runtime_normal = run_fermion_math_and_normal_order(
            n_qubits=10, term_length=5, power=5)
        self.assertLess(runtime_math, runtime_upper_bound)
        self.assertLess(runtime_normal, runtime_upper_bound)

        # Run FermionOperator.jordan_wigner_sparse() benchmark.
        runtime = run_jordan_wigner_sparse(n_qubits=10)
        self.assertLess(runtime, 600)

        # Run (Parallel)LinearQubitOperator benchmark.
        runtime_sequential, runtime_parallel = run_linear_qubit_operator(
            n_qubits=10, n_terms=10, processes=10)
        self.assertLess(runtime_sequential, runtime_upper_bound)
        self.assertLess(runtime_parallel, runtime_upper_bound)

    def test_can_run_examples_jupyter_notebooks(self):
        for filename in os.listdir(self.examples_folder):
            if not filename.endswith('.ipynb'):
                continue
            path = os.path.join(self.examples_folder, filename)
            self.assert_jupyter_notebook_has_working_code_cells(path)

    def assert_jupyter_notebook_has_working_code_cells(self, path):
        """Checks that code cells in a Jupyter notebook actually run in
        sequence.

        State is kept between code cells. Imports and variables defined in one
        cell will be visible in later cells.
        """

        notebook = nbformat.read(path, nbformat.NO_CONVERT)
        state = {}

        for cell in notebook.cells:
            if cell.cell_type == 'code':
                self.assert_code_cell_runs_and_prints_expected(cell, state)

    def assert_code_cell_runs_and_prints_expected(self, cell, state):
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

        actual_lines = [canonicalize_printed_line(line) for line in
                        output_lines]

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
    for match in re.finditer(r"[0-9]+\.[0-9]+(e-[0-9]+)?", line):
        start = match.start()
        end = match.end()
        result.append(line[prev_end:start])
        result.append(format(round(float(line[start:end]), 4), '.4f'))
        prev_end = end
    result.append(line[prev_end:])
    return ''.join(result).rstrip()
