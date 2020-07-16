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

import numpy
import nbformat


class ExamplesTest(unittest.TestCase):

    def setUp(self):

        self.examples_folder = os.path.join(
            os.path.dirname(__file__),  # Start at this file's directory.
        )

    def test_performance_benchmarks(self):
        """Unit test for examples/performance_benchmark.py."""

        # Import performance benchmarks and seed random number generator.
        sys.path.append(self.examples_folder)
        from performance_benchmarks import (
            run_diagonal_commutator,
            run_fermion_math_and_normal_order,
            run_jordan_wigner_sparse,
            run_molecular_operator_jordan_wigner,
            run_linear_qubit_operator,
        )
        numpy.random.seed(1)

        runtime_upper_bound = 600

        # Run diagonal commutator benchmark
        runtime_standard, runtime_diagonal = run_diagonal_commutator()
        self.assertLess(runtime_standard, runtime_upper_bound)
        self.assertLess(runtime_diagonal, runtime_upper_bound)

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
        print("Examples folder ", self.examples_folder)
        for filename in os.listdir(self.examples_folder):
            if not filename.endswith('.ipynb'):
                continue

            path = os.path.join(self.examples_folder, filename)
            notebook = nbformat.read(path, nbformat.NO_CONVERT)
            state = {}

            for cell in notebook.cells:
                if cell.cell_type == 'code' and not is_matplotlib_cell(cell):
                    try:
                        exec(strip_magics_and_shows(cell.source), state)
                    # coverage: ignore
                    except:
                        print('Failed to run {}.'.format(path))
                        raise


def is_matplotlib_cell(cell):
    return "%matplotlib" in cell.source


def strip_magics_and_shows(text):
    """Remove Jupyter magics and pyplot show commands."""
    lines = [line for line in text.split('\n')
             if not contains_magic_or_show(line)]
    return '\n'.join(lines)


def contains_magic_or_show(line):
    return (line.strip().startswith('%') or
            'pyplot.show(' in line or
            'plt.show(' in line)
