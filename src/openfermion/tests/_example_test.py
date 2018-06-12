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
import subprocess
import sys
import tempfile
import unittest

import nbformat
import numpy

from openfermion.config import THIS_DIRECTORY


class ExampleTest(unittest.TestCase):
    """Unit tests for example scripts."""

    def setUp(self):
        string_length = len(THIS_DIRECTORY)
        self.directory = THIS_DIRECTORY[:(string_length - 15)] + 'examples/'
        self.demo_name = 'openfermion_tutorial.ipynb'
        self.binary_code_transforms_demo = 'binary_code_transforms_demo.ipynb'
        self.jw_bk_demo = 'jordan_wigner_and_bravyi_kitaev_transforms.ipynb'

    def test_demo(self):
        """Unit test for demo."""
        # Determine if python 2 or 3 is being used.
        major_version, minor_version = sys.version_info[:2]
        if major_version == 2 or minor_version == 6:
            version = str(major_version)

            # Run ipython notebook via nbconvert and collect output.
            with tempfile.NamedTemporaryFile(suffix='.ipynb') as output_file:
                args = ['jupyter',
                        'nbconvert',
                        '--to',
                        'notebook',
                        '--execute',
                        '--ExecutePreprocessor.timeout=600',
                        '--ExecutePreprocessor.kernel_name=python{}'.format(
                            version),
                        '--output',
                        output_file.name,
                        self.directory + self.demo_name]
                subprocess.check_call(args)
                output_file.seek(0)
                nb_data = nbformat.read(output_file, nbformat.current_nbformat)

            # Parse output and make sure there are no errors.
            errors = [output for cell in nb_data.cells if "outputs" in cell for
                      output in cell["outputs"] if
                      output.output_type == "error"]
        else:
            errors = []
        self.assertEqual(errors, [])

    def test_binary_code_transforms_demo(self):
        """Unit test for demo."""
        # Determine if python 2 or 3 is being used.
        major_version, minor_version = sys.version_info[:2]
        if major_version == 2 or minor_version == 6:
            version = str(major_version)

            # Run ipython notebook via nbconvert and collect output.
            with tempfile.NamedTemporaryFile(suffix='.ipynb') as output_file:
                args = ['jupyter',
                        'nbconvert',
                        '--to',
                        'notebook',
                        '--execute',
                        '--ExecutePreprocessor.timeout=600',
                        '--ExecutePreprocessor.kernel_name=python{}'.format(
                            version),
                        '--output',
                        output_file.name,
                        self.directory + self.binary_code_transforms_demo]
                subprocess.check_call(args)
                output_file.seek(0)
                nb_data = nbformat.read(output_file, nbformat.current_nbformat)

            # Parse output and make sure there are no errors.
            errors = [output for cell in nb_data.cells if "outputs" in cell for
                      output in cell["outputs"] if
                      output.output_type == "error"]
        else:
            errors = []
        self.assertEqual(errors, [])

    def test_jordan_wigner_and_bravyi_kitaev_transforms_demo(self):
        """Unit test for demo."""
        # Determine if python 2 or 3 is being used.
        major_version, minor_version = sys.version_info[:2]
        if major_version == 2 or minor_version == 6:
            version = str(major_version)

            # Run ipython notebook via nbconvert and collect output.
            with tempfile.NamedTemporaryFile(suffix='.ipynb') as output_file:
                args = ['jupyter',
                        'nbconvert',
                        '--to',
                        'notebook',
                        '--execute',
                        '--ExecutePreprocessor.timeout=600',
                        '--ExecutePreprocessor.kernel_name=python{}'.format(
                            version),
                        '--output',
                        output_file.name,
                        self.directory + self.jw_bk_demo]
                subprocess.check_call(args)
                output_file.seek(0)
                nb_data = nbformat.read(output_file, nbformat.current_nbformat)

            # Parse output and make sure there are no errors.
            errors = [output for cell in nb_data.cells if "outputs" in cell for
                      output in cell["outputs"] if
                      output.output_type == "error"]
        else:
            errors = []
        self.assertEqual(errors, [])

    def test_performance_benchmarks(self):
        """Unit test for examples/performance_benchmark.py."""

        # Import performance benchmarks and seed random number generator.
        sys.path.append(self.directory)
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
