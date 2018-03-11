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
import nbformat
import numpy
import os
import subprocess
import sys
import tempfile
import unittest

from openfermion.config import THIS_DIRECTORY


class ExampleTest(unittest.TestCase):

    def setUp(self):
        string_length = len(THIS_DIRECTORY)
        self.directory = THIS_DIRECTORY[:(string_length - 15)] + 'examples/'
        self.demo_name = 'openfermion_demo.ipynb'
        self.binary_code_transforms_demo = 'binary_code_transforms_demo.ipynb'
        self.givens_rotations_demo_name = 'givens_rotations.ipynb'

    def test_demo(self):
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
                nb = nbformat.read(output_file, nbformat.current_nbformat)

            # Parse output and make sure there are no errors.
            errors = [output for cell in nb.cells if "outputs" in cell for
                      output in cell["outputs"] if
                      output.output_type == "error"]
        else:
            errors = []
        self.assertEqual(errors, [])

    def test_binary_code_transforms_demo(self):
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
                nb = nbformat.read(output_file, nbformat.current_nbformat)

            # Parse output and make sure there are no errors.
            errors = [output for cell in nb.cells if "outputs" in cell for
                      output in cell["outputs"] if
                      output.output_type == "error"]
        else:
            errors = []
        self.assertEqual(errors, [])

    def test_performance_benchmarks(self):

        # Import performance benchmarks and seed random number generator.
        sys.path.append(self.directory)
        from performance_benchmarks import (
            benchmark_fermion_math_and_normal_order,
            benchmark_jordan_wigner_sparse,
            benchmark_molecular_operator_jordan_wigner)
        numpy.random.seed(1)

        # Run InteractionOperator.jordan_wigner_transform() benchmark.
        n_qubits = 10
        runtime = benchmark_molecular_operator_jordan_wigner(n_qubits)
        self.assertLess(runtime, 600)

        # Run benchmark on FermionOperator math and normal-ordering.
        n_qubits = 10
        term_length = 5
        power = 5
        runtime_math, runtime_normal = benchmark_fermion_math_and_normal_order(
            n_qubits, term_length, power)
        self.assertLess(runtime_math, 600)
        self.assertLess(runtime_normal, 600)

        # Run FermionOperator.jordan_wigner_sparse() benchmark.
        n_qubits = 10
        runtime = benchmark_jordan_wigner_sparse(n_qubits)
        self.assertLess(runtime, 600)
