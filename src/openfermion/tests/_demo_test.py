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
import os
import subprocess
import sys
import tempfile
import unittest

from openfermion.config import THIS_DIRECTORY


class ExampleTest(unittest.TestCase):

    def setUp(self):
        string_length = len(THIS_DIRECTORY)
        directory = THIS_DIRECTORY[:(string_length - 15)] + 'examples/'
        demo_name = 'openfermion_demo.ipynb'
        self.path = directory + demo_name

    def test_demo(self):
        """Execute a notebook via nbconvert and collect output."""

        # Determine if python 2 or 3 is being used.
        version = sys.version_info[0]

        # Run ipython notebook.
        with tempfile.NamedTemporaryFile(suffix='.ipynb') as output_file:
            args = ['jupyter',
                    'nbconvert',
                    '--to',
                    'notebook',
                    '--execute',
                    '--ExecutePreprocessor.timeout=60',
                    '--ExecutePreprocessor.kernel_name=python%i' % version,
                    '--output',
                    output_file.name,
                    self.path]
            subprocess.check_call(args)
            output_file.seek(0)
            nb = nbformat.read(output_file, nbformat.current_nbformat)

        # Parse output and make sure there are no errors.
        errors = [output for cell in nb.cells if "outputs" in cell for
                  output in cell["outputs"] if
                  output.output_type == "error"]
        self.assertEqual(errors, [])
