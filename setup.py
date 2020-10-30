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
import io
import os

from setuptools import setup, find_packages


# This reads the __version__ variable from openfermion/_version.py
__version__ = ''
exec(open('src/openfermion/_version.py').read())

# Readme file as long_description:
long_description = ('===========\n' +
                    'OpenFermion\n' +
                    '===========\n')
stream = io.open('README.rst', encoding='utf-8')
stream.readline()
long_description += stream.read()

# Read in package requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

docs_files_gen = os.walk('docs')
docs_data_files_tuples = []
for cwd, subdirs, files in list(docs_files_gen)[1:]:
    if 'ipynb_checkpoints' in cwd:
        continue
    docs_data_files_tuples.append(
        (os.path.join('openfermion',
                      cwd), [os.path.join(cwd, file) for file in files]))

setup(
    name='openfermion',
    version=__version__,
    author='The OpenFermion Developers',
    author_email='help@openfermion.org',
    url='http://www.openfermion.org',
    description=('The electronic structure package for quantum computers.'),
    long_description=long_description,
    install_requires=requirements,
    license='Apache 2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [
            os.path.join('src', 'openfermion', 'testing', '*.npy'),
            os.path.join('src', 'openfermion', 'testing', '*.hdf5'),
        ],
    },
    data_files=docs_data_files_tuples,
)
