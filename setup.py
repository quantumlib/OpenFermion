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

import os
import runpy

from setuptools import find_packages, setup

# This reads the __version__ variable from openfermion/_version.py
__version__ = runpy.run_path('src/openfermion/_version.py')['__version__']
assert __version__, 'Version string cannot be empty'

# The readme file is used as the long_description:
long_description = '===========\n' + 'OpenFermion\n' + '===========\n\n'
with open('README.rst', 'r', encoding='utf-8') as readme:
    long_description += readme.read()

# Read in package requirements.txt.
with open('dev_tools/requirements/deps/runtime.txt') as r:
    requirements = r.readlines()
requirements = [r.strip() for r in requirements]
requirements = [r for r in requirements if not r.startswith('#')]

# Read in resource estimates requirements.
with open('dev_tools/requirements/deps/resource_estimates_runtime.txt') as r:
    resource_requirements = r.readlines()
resource_requirements = [r.strip() for r in resource_requirements]
resource_requirements = [r for r in resource_requirements if not r.startswith('#')]

setup(
    name='openfermion',
    version=__version__,
    url='https://quantumai.google/openfermion',
    license='Apache 2',
    author='The OpenFermion Developers',
    author_email='openfermion-dev@googlegroups.com',
    maintainer='Google Quantum AI open-source maintainers',
    maintainer_email='quantum-oss-maintainers@google.com',
    description=('The electronic structure package for quantum computers.'),
    long_description=long_description,
    python_requires='>=3.10.0',
    install_requires=requirements,
    extras_require={'resources': resource_requirements},
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [
            os.path.join('src', 'openfermion', 'resource_estimates', 'integrals', '*.h5'),
            os.path.join('src', 'openfermion', 'testing', '*.npy'),
            os.path.join('src', 'openfermion', 'testing', '*.hdf5'),
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Quantum Computing',
    ],
    keywords=[
        'algorithms',
        'api',
        'application programming interface',
        'chemistry',
        'cirq',
        'electronic structure',
        'fermion',
        'fermionic systems',
        'google quantum',
        'google',
        'hamiltonians',
        'high performance',
        'nisq',
        'noisy intermediate-scale quantum',
        'python',
        'quantum algorithms',
        'quantum chemistry',
        'quantum circuit simulator',
        'quantum circuit',
        'quantum computer simulator',
        'quantum computing',
        'quantum development kit',
        'quantum programming language',
        'quantum programming',
        'quantum simulation',
        'quantum',
        'qubit hamiltonians',
        'qubit',
        'sdk',
        'simulation',
        'software development kit',
    ],
)
