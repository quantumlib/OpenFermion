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

"""This module provides functions to interface with scipy.sparse."""
from __future__ import absolute_import

from functools import reduce
from future.utils import iteritems

import itertools
import multiprocessing
import numpy
import numpy.linalg
import scipy
import scipy.sparse
import scipy.sparse.linalg
import warnings


from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("hello_world.pyx")
)
